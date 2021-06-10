"""
Keras implementation for Deep Embedded Clustering (GMAE) algorithm:

        Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016.

Usage:
    use `python GMAE.py -h` for help.

Author:
    Xifeng Guo. 2017.1.30
"""

from time import time, ctime
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling, Constant
from keras import regularizers
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import metrics
import math, imageio, random

# tt = ctime().replace(' ','-')

LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (3, 1e-3),
    (5, 1e-5),
    (7, 1e-7),
]


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon


def xent_loss_linear_layer(args):
    input, output = args
    # reconstruction loss
    xent_loss = 0.5 * K.mean((input - output) ** 2, 1)
    return xent_loss


def xent_loss_sigmoid_layer(args):
    input, output = args
    # reconstruction loss
    xent_loss = K.mean(-input*K.log(output+K.epsilon())-(1-input)*K.log(1-output+K.epsilon()), 1)
    return xent_loss


def kl_loss_layer(args):
    z_prior_mean_2, y = args
    # kl loss based on GMM
    kl_loss = 0.5 * (z_prior_mean_2)
    kl_loss = K.sum(K.batch_dot(K.expand_dims(y, 1), kl_loss), 1)
    return kl_loss


def mi_loss_layer(args):
    y = args
    # MI loss
    cat_loss = - K.mean(y * K.log(y + K.epsilon()), 0)
    y_ = K.mean(y, 0)
    h_loss = - y_ * K.log(y_ + K.epsilon())
    mi_loss = cat_loss - h_loss
    return K.sum(mi_loss)


def GMM_update(y, z, u):
    pi = np.sum(y,axis=0)/np.shape(y)[0]
    u = (np.dot(y.T,z).T/np.sum(y,axis=0)).T
    z_3 = np.expand_dims(z,1)
    u_3 = np.expand_dims(u,0)
    y_3 = np.expand_dims(y,2)
    var = (np.sum(y_3*(z_3-u_3)**2,axis=0).T/np.sum(y,axis=0)).T
    # regularize var
    # var = 10*(var.T/np.sum(var, 1)).T
    # var = var/var.mean()
    return pi, u, var


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr


def autoencoder(dims, act='relu', out_act='linear', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='embedding')(h)  # hidden layer, features are extracted from here
    log_var = Dense(dims[-1], kernel_initializer=init, name='log_var')(h)  # log_var layer
    # z = Lambda(sampling, output_shape=(dims[-1],))([h, log_var])

    # y = h
    dx = Input(shape=(dims[-1],), name='decoder_input')
    y = dx
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], activation=out_act, kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=dx, outputs=y, name='decoder'), Model(inputs=x, outputs=h, name='encoder'), log_var


class Gaussian(Layer):
    """这是个简单的层，定义q(z|y)中的均值参数，每个类别配一个均值。
    然后输出“z - 均值”，为后面计算loss准备。
    """
    def __init__(self, num_classes, gau_learnable, **kwargs):
        self.num_classes = num_classes
        self.gau_learnable = gau_learnable
        super(Gaussian, self).__init__(**kwargs)
    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.mean = self.add_weight(name='gau_mean',
                                    shape=(self.num_classes, latent_dim),
                                    initializer='zeros',
                                    trainable=self.gau_learnable[0])
        self.var = self.add_weight(name='var',
                                   shape=(self.num_classes, latent_dim),
                                   initializer='ones',
                                   trainable=self.gau_learnable[1])
        self.pi = self.add_weight(name='pi',
                                  shape=(self.num_classes,),
                                  initializer=Constant(1.0 / self.num_classes),
                                  # initializer='ones',
                                  trainable=False)
    def call(self, inputs):
        z = inputs # z.shape=(batch_size, latent_dim)
        z_3 = K.expand_dims(z, 1)
        batch_mean = K.expand_dims(self.mean, 0)
        # batch_var = K.expand_dims(K.square(self.var), 0)
        batch_var = K.expand_dims(self.var, 0)
        z_minus_mean_2_var = K.square((z_3 - batch_mean)) / batch_var
        p_z_y = K.expand_dims(self.pi, 0) * K.exp(-K.sum(K.log(2*math.pi*batch_var)+z_minus_mean_2_var, axis=2) / 2)
        y = p_z_y / K.sum(p_z_y, axis=-1, keepdims=True) #K.softmax(-K.sum(z_minus_mean_2_var, axis=2) / 2)

        # GMM updating formulation
        # pi = K.sum(y, axis=0) / K.shape(y)[0]
        # u = (K.dot(y.T, z).T / K.sum(y, axis=0)).T
        # # z_3 = np.expand_dims(z, 1)
        # u_3 = np.expand_dims(u, 0)
        # y_3 = np.expand_dims(y, 2)
        # var = (K.sum(y_3 * (z_3 - u_3) ** 2, axis=0).T / K.sum(y, axis=0)).T

        return [z_minus_mean_2_var, y]
        # return [z_minus_mean_2_var+K.log(batch_var), y]
        # z_minus_mean_2 = K.square((z - batch_mean))
        # y = -K.sum(z_minus_mean_2, axis=2) / 2
        # y = K.softmax(y)
        # return [z_minus_mean_2, y]
    def compute_output_shape(self, input_shape):
        return [(None, self.num_classes, input_shape[-1]),(None, self.num_classes)]


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GMAE(object):
    def __init__(self,
                 dims,
                 out_act,
                 n_clusters=10,
                 alpha=1.0,
                 with_var=False,
                 init='glorot_uniform',
                 gau_learnable=[True, True]):

        super(GMAE, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.out_act = out_act
        self.with_var = with_var
        self.decoder, self.encoder, self.log_var = autoencoder(self.dims, out_act=out_act, init=init)
        self.autoencoder = Model(inputs=self.encoder.input, outputs=self.decoder(self.encoder.output), name='AE')

        if with_var:
            z = Lambda(sampling, output_shape=(dims[-1],))([self.encoder.output, self.log_var])
        else:
            z = self.encoder.output

        # prepare gaussian
        self.gaussian = Gaussian(self.n_clusters, gau_learnable, name='gaussian')

        # prepare classifier
        # xc = Input(shape=(self.dims[-1],))
        # # y = Dense(intermediate_dim, activation='relu')(z)
        # y = Dense(self.n_clusters, activation='softmax', kernel_initializer=init)(xc)
        # self.classifier = Model(xc, y)  # 隐变量分类器

        xc = Input(shape=(self.dims[-1],))
        _, yc = self.gaussian(xc)
        self.classifier = Model(xc, yc)  # 隐变量分类器

        # prepare GMAE
        # self.y = self.classifier(z)
        self.z_prior_mean_2, self.y = self.gaussian(z)
        self.gmae = Model(self.encoder.input, [self.decoder(z), self.z_prior_mean_2, self.y])
        # self.GMM_DEC = Model(self.encoder.input, self.y)
        # adopt k-means and t-distribution to initial parameters of gmae
        xi = Input(shape=(self.dims[-1],))
        y_initial = ClusteringLayer(self.n_clusters, name='initial_posterior_gmm')(xi)
        self.initial_posterior = Model(xi, y_initial)

        # # prepare DEC model
        # clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        # self.DEC = Model(inputs=self.encoder.input, outputs=clustering_layer)
        # # self.model = self.encoder

    # def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp/'):
    #     print('...Pretraining...')
    #     if self.out_act == 'linear':
    #         self.autoencoder.compile(optimizer=optimizer, loss='mse')
    #     else:
    #         self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
    #
    #     csv_logger = callbacks.CSVLogger(save_dir + 'pretrain_log.csv')
    #     cb = [csv_logger]
    #     # logging file
    #     import csv
    #     logfile = open(save_dir + 'pre_ae_log.csv', 'w')
    #     logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'acc_km', 'nmi_km', 'ari_km'])
    #     logwriter.writeheader()
    #     if y is not None:
    #         class PrintACC(callbacks.Callback):
    #             def __init__(self, x, y):
    #                 self.x = x
    #                 self.y = y
    #                 super(PrintACC, self).__init__()
    #
    #             def on_train_begin(self, logs=None):
    #                 feature_model = Model(self.model.input,
    #                                       self.model.get_layer('embedding').output)
    #                 features = feature_model.predict(self.x)
    #                 km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
    #                 y_pred = km.fit_predict(features)
    #                 # print()
    #                 acc_km = np.round(metrics.acc(self.y, y_pred), 5)
    #                 nmi_km = np.round(metrics.nmi(self.y, y_pred), 5)
    #                 ari_km = np.round(metrics.ari(self.y, y_pred), 5)
    #                 logdict = dict(epoch=0, acc_km=acc_km, nmi_km=nmi_km, ari_km=ari_km)
    #                 logwriter.writerow(logdict)
    #                 print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
    #                       % (acc_km, nmi_km))
    #
    #             def on_epoch_end(self, epoch, logs=None):
    #                 if int(epochs/10) != 0 and epoch % int(epochs/20) != 0:
    #                     return
    #                 feature_model = Model(self.model.input,
    #                                       self.model.get_layer('embedding').output)
    #                 features = feature_model.predict(self.x)
    #                 km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
    #                 y_pred = km.fit_predict(features)
    #                 # print()
    #                 acc_km = np.round(metrics.acc(self.y, y_pred), 5)
    #                 nmi_km = np.round(metrics.nmi(self.y, y_pred), 5)
    #                 ari_km = np.round(metrics.ari(self.y, y_pred), 5)
    #                 logdict = dict(epoch=epoch+1, acc_km=acc_km, nmi_km=nmi_km, ari_km=ari_km)
    #                 logwriter.writerow(logdict)
    #                 print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
    #                       % (acc_km, nmi_km))
    #
    #         cb.append(PrintACC(x, y))
    #
    #     # begin pretraining
    #     # save initial weights
    #     self.autoencoder.save_weights(save_dir + 'ae_weights_ini.h5')
    #     print('Initial weights are saved to %sae_weights_ini.h5' % save_dir)
    #     t0 = time()
    #     self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
    #     logfile.close()
    #     print('Pretraining time: %ds' % round(time() - t0))
    #     self.autoencoder.save_weights(save_dir + 'ae_weights.h5')
    #     print('Pretrained weights are saved to %sae_weights.h5' % save_dir)
    #     self.pretrained = True

    # def load_DEC_weights(self, weights):  # load weights of GMAE model
    #     self.DEC.load_weights(weights)

    def load_gmae_weights(self, weights):  # load weights of GMAE model
        self.gmae.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    # def predict(self, x):  # predict cluster labels using the output of clustering layer
    #     q = self.DEC.predict(x, verbose=0)
    #     return q.argmax(1)

    # @staticmethod
    # def target_distribution(q):
    #     weight = q ** 2 / q.sum(0)
    #     return (weight.T / weight.sum(1)).T

    # def DEC_compile(self, optimizer='sgd', loss_weight=[0.5,-1], loss='kld'):
    #
    #     # MI
    #     q_tmp = self.DEC(self.encoder.input)
    #     h1_loss = - K.sum(K.mean(q_tmp * K.log(q_tmp + 1e-8), 0))
    #     q_tmp_ = K.mean(q_tmp, 0)
    #     h2_loss = -K.sum(q_tmp_ * K.log(q_tmp_ + 1e-8))
    #     self.DEC.add_loss(loss_weight[0] * h1_loss + loss_weight[1] * h2_loss)
    #     self.DEC.compile(optimizer=optimizer, loss=loss)

    # def GMM_DEC_compile(self, optimizer='sgd', loss_weight=[0.5,-1], loss='kld'):
    #     # MI
    #     q_tmp = self.GMM_DEC(self.encoder.input)
    #     h1_loss = - K.sum(K.mean(q_tmp * K.log(q_tmp + 1e-8), 0))
    #     q_tmp_ = K.mean(q_tmp, 0)
    #     h2_loss = -K.sum(q_tmp_ * K.log(q_tmp_ + 1e-8))
    #     self.GMM_DEC.add_loss(loss_weight[0] * h1_loss + loss_weight[1] * h2_loss)
    #     self.GMM_DEC.compile(optimizer=optimizer, loss=loss)

    # def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3,
    #         update_interval=140, save_dir='./results/temp', time_flag=tt, pre_model='ae'):
    #
    #     print('Update interval', update_interval)
    #     save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
    #     print('Save interval', save_interval)
    #
    #     # Step 1: initialize cluster centers using k-means
    #     t1 = time()
    #     print('Initializing cluster centers with k-means.')
    #     kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
    #     y_pred = kmeans.fit_predict(self.encoder.predict(x))
    #     y_pred_last = np.copy(y_pred)
    #     self.DEC.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    #
    #     # Step 2: deep clustering
    #     # logging file
    #     import csv
    #     logfile = open(save_dir + '/'+pre_model+'_dec_log_'+time_flag+'.csv', 'w')
    #     logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
    #     logwriter.writeheader()
    #
    #     loss = 0
    #     index = 0
    #     index_array = np.arange(x.shape[0])
    #     for ite in range(int(maxiter)):
    #         if ite % update_interval == 0:
    #             # gmm = mixture.GaussianMixture(n_components=len(np.unique(y)), n_init=1, covariance_type='diag')
    #             # gmm.fit(self.model.predict(x))
    #
    #             q = self.DEC.predict(x, verbose=0)
    #             p = self.target_distribution(q)  # update the auxiliary target distribution p
    #
    #             # evaluate the clustering performance
    #             y_pred = q.argmax(1)
    #             if y is not None:
    #                 acc = np.round(metrics.acc(y, y_pred), 5)
    #                 nmi = np.round(metrics.nmi(y, y_pred), 5)
    #                 ari = np.round(metrics.ari(y, y_pred), 5)
    #                 loss = np.round(loss, 5)
    #                 logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=loss)
    #                 logwriter.writerow(logdict)
    #                 print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)
    #
    #             # check stop criterion
    #             delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
    #             y_pred_last = np.copy(y_pred)
    #             if ite > 0 and delta_label < tol:
    #                 print('delta_label ', delta_label, '< tol ', tol)
    #                 print('Reached tolerance threshold. Stopping training.')
    #                 logfile.close()
    #                 break
    #
    #         # train on batch
    #         # if index == 0:
    #         #     np.random.shuffle(index_array)
    #         idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    #         loss = self.DEC.train_on_batch(x=x[idx], y=p[idx])
    #         index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
    #
    #         # save intermediate model
    #         if ite % save_interval == 0:
    #             print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
    #             self.DEC.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')
    #
    #         ite += 1
    #
    #     # save the trained model
    #     logfile.close()
    #     print('saving model to:', save_dir + '/DEC_model_final.h5')
    #     self.DEC.save_weights(save_dir + '/DEC_model_final.h5')
    #
    #     return y_pred

    def fit(self, x, y=None, optimizer='adam', epochs=300, batch_size=256, save_dir='results/temp',
            loss_weight=[1,1,1,-1], update_epochs=5, update_batch=None, record_epochs=2, save_epochs=4, tol=5e-4):
        print('...GMMtraining...')

        # # initialize cluster centers
        # # k-means
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, n_jobs=8)
        features = self.encoder.predict(x)
        kmeans.fit(features)
        self.initial_posterior.get_layer(name='initial_posterior_gmm').set_weights([kmeans.cluster_centers_])
        y_init = self.initial_posterior.predict(features)
        pi, mean, var = GMM_update(y_init, features, kmeans.cluster_centers_)
        self.gmae.get_layer(name='gaussian').set_weights([kmeans.cluster_centers_, var, pi])
        # # gmm
        # # gmm = mixture.GaussianMixture(n_components=self.n_clusters, n_init=20, covariance_type='diag')
        # # gmm.fit(self.encoder.predict(x))
        # # y_pred = gmm.predict(self.encoder.predict(x))
        # # self.GMM.get_layer(name='gaussian').set_weights([gmm.means_])
        # y_pred_last = np.copy(y_pred)

        # compute loss
        # reconstruction loss
        if self.out_act == 'linear':
            xent_loss = 0.5 * K.mean((self.gmae.input - self.gmae.output[0]) ** 2)
        else:
            xent_loss = K.mean(-self.gmae.input * K.log(self.gmae.output[0] + K.epsilon()) - (1 - self.gmae.input) * K.log(1 - self.gmae.output[0] + K.epsilon()))
        # kl loss based on GMM
        z_log_var = K.expand_dims(self.log_var, 1)
        if self.with_var:
            kl_loss = - 0.5 * (z_log_var - self.z_prior_mean_2)
        else:
            kl_loss = 0.5 * (self.z_prior_mean_2)
        kl_loss = K.mean(K.batch_dot(K.expand_dims(self.y, 1), kl_loss))
        # MI loss
        cat_loss = - K.mean(self.y * K.log(self.y + K.epsilon()), 0)
        y_ = K.mean(self.y, 0)
        h_loss = - y_ * K.log(y_ + K.epsilon())
        mi_loss = K.sum(cat_loss - h_loss)

        gmae_loss = loss_weight[0] * xent_loss + loss_weight[1] * kl_loss + loss_weight[2] * mi_loss
        self.gmae.add_loss(gmae_loss)

        self.gmae.compile(optimizer=optimizer)

        # logging file
        import csv
        logfile = open(save_dir + 'gmm_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'acc_km', 'nmi_km', 'ari_km', 'acc_cl', 'nmi_cl',
                                                        'ari_cl', 'loss'])
        logwriter.writeheader()

        class GMAECallback(callbacks.Callback):
            def __init__(self, x, y, cl, initial_p, n_c):
                super(GMAECallback, self).__init__()
                self.x = x
                self.y = y
                self.cl = cl
                self.initial_p = initial_p
                self.n_c = n_c

            def on_train_begin(self, logs=None):
                self.t0 = time()
                self.en = Model(self.model.input,
                                self.model.get_layer('embedding').output)
                features = self.en.predict(self.x)
                # initialize cluster centers and other parameters of gaussian
                # k-means
                self.kmeans = KMeans(n_clusters=self.n_c, n_init=20, n_jobs=8)
                y_pred = self.kmeans.fit_predict(features)
                self.initial_p.get_layer(name='initial_posterior_gmm').set_weights([self.kmeans.cluster_centers_])
                y_init = self.initial_p.predict(features)
                pi, mean, var = GMM_update(y_init, features, self.kmeans.cluster_centers_)
                pre_gau_weights = self.model.get_layer(name='gaussian').get_weights()
                self.model.get_layer(name='gaussian').set_weights([self.kmeans.cluster_centers_, var, pi])

                # gmm
                gmm = mixture.GaussianMixture(n_components=self.n_c, n_init=20, covariance_type='diag')
                gmm.fit(features)
                y_gmm = gmm.predict(features)
                self.y_pred_last = np.copy(y_pred)

                # save initial model
                print('saving model to:', save_dir + 'GMM_model_' + str(0) + '.h5')
                self.model.save_weights(save_dir + 'GMM_model_' + str(0) + '.h5')

                # record acc
                y_p = self.cl.predict(features)
                y_cl = y_p.argmax(axis=1)
                if self.y is not None:
                    y_km = y_pred
                    acc_km = np.round(metrics.acc(self.y, y_km), 5)
                    nmi_km = np.round(metrics.nmi(self.y, y_km), 5)
                    ari_km = np.round(metrics.ari(self.y, y_km), 5)
                    acc_gmm = np.round(metrics.acc(self.y, y_gmm), 5)
                    nmi_gmm = np.round(metrics.nmi(self.y, y_gmm), 5)
                    ari_gmm = np.round(metrics.ari(self.y, y_gmm), 5)
                    acc_cl = np.round(metrics.acc(self.y, y_cl), 5)
                    nmi_cl = np.round(metrics.nmi(self.y, y_cl), 5)
                    ari_cl = np.round(metrics.ari(self.y, y_cl), 5)
                    loss = np.round(0, 5)
                    logdict = dict(epoch=0, acc_km=acc_km, nmi_km=nmi_km, ari_km=ari_km, acc_cl=acc_km,
                                   nmi_cl=nmi_km, ari_cl=ari_km, loss=loss)
                    logwriter.writerow(logdict)
                    logdict = dict(epoch=0, acc_km=acc_km, nmi_km=nmi_km, ari_km=ari_km, acc_cl=acc_gmm,
                                   nmi_cl=nmi_gmm, ari_cl=ari_gmm, loss=loss)
                    logwriter.writerow(logdict)
                    logdict = dict(epoch=0, acc_km=acc_km, nmi_km=nmi_km, ari_km=ari_km, acc_cl=acc_cl,
                                   nmi_cl=nmi_cl, ari_cl=ari_cl, loss=loss)
                    logwriter.writerow(logdict)
                    print('epoch %d: acc_km = %.5f, nmi_km = %.5f, ari_km = %.5f, acc_cl = %.5f, nmi_cl = %.5f, '
                          'ari_cl = %.5f' % (0, acc_km, nmi_km, ari_km, acc_cl, nmi_cl, ari_cl), '; loss = ',
                          loss)

            def on_batch_end(self, batch, logs=None):
                batch = batch + 1

                # update parameters of gaussian
                if update_batch is not None and batch % update_batch == 0:
                    features = self.en.predict(self.x)
                    y_p = self.cl.predict(features)
                    pre_gau_weights = self.model.get_layer(name='gaussian').get_weights()
                    pi, mean, var = GMM_update(y_p, features, pre_gau_weights[0])
                    self.model.get_layer(name='gaussian').set_weights([mean, var, pi])

            def on_epoch_end(self, epoch, logs=None):
                features = self.en.predict(self.x)
                epoch = epoch + 1

                # update learning_rate
                # if not hasattr(self.model.optimizer, "lr"):
                #     raise ValueError('Optimizer must have a "lr" attribute.')
                #     # Get the current learning rate from model's optimizer.
                # lr = float(K.get_value(self.model.optimizer.lr))
                # # Call schedule function to get the scheduled learning rate.
                # scheduled_lr = lr_schedule(epoch, lr)
                # # Set the value back to the optimizer before this epoch starts
                # K.set_value(self.model.optimizer.lr, scheduled_lr)
                # print("\nEpoch %05d: Learning rate is %6.8f." % (epoch, scheduled_lr))

                # update means of gaussian with kmeans
                # if update_epochs is not None and epoch % update_epochs == 0:
                #     self.kmeans.fit(features)
                #     pre_gau_weights = self.model.get_layer(name='gaussian').get_weights()
                #     self.model.get_layer(name='gaussian').set_weights([self.kmeans.cluster_centers_, pre_gau_weights[1],
                #                                                        pre_gau_weights[2]])


                # save intermediate model
                if epoch % save_epochs == 0:
                    print('saving model to:', save_dir + 'GMM_model_' + str(epoch) + '.h5')
                    self.model.save_weights(save_dir + 'GMM_model_' + str(epoch) + '.h5')

                # record acc
                y_p = self.cl.predict(features)
                y_cl = y_p.argmax(axis=1)
                if self.y is not None:
                    if epoch % record_epochs == 0:
                        # kmeans = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=8)
                        y_km = self.kmeans.fit_predict(features)

                        acc_km = np.round(metrics.acc(self.y, y_km), 5)
                        nmi_km = np.round(metrics.nmi(self.y, y_km), 5)
                        ari_km = np.round(metrics.ari(self.y, y_km), 5)
                        acc_cl = np.round(metrics.acc(self.y, y_cl), 5)
                        nmi_cl = np.round(metrics.nmi(self.y, y_cl), 5)
                        ari_cl = np.round(metrics.ari(self.y, y_cl), 5)
                        loss = np.round(logs.get('loss'), 5)
                        logdict = dict(epoch=epoch, acc_km=acc_km, nmi_km=nmi_km, ari_km=ari_km, acc_cl=acc_cl,
                                       nmi_cl=nmi_cl, ari_cl=ari_cl, loss=loss)
                        logwriter.writerow(logdict)
                        print('epoch %d: acc_km = %.5f, nmi_km = %.5f, ari_km = %.5f, acc_cl = %.5f, nmi_cl = %.5f, '
                              'ari_cl = %.5f' % (epoch, acc_km, nmi_km, ari_km, acc_cl, nmi_cl, ari_cl), '; loss = ', loss)

                # update parameters of gaussian
                if update_epochs is not None and epoch % update_epochs == 0:
                    pre_gau_weights = self.model.get_layer(name='gaussian').get_weights()
                    pi, mean, var = GMM_update(y_p, features, pre_gau_weights[0])
                    self.model.get_layer(name='gaussian').set_weights([pre_gau_weights[0], pre_gau_weights[1], pi])

                # check stop criterion
                delta_label = np.sum(y_cl != self.y_pred_last).astype(np.float32) / y_cl.shape[0]
                self.y_pred_last = np.copy(y_cl)
                if epoch > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    self.model.stop_training = True

            def on_train_end(self, logs=None):
                print('GMMtraining time: %ds' % round(time() - self.t0))
                # save the trained model
                logfile.close()
                print('saving model to:', save_dir + 'GMM_model_final.h5')
                self.model.save_weights(save_dir + 'GMM_model_final.h5')

        cb = []
        cb.append(GMAECallback(x, y, self.classifier, self.initial_posterior, self.n_clusters))

        # begin GMM training
        self.gmae.fit(x, batch_size=batch_size, epochs=epochs, callbacks=cb)

        y_pred = self.classifier.predict(self.encoder.predict(x)).argmax(axis=1)
        return y_pred