from GMAE import GMAE
import os, csv, time
from datasets import load_data
from keras.optimizers import SGD, Adam
from keras.initializers import VarianceScaling
import numpy as np


expdir='./results/exp1'
if not os.path.exists(expdir):
    os.mkdir(expdir)

logfile = open(expdir + '/results.csv', 'a')
logwriter = csv.DictWriter(logfile, fieldnames=['trials', 'acc', 'nmi', 'ari'])
logwriter.writeheader()

trials=10
for db in ['mnist', 'reuters', 'reuters10k', 'stl', 'har']: #['usps', 'reuters', 'reuters10k', 'stl', 'mnist', 'fmnist', 'har']:
    logwriter.writerow(dict(trials=db, acc='', nmi='', ari=''))
    save_db_dir = os.path.join(expdir, db)
    if not os.path.exists(save_db_dir):
        os.mkdir(save_db_dir)

    x, y = load_data(db)
    n_clusters = len(np.unique(y))

    init = 'glorot_uniform'
    optimizer = 'adam'
    loss_weight = [1, 1, 1, -1]
    gau_learnable = [True, True]
    epochs = 300
    record_epochs = 2
    tol = 5e-4
    out_act = 'linear'
    # setting parameters
    if db == 'mnist':
        init = 'lecun_uniform'
        optimizer = Adam(lr=0.00002, beta_1=0.9, beta_2=0.999, decay=0.0)
        out_act = 'sigmoid'
    elif db == 'reuters10k':
        loss_weight = [.1, 10, 1, -1]
        optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, decay=0.0)
    elif db == 'reuters':
        loss_weight = [.1, 10, 1, -1]
        optimizer = Adam(lr=1e-7, beta_1=0.9, beta_2=0.999, decay=0.0)
    elif db == 'stl':
        optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0)
    elif db == 'har':
        optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0)

    # prepare model
    dims = [x.shape[-1], 500, 500, 2000, 10]

    '''Training for base and nosp'''
    results = np.zeros(shape=(trials, 3))
    baseline = np.zeros(shape=(trials, 3))
    metrics0 = []
    metrics1 = []
    metrics2 = []
    metrics3 = []

    for i in range(trials):  # base
        # i = 15
        gmae = GMAE(dims=[x.shape[-1], 500, 500, 2000, 10], out_act=out_act, n_clusters=n_clusters, init=init)
        gmae.autoencoder.load_weights(save_db_dir + '/ae_weights.h5')

        save_dir = os.path.join(save_db_dir, 'trial%d' % i)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        gmae.fit(x=x, y=y, optimizer=optimizer, loss_weight=loss_weight,
                 save_dir=save_dir+'/', epochs=epochs, record_epochs=record_epochs, tol=tol)

        log = open(os.path.join(save_dir, 'gmm_log.csv'), 'r')
        reader = csv.DictReader(log)
        metrics = []
        for row in reader:
            metrics.append([row['acc_cl'], row['nmi_cl'], row['ari_cl']])
        metrics2.append(metrics[-1])
        log.close()

    metrics2 = np.asarray(metrics2, dtype=float)

    for t, line in enumerate(metrics2):
        logwriter.writerow(dict(trials=t, acc=line[0], nmi=line[1], ari=line[2]))
    logwriter.writerow(dict(trials=' ', acc=np.mean(metrics2, 0)[0], nmi=np.mean(metrics2, 0)[1], ari=np.mean(metrics2, 0)[2]))

logfile.close()
