# Gaussian Mixture AutoEncoder (GMAE)

Keras implementation for paper:

* Deep Unsupervised Clustering By Information Maximization on Gaussian Mixture Autoencoders.

## Requirements
1. Python == 3.6
2. keras == 2.2.4
3. scikit-learn == 0.23.2
4. numpy == 1.19.2

## Usage
1. Prepare datasets.    
    Download **STL**:
    ```
    cd data/stl
    bash get_data.sh
    cd ../..
    ```

    **MNIST** can be downloaded automatically when you run the code.

2. Run experiment.   
    ```python run_exp.py```  
    The GMAE model will be saved to "results/DEC_model_final.h5".