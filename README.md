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

## Results

```
python run_exp.py
```
Table 1. Mean performance of GMAE over 10 trials. See [results.csv](./results/exp1/results.csv) for detailed results for each trial.  

   |        |     |  acc  |  nmi  |  ari  |  
   :--------|:---:|:-----:|:-----:|:-----:|
   |mnist   | mean |  0.9717   |  0.9260   | 0.9385    |
   |        | std |  0.00035   |  0.00073   | 0.00076   |
   |reuters  | mean |  0.7846   |  0.6356   | 0.6946    |
   |        | std |  0.00006   |  0.00076   | 0.00095   |
   |reuters10K| mean |  0.8385   |  0.6033   | 0.6764    |
   |        | std |  0.0063   |  0.012   | 0.014    |
   |stl     | mean |  0.9353   |  0.8645   | 0.8619    |
   |        | std |  0.00075   |  0.0013   | 0.0016   |
   |HAR | mean |  0.8922   |  0.8401   | 0.7909    |
   |        | std |  0.0030   |  0.0052   | 0.0051    |

