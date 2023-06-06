import os
import pandas as pd
import numpy as np

from skimage.io import imread_collection

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns


#################################################################################################
# DATA MODULE
#################################################################################################

# functions

def get_data(df):
    
    img_dir = 'toybrains/images'
    
    # set up the image collection from list of paths
    
    image_paths = [os.path.join(img_dir, str(i).zfill(5) + '.jpg') for i in df['subjectID']]
    image_collection = imread_collection(image_paths)
    
    # convert image into numpy array
    
    X = np.array(image_collection)
    
    # reshape the input array to numpy (batch, feature) 2D
    
    X = X.reshape(X.shape[0], -1)
    
    # set up the corresponding labels
    
    y = [i for i in df['label']]
    
    return X, y

def get_reduc_loader(dataset, method='PCA', n_components=3, seed=42, save=False):
    '''
    conduct the dimensionality reduction and return values
    
    PARAMETER
    ---------
    dataset : tuple
        tuple of (DF_train, DF_val, DF_test)
    
    method : string, default : PCA
        dimensionality reduction method
        
    seed : integer, default : 42
        random seed
        
    save : boolean, default : False
        save the values in new csv
        
    '''
    
    # Add more methods here as needed
    
    methods = {
        "tSNE" : TSNE, # (TODO) transform error
        "PCA" : PCA,
        "MDS" : MDS,
        "ICA" : FastICA,
        "LDA" : LinearDiscriminantAnalysis, # (TODO) no random state, X, y
    }
    
    assert method in methods, f"Invalid method: {method}, if you need please add it manually"
    
    DF_train, DF_val, DF_test = dataset
    
    tr_X, target_train = get_data(df = DF_train)
    vl_X, target_val = get_data(df = DF_val)
    te_X, target_test = get_data(df = DF_test)

    # Get the corresponding class from the dictionary
    
    reducer_class = methods[method]

    # Create an instance of the dimensionality reduction class
    
    reducer = reducer_class(n_components=n_components) #, random_state=seed)

    # Apply dimensionality reduction
    
    data_train = reducer.fit_transform(tr_X)
    data_val = reducer.transform(vl_X)
    data_test = reducer.transform(te_X)
    
    # Save the model into new csv file
    
    if save:
        print("TODO: implement save function")
        print(f"Train data : {data_train.shape}, label : {len(target_train)}"
              f"Validation data : {data_val.shape}, label : {len(target_val)}"
              f"Test data : {data_test.shape}, label : {len(target_test)}")
        
    return (data_train, target_train, data_val, target_val, data_test, target_test)

#################################################################################################
# Scikit-learn Model
#################################################################################################

# Reference: 
# https://inria.github.io/scikit-learn-mooc/python_scripts/linear_models_sol_05.html
# https://github.com/ritterlab/ML_for_alcohol_misuse/blob/main/MLpipeline/config/classification.py
    
def run_logistic_regression(dataset):
    
    # Load dataset
    
    (data_train, target_train, data_val, target_val, data_test, target_test) = dataset
    
    # Set up the pipeline with StandardScaler and LogisticRegression
    
    pipe = make_pipeline(VarianceThreshold(), StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)) #, penalty="l2"))
    parameters = {'logisticregression__C': [0.1, 1, 10, 100]}
    
    # Use GridSearchCV to find the optimal hyperparameters for the pipeline

    clf = GridSearchCV(pipe, param_grid=parameters)
    
    # Train and fit logistic regression model

    clf.fit(data_train, target_train)
    
    # Predict using the trained model

    # y_pred = clf.predict(data_val)

    # Calculate accuracy
    
    # accuracy = accuracy_score(target_val, y_pred)
    
    # Calculate accuracy

    tr_acc = clf.score(data_train, target_train)
    vl_acc = clf.score(data_val, target_val)
    te_acc = clf.score(data_test, target_test)

    print(f"Train Accuracy: {tr_acc:>8.4f} "
          f"Validation Accuracy: {vl_acc:>8.4f} "
          f"Test Accuracy: {te_acc:>8.4f}")
    
    return (tr_acc, vl_acc, te_acc)

    
#################################################################################################
# Visualization
#################################################################################################

# Reference:
# https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
# https://inria.github.io/scikit-learn-mooc/python_scripts/linear_models_sol_05.html