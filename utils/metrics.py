import numpy as np
from sklearn.metrics import log_loss
from sklearn.dummy import DummyClassifier
from scipy.special import softmax, expit


def explained_deviance(y_true, y_pred_logits=None, y_pred_probas=None, 
                       returnloglikes=False, unique_y=[0,1]):
    """Computes explained_deviance score to be comparable to explained_variance
    Function taken from https://github.com/RoshanRane/Deviance_explained/blob/main/deviance.py"""
    
    assert y_pred_logits is not None or y_pred_probas is not None, "Either the predicted probabilities \
(y_pred_probas) or the predicted logit values (y_pred_logits) should be provided. But neither of the two were provided."
    
    if y_pred_logits is not None and y_pred_probas is None:
        # check if binary or multiclass classification
        if y_pred_logits.ndim == 1: 
            y_pred_probas = expit(y_pred_logits)
        elif y_pred_logits.ndim == 2: 
            y_pred_probas = softmax(y_pred_logits, axis=-1)
        else: # invalid
            raise ValueError(f"logits passed seem to have incorrect shape of {y_pred_logits.shape}")
        
    if y_pred_probas.ndim == 1: y_pred_probas = np.stack([1-y_pred_probas, y_pred_probas], axis=-1)
    total_probas = y_pred_probas.sum(axis=-1).round(decimals=4)
    assert (abs(total_probas-1.)<0.1).all(), f"the probabilities do not sum to one, {total_probas}" 
    unique_y = np.unique(y_true)
    # compute a null model's predicted probability
    X_dummy = np.zeros(len(y_true))
    y_null_probas = DummyClassifier(strategy='prior').fit(X_dummy, y_true).predict_proba(X_dummy)
    #strategy : {"most_frequent", "prior", "stratified", "uniform",  "constant"}
    # suggestion from https://stackoverflow.com/a/53215317
    llf = -log_loss(y_true, y_pred_probas, normalize=False, labels=unique_y)
    llnull = -log_loss(y_true, y_null_probas, normalize=False, labels=unique_y)
    ### McFadden’s pseudo-R-squared: 1 - (llf / llnull)
    explained_deviance = 1 - (llf / llnull)
    ## Cox & Snell’s pseudo-R-squared: 1 - exp((llnull - llf)*(2/nobs))
    # explained_deviance = 1 - np.exp((llnull - llf) * (2 / len(y_pred_probas))) ## TODO, not implemented
    if returnloglikes:
        return explained_deviance, {'loglike_model':llf, 'loglike_null':llnull}
    else:
        return explained_deviance


def d2_metric_probas(y, y_pred):
    return explained_deviance(y_true=y, y_pred_probas=y_pred)
