import numpy as np
from scipy.special import softmax, expit
from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
from sklearn.dummy import DummyClassifier as _DummyClassifier
from sklearn.metrics import log_loss as _log_loss


####################################################################################################

def _explained_deviance(y_true, y_pred_logits=None, y_pred_probas=None, 
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
    # strategy can be {"most_frequent", "prior", "stratified", "uniform",  "constant"}
    y_null_probas = _DummyClassifier(strategy="prior").fit(X_dummy, y_true).predict_proba(X_dummy)
    # suggestion from https://stackoverflow.com/a/53215317
    llf = -_log_loss(y_true, y_pred_probas, normalize=False, labels=unique_y) 
    llnull = -_log_loss(y_true, y_null_probas, normalize=False, labels=unique_y) 
    ### McFadden’s pseudo-R-squared: 1 - (llf / llnull)
    explained_deviance = 1 - (llf / llnull)
    ## Cox & Snell’s pseudo-R-squared: 1 - exp((llnull - llf)*(2/nobs))
    # explained_deviance = 1 - np.exp((llnull - llf) * (2 / len(y_pred_probas))) ## TODO, not implemented
    if returnloglikes:
        return explained_deviance, {'loglike_model':llf, 'loglike_null':llnull}
    else:
        return explained_deviance



def d2(y, y_pred):
    # convert values of y to one-hot encoded if not already
    y = np.array(y)
    if  y[0] not in [0,1]:
        enc = _OneHotEncoder(sparse_output=False).fit(y.reshape(-1,1))
        y = enc.transform(y.reshape(-1,1))
        if y.shape[1] > 2: 
            raise ValueError("explained_deviance currently only supports binary labels but y has {} classes".format(y.shape[1]))
        else:
            y = y[:,1].squeeze()

    return _explained_deviance(y_true=y, y_pred_probas=y_pred, unique_y=[0,1]) # unique_y TODO remove hardcoding


####################################################################################################

def r2_logodds(y, y_pred):
    """Computes the R^2 of the log-odds space for the binary variable y. 
    Args:
        y      : a binary variable
        y_pred : y_pred_probas (probability of class 1)
    """
    # check that y is binary
    y = np.array(y)
    y_states = np.unique(y)
    if len(y_states) > 2:
        raise ValueError("r2_logodds() metric is only defined for binary variables")
    # if y_true is string convert to binary
    if isinstance(y_states[0], str):
        y = _OneHotEncoder(sparse_output=False).fit(y.reshape(-1,1))
        y = y[:,1].squeeze()
    # convert y to (pseudo) probabilities
    y_true_probas = np.clip(y, 0.01, .99)
    y_true_logodds = np.log(y_true_probas / (1 - y_true_probas))
    # compute the log odds from the probabilities
    y_pred_logodds = np.log(y_pred / (1 - y_pred))
    # compute the R^2 of the log-odds space
    ss_res = np.sum((y_true_logodds - y_pred_logodds) ** 2)
    ss_tot = np.sum((y_true_logodds - np.mean(y_true_logodds)) ** 2)
    r2_logodds = 1 - ss_res / ss_tot
    return r2_logodds

####################################################################################################

def r2_odds(y, y_pred):
    """Computes the R^2 of the inverse odds space z=(p/1-p) for the binary variable y. 
    Args:
        y      : a binary variable
        y_pred : y_pred_probas (probability of class 1)
    """
    # check that y is binary
    y = np.array(y)
    y_states = np.unique(y)
    if len(y_states) > 2:
        raise ValueError("r2_logodds() metric is only defined for binary variables")
    # if y_true is string convert to binary
    if isinstance(y_states[0], str):
        y = _OneHotEncoder(sparse_output=False).fit(y.reshape(-1,1))
        y = y[:,1].squeeze()
    # convert y to (pseudo) probabilities
    # # add a error of 0.01 to avoid division by zero and scale to 0 to 100
    y_true_probas = np.clip(y, 0.01, .99)
    y_true_invodds = y_true_probas / (1 - y_true_probas)
    # compute the inverse odds from the probabilities
    y_pred_invodds = y_pred / (1 - y_pred)
    # compute the R^2 of the log-odds space
    ss_res = np.sum((y_true_invodds - y_pred_invodds) ** 2)
    ss_tot = np.sum((y_true_invodds - np.mean(y_true_invodds)) ** 2)
    r2_odds = 1 - ss_res / ss_tot
    return r2_odds

####################################################################################################

def loglikelihood_ratio(y, y_pred, y_true_probas=None):
    ''' Compute the loglikelikehood(y_true_probas)/ loglikelikehood(y_pred_probas) )'''
    y = np.array(y)
    states = np.unique(y)
    n_states = len(states)
    
    # check if one hot encoding is needed first
    if isinstance(states[0], str):
        enc = _OneHotEncoder(sparse_output=False).fit(y.reshape(-1,1))
        y = enc.transform(y.reshape(-1,1))
        n_states = y.shape[-1]
        y = y[:,1].squeeze()
    
    assert n_states == 2, "loglikelihood_ratio() currently only supports binary labels"

    ## calculate pseudo probabilities of y, if probabilities are not provided
    if y_true_probas is None:
        y_true_probas = np.clip(y, 0.01, .99)

    ## compute the true log-likelihood
    ll_true = _log_loss(y, y_true_probas, normalize=True, labels=states)
    ll_pred = _log_loss(y, y_pred, normalize=True, labels=states) 

    return ll_pred / ll_true
    
