import numpy as np
from scipy.special import softmax, expit
from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
from sklearn.dummy import DummyClassifier as _DummyClassifier
from sklearn.metrics import log_loss as _log_loss


def _get_y_pseudo_proba(y):
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
    return y_true_probas

def _logit(p):
    return np.log(p / (1 - p))

####################################################################################################
def deviance(y, y_pred, labels=[0,1]):
    '''deviance = -2 * (log-likelihood of the model - log-likelihood of the saturated model)
    The saturated model is the model that perfectly predicts the observed data.
    '''
    ll_satur = np.round(-_log_loss(y, y, normalize=False, labels=labels)     , decimals=5)
    ll_model = np.round(-_log_loss(y, y_pred, normalize=False, labels=labels), decimals=5)
    return -2 * (ll_model - ll_satur)
####################################################################################################

def _explained_deviance(y_true, y_pred_logits=None, y_pred_probas=None, 
                        null_model_strategy="uniform",
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

    # compute the null model's predicted probability
    X_dummy = np.zeros(len(y_true))
    if null_model_strategy == "uniform":
        y_null_probas = np.full(y_pred_probas.shape, 1/len(unique_y))
    else: # other strategy can be {"most_frequent", "prior", "stratified",  "constant"}
        y_null_probas = _DummyClassifier(strategy=null_model_strategy).fit(X_dummy, y_true).predict_proba(X_dummy)
    # deviance = 2*(loglike_satur - loglike_model) = 2*(-ll_model) = (2*negative_loglikelihood_model) suggestion from https://stackoverflow.com/a/53215317
    deviance_model = 2 * _log_loss(y_true, y_pred_probas, normalize=False, labels=unique_y) 
    deviance_null  = 2 * _log_loss(y_true, y_null_probas, normalize=False, labels=unique_y) 

    ### McFadden’s pseudo-R-squared: 1 - (ll_model / ll_null)
    explained_deviance = 1 - (deviance_model / deviance_null)
    ## Cox & Snell’s pseudo-R-squared: 1 - exp((ll_null - ll_model)*(2/nobs))
    # explained_deviance = 1 - np.exp((ll_null - ll_model) * (2 / len(y_pred_probas))) ## TODO, not implemented
    if returnloglikes:
        return explained_deviance, {'deviance_model':deviance_model, 'deviance_null':deviance_null}
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

def logodds_metric(y, y_pred, metric='r2'):
    """Computes the Mean absolute error of the log-odds space for the binary variable y. 
    Args:
        y      : a binary variable
        y_pred : y_pred_probas (probability of class 1
        metric : str, one of ['r2', 'mae', 'mse']
    """
    metric = metric.lower()
    assert metric in ['r2', 'mae', 'mse'], "metric should be one of ['r2', 'mae', 'mse']"
    y_true_probas = _get_y_pseudo_proba(y)
    # compute the log odds from the probabilities
    y_true_logodds = _logit(y_true_probas)
    y_pred_logodds = _logit(y_pred)

    # compute the explained variance of the log-odds space
    if metric=='r2':
        ss_res = np.sum((y_true_logodds - y_pred_logodds) ** 2)
        ss_tot = np.sum((y_true_logodds - np.mean(y_true_logodds)) ** 2)
        logodds_metric = 1 - (ss_res / ss_tot)
    elif metric=='mae':
        logodds_metric = np.mean(np.abs(y_true_logodds - y_pred_logodds))
    elif metric=='mse':
        logodds_metric = np.mean((y_true_logodds - y_pred_logodds) ** 2)

    return logodds_metric

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
    
