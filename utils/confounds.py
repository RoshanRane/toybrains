import numpy as np
from sklearn.base import BaseEstimator
from scipy.stats import ttest_ind, pearsonr
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict, \
    RepeatedStratifiedKFold, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score, get_scorer
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection import BaseCrossValidator
from sklearn import linear_model
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from itertools import combinations
from copy import deepcopy
import multiprocessing as mp



class CounterBalance(BaseEstimator):
    
    def __init__(self, 
                 oversample=None, random_state=None, debug=False):
        self.oversample = oversample
        self.random_state = random_state
        self.debug = debug
        

    def fit_resample(self, X, y, groups=None, cb_by=None):
        if self.debug: 
            print("[confounds.py] CounterBalance.fit_resample() called with \
X.shape={}  y.shape={}  cb_by.shape={}  groups.shape={}".format(
                X.shape, y.shape, 
                cb_by.shape if cb_by is not None else 'None',
                groups.shape if groups is not None else 'None'))
            print("Distribution of labels/grp before counter balancing:")
        
        if cb_by is None: cb_by=y
        if groups is None: # then set all data points as a single group
            groups= np.zeros_like(y)     

        self.keep_idxs = self._get_cb_sampled_idxs(groups, cb_by)
        
        return X.iloc[self.keep_idxs], y.iloc[self.keep_idxs]
    
    
    def _get_cb_sampled_idxs(self, groups, cb_by):
        
        rng = np.random.default_rng(self.random_state)
        df = pd.DataFrame({"idx":range(len(cb_by)), "cb_by":cb_by, "groups":groups})
        keep_idxs = np.array([])
        
        for g, dfi in df.groupby("groups"):
            
            counts = dfi.cb_by.value_counts()                 
            if self.debug: print(g,"\t", dict(counts.sort_index()))
                
            # Completely skip groups that have less than 2 members in any one of the classes or have just 1 class
            if counts.min() < 2 or (len(counts) < 2):
                raise Exception("Group={} has only 1 class or very few subjects in 1 of the classes. Excluding it".format(g))
                continue
     
            if self.oversample is None:
                # randomly chose to either subsample the larger group or over-sample in the smaller group
                oversample_i = rng.choice([True, False])
            else:
                oversample_i = self.oversample
            
            for _, dfii in dfi.groupby("cb_by"):
                # oversample the smaller group if oversample_i = True
                if oversample_i: 
                    # keep the entire largest group as is
                    if len(dfii) == counts.max():
                        keep_idxs = np.append(keep_idxs, dfii.idx)
                    else:
                        # in smaller group, ensure all indices occur at least once
                        keep_idxs = np.append(keep_idxs, dfii.idx) 
                        # oversample for the remaining counts to be filled
                        idxs = rng.choice(dfii.idx, size=counts.max()-len(dfii), replace=True)
                        keep_idxs = np.append(keep_idxs, idxs)   
                        
                # subsample the larger group if oversample_i = False
                else: 
                    # keep the entire smaller group as is
                    if len(dfii) == counts.min():
                        keep_idxs = np.append(keep_idxs, dfii.idx)
                    else:
                        idxs = rng.choice(dfii.idx, size=counts.min(), replace=False)
                        keep_idxs = np.append(keep_idxs, idxs)  
        
        if self.debug:
            print("[confounds.py] After counter balancing:")
            for g, dfi in df.set_index("idx").loc[keep_idxs].groupby("groups"):
                print(g, "\t", dict(dfi.cb_by.value_counts().sort_index()))

        return np.sort(keep_idxs).astype(int)
        

class ConfoundRegressorCategoricalX(BaseEstimator):
    """This is class to regress out categorical confounds. It is inspired by
    the work of Snoek et al., 2019, but is different in a few ways:
    todo
    """    
    
    def __init__(self, debug=False):
        self.debug = debug
        self.weights_ = None


    def fit_resample(self, X, y=None, groups=None):
        """[summary]
        Args:
            X (np.array): A numpy array of shape (n_samples, n_features+1). 
                The last column holds the categorical confound.
            y : Does nothing. Defaults to None.
        Returns:
            class: Returns the class.
        """             
        if self.debug: 
            print("[confounds.py] ConfoundRegressor.fit_resample() called with \
X.shape={}  y.shape={}  groups.shape={}".format(
                X.shape, y.shape, groups.shape if groups is not None else 'None'))
            
        X_new = np.zeros_like(X)        
        groups = groups.ravel()
        # Calculate means of each group 
        self.grp_means = {}
        total_mean = X.mean(0)
        
        for g in np.unique(groups):
            grp_mean = X[groups == g].mean(axis=0) - total_mean
            # Time to subtract the category mean from X     
            X_new[groups == g] = X[groups == g] - grp_mean
            
            if self.debug:   
                v = X[groups == g]
                print("[confounds.py] {} before cr: \t(min,mean,max)=({:.2f},{:.2f},{:.2f})\tshape={}".format(
                    g, v.min(), v.mean(), v.max(), v.shape))
                v = X_new[groups == g]
                print("[confounds.py] {}  after cr: \t(min,mean,max)=({:.2f},{:.2f},{:.2f})".format(
                    g, v.min(), v.mean(), v.max()))
                
        return X_new, y

class ConfoundRegressorContinuousX(BaseEstimator) :
    """This is class to regress out continous confounds. It is inspired by
    the work McNamee Roseanne: Regression modelling and other methods to control confounding (2005)
    todo
    """    
    def __init__(self, debug=False):
        self.debug = debug

    def fit_resample(self, X, y=None, age=None): 
        """
        Args:
            X (np.array): A numpy array of shape (n_samples, n_features+1). 
            The last column holds the continous confounder.
            y : Does nothing. Defaults to None.
        Returns:
            class: Residuals.
        """

        if self.debug: 
            print(
                "[DEBUG] ConfoundRegressorContinuousX.regress_features()called with \X.shape={}y.shape={}".format(X.shape, y.shape))       

        confounder = age


        if len(np.unique(confounder))/ len(confounder) < 0.1:
            raise ValueError("This methos is only available for continuous condounders!")


        # calculate the residuals of each feature in X
        residuals = []
        for feature in np.transpose(X):
            clf = linear_model.LinearRegression(fit_intercept=True)
            clf.fit(confounder.reshape(-1, 1), feature)
            residual =  feature -  clf.coef_[0] * confounder  
            residuals.append(residual)
        X_residualized = np.transpose(residuals)


        return X_residualized , y

# class PredictionBasedCorrection():

#     def __init__(self, m, confound):
#         self.m = m
#         self.confound = confound

#     def _one_pt_round(self, i):
#         X_tv_permuted = np.random.permutation(self.m.X_tv)
#         X_test_permuted = np.random.permutation(self.m.X_test)
#         self.estimator.fit(X_tv_permuted, self.m.y_tv)
#         y_pred = self.estimator.predict(X_test_permuted)
#         R = pd.DataFrame({
#             "y_true" : self.m.y_test, 
#             "y_pred" : y_pred, 
#             "confound" : self.m.conf_dict_test[self.confound]
#         })
#         res = explain_var(R)
#         return(res["delta_predictions"], res["delta_confounds"])


#     def permutation_test(self, n_permutations=1000, n_jobs=None):
#         assert self.m.estimator, "Run 'm.run()' first!"
#         self.estimator = deepcopy(self.m.estimator)

#         results = {}

#         model_name = type(self.estimator["model"]).__name__
#         print("This is model {}".format(model_name))

#         self.estimator.fit(self.m.X_tv, self.m.y_tv)
#         y_pred = self.estimator.predict(self.m.X_test)
#         R = pd.DataFrame({
#             "y_true" : self.m.y_test, 
#             "y_pred" : y_pred, 
#             "confound" : self.m.conf_dict_test[self.confound]
#         })
#         res = explain_var(R)
#         score = res["delta_predictions"]

#         d2pred_scores = np.zeros(n_permutations)
#         d2conf_scores
#         for i in range(n_permutations):
#             d2pred_scores[i], d2conf_scores[i] = self._one_pt_round(i)
            
#         C = sum(self.permutation_scores > score)
#         pvalue = (C+1)/(n_permutations+1)
#         print("The P-value is {}".format(pvalue))
#         results = {
#             "permutation_scores" : list(self.permutation_scores.astype(float)), 
#             "pvalue" : pvalue, 
#             "score" : score
#         }
#         return(results)


#     def plot(self, results):
#         df = pd.DataFrame(results)
#         df = df.reset_index().melt(id_vars="index", var_name='model', value_name="score")
#         df = df.rename(columns={"index":"q"})
#         df = df.query("q in ['delta_confounds', 'delta_predictions', 'shared']")
#         df.q = df.q.map({"delta_confounds":"confounds only", "delta_predictions":"predictions only", "shared":"confounds and predictions"})

#         from plotnine import ggplot, aes, geom_bar, coord_flip, labs, theme_classic, theme, scale_fill_manual
#         plot = (ggplot(df, aes(x="model", y="score", fill="q")) +
#         geom_bar(stat='identity') +
#         coord_flip() +
#         labs(y=r"$D^2$", x="") +
#         theme_classic() +
#         theme(aspect_ratio = 0.25) +
#         scale_fill_manual(values = ["#F0E442", "#9CCA9D", "#56B4E9"],
#                             name='variance explained by'))
#         return(plot)



# def explain_var(df, datatype="categorical"):
#     if datatype == "categorical":
#         r2_conf = smf.logit("y_true ~ confound", data=df).fit().prsquared 
#         r2_pred = smf.logit("y_true ~ y_pred", data=df).fit().prsquared 
#         r2_conf_pred = smf.logit("y_true ~ y_pred + confound", data=df).fit().prsquared 
#     if datatype == "continuous":
#         r2_conf = smf.ols(formula="y_true ~ confound", data=df).fit().rsquared
#         r2_pred = smf.ols(formula="y_true ~ y_pred", data=df).fit().rsquared
#         r2_conf_pred = smf.ols(formula="y_true ~ y_pred + confound", data=df).fit().rsquared
#     return(_decompose_r2(r2_conf, r2_pred, r2_conf_pred))

# def _decompose_r2(r2_conf, r2_pred, r2_conf_pred):
#     conf_unexplained = 1 - r2_conf
#     pred_unexplained = 1 - r2_pred
#     delta_pred = r2_conf_pred - r2_conf
#     delta_conf = r2_conf_pred - r2_pred
#     partial_pred = delta_pred / conf_unexplained
#     partial_conf = delta_conf / pred_unexplained
#     shared = r2_conf_pred - delta_conf - delta_pred
#     res = {
#         "confounds" : r2_conf, 
#         "predictions" : r2_pred, 
#         "confounds+predictions" : r2_conf_pred, 
#         "delta_confounds" : delta_conf, 
#         "delta_predictions" : delta_pred, 
#         "partial_confounds" : partial_conf, 
#         "partial_predictions" : partial_pred, 
#         "shared" : shared
#     }
#     return(res)