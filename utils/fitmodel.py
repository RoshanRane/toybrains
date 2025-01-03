import numpy as np

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import make_scorer, get_scorer, get_scorer_names

from scipy.special import logit
# import shap

# add custom imports
from utils.metrics import d2, logodds_metric, loglikelihood_ratio

def check_if_continuous(states):
    '''Check if the states of the label are continuous or not'''
    continuous = True
    for v in states:
        try:
            float(v)
        except ValueError:
            continuous = False
            break
    return continuous
    

def fitmodel(df_train, df_test,
            X_cols, y_col,
            model_name='LR', model_params={},
            holdout_data=None,
            # compute_shap=False, 
            random_seed=None,
            metrics=["r2"]):
    '''Fit a model to the given dataset and estimate the model performance using cross-validation.
    Also, estimate the SHAP scores if compute_shap is set to True.
    '''
    results = {}

    # if the label y is string formated (categories) then one-hot encode it
    if isinstance(df_train[y_col].iloc[0], str):
        df_train[y_col] = df_train[y_col].astype('category')
        df_test[y_col] = df_test[y_col].astype('category')
        # convert the categories to integers
        df_train[y_col] = df_train[y_col].cat.codes
        df_test[y_col] = df_test[y_col].cat.codes
    
    # split the data into input features and output labels
    train_X, train_y = df_train[X_cols], df_train[y_col]
    test_X, test_y = df_test[X_cols], df_test[y_col]
    
    # when X= attr_* filter continuous vs categorical columns and scale only the categorical
    if len(X_cols)<100:
        input_attrs = True
        cat_col_names_selector = selector(dtype_include=object)
        cont_col_names_selector = selector(dtype_exclude=object)
        
        cont_col_names = cont_col_names_selector(train_X)
        cat_col_names = cat_col_names_selector(train_X)

        categorical_preprocessor = OneHotEncoder(handle_unknown='ignore')
        continuous_preprocessor = StandardScaler()
        preprocessor = ColumnTransformer([
                        ("one-hot-encoder", categorical_preprocessor, cat_col_names),
                        ("minmax_scaler", continuous_preprocessor, cont_col_names),
                        ])
        
    else:
        input_attrs = False
        preprocessor = make_pipeline(
                        VarianceThreshold(), 
                        StandardScaler())
    
    # set the model and its hyperparameters
    n_classes = train_y.nunique()
    # check that all states of the label are numerical, if yes then its a regression task
    regression_task = False
    if n_classes>2:
        regression_task = check_if_continuous(train_y.unique())

    if model_name.upper() == 'LR':
        if regression_task: # use ElasticNet for regression and change model_params accordingly
            model_params_reg = dict(l1_ratio=0) # l2 by default
            # replace 'penalty' with 'l1_ratio' for ElasticNet
            if 'penalty' in model_params and model_params['penalty']=='l1':
                    model_params_reg.update(dict(l1_ratio=1))
            # replace 'C' with 'alpha' for ElasticNet
            if 'C' in model_params: model_params_reg.update(dict(alpha=1.0/model_params['C']))
            model = ElasticNet(random_state=random_seed,
                                **model_params_reg)
        else:
            # if no model_params are explicitly provided then default to rbf kernel 
            if 'penalty' not in model_params: model_params.update(dict(penalty='l2'))
            if 'C' not in model_params: model_params.update(dict(C=1.0))
            if 'solver' not in model_params: model_params.update(dict(solver='lbfgs'))
            # multiclass classification
            if n_classes > 2: model_params.update(dict(multi_class='multinomial'))

            model = LogisticRegression(max_iter=2000, random_state=random_seed,
                                            **model_params) 
    elif model_name.upper() == 'SVM':
        # if no model_params are explicitly provided then default to rbf kernel 
        if 'kernel' not in model_params: model_params.update(dict(kernel='rbf'))
        if model_params['kernel'] == 'linear':
            model_params.update(dict(penalty='l2', loss='squared_hinge', C=1.0))
            # add predict_proba function as LinearSVC does not have it
            def _predict_proba(self, X):
                logits = self.decision_function(X)
                probas = 1 / (1 + np.exp(-logits))
                return np.array([1 - probas, probas]).T
        else:
            if 'gamma' not in model_params: model_params.update(dict(gamma='scale'))

        if regression_task: # TODO test
            if model_params['kernel']=='linear':
                model_params_lin = model_params.copy()
                model_params_lin.pop('kernel', None)
                model = LinearSVR(random_state=random_seed, dual='auto', **model_params_lin)
                model.predict_proba = lambda X: _predict_proba(model, X)
            else:
                model = SVR(random_state=random_seed, probability=True,
                        **model_params)
            model = SVR(random_state=random_seed, probability=True,
                        **model_params)
        else:
            if model_params['kernel']=='linear':
                model_params_lin = model_params.copy()
                model_params_lin.pop('kernel', None)
                model = LinearSVC(random_state=random_seed, dual='auto', **model_params_lin)
                model.predict_proba = lambda X: _predict_proba(model, X)
            else:
                model = SVC(random_state=random_seed, probability=True,
                        **model_params)
            
    elif model_name.upper() == 'RF':
        if regression_task:
            if 'n_estimators' not in model_params: model_params.update(dict(n_estimators=200))
            if 'max_depth' not in model_params: model_params.update(dict(max_depth=5))
            model = RandomForestRegressor(random_state=random_seed,
                                        **model_params)
        else:
            model = RandomForestClassifier(random_state=random_seed,
                                        **model_params)
            
    elif model_name.upper() == 'MLP':                
        if 'hidden_layer_sizes' not in model_params:
            if input_attrs:
                model_params.update(dict(hidden_layer_sizes=(200,100,20)))
            else:
                model_params.update(dict(hidden_layer_sizes=(5000,100,20)))

        if 'max_iter' not in model_params: model_params.update(dict(max_iter=1000))
        if regression_task:
            model = MLPRegressor(random_state=random_seed, early_stopping=True,
                                **model_params)
        else:
            model = MLPClassifier(random_state=random_seed, early_stopping=True,
                                **model_params)
    else:
        ## TODO support sklearn.linear_model.RidgeClassifier, tree.DecisionTreeClassifier, svm.SVC, sklearn.svm.LinearSVC, 
        raise ValueError(f"model_name '{model_name}' is invalid.\
Currently supported models are ['LR', 'SVM', 'RF', 'MLP']")
    
    # Train and fit the model
    clf = make_pipeline(preprocessor, model)
    clf.fit(train_X, train_y)
    
    # Store the y_pred_probas[:,1] and the the y_true for using other metrics in the future
    if not regression_task:
        y_pred_probas = clf.predict_proba(test_X)
        y_pred_probas = y_pred_probas[:,-1]
    else:
        y_pred_probas = clf.predict(test_X)

    results.update({"y_pred_probas_test": y_pred_probas.tolist(),
                    "y_true_test": test_y.tolist(),
                    "y_true_probas_test": df_test[f'probas_{y_col}'].tolist(), 
                    "test_idx": df_test.index.tolist()
                    })
                    
    # also save the model coefficients if available
    if hasattr(clf[-1], 'coef_'):
        preprocessing, best_model = clf[:-1], clf[-1]
        coefs = best_model.coef_
        if coefs.ndim>1: coefs = coefs[0]
        # transform the existing feature_names to include the one-hot encoded features
        new_feature_names = preprocessing.get_feature_names_out(X_cols)
        # remove preprocessor names from feature names
        new_feature_names = [name.split("__")[-1] for name in new_feature_names]
        assert len(new_feature_names)==len(coefs), f"Number of features ({len(new_feature_names)}) \
and coefficients ({len(coefs)}) do not match after one-hot encoding"
        results.update({"model_coefs": [(fea_name, coef) for fea_name, coef in zip(new_feature_names, coefs)]})
                    
    # estimate all requested metrics using the best model   
    for metric_name in metrics:
        # if classification then use d2_metric_probas instead of r2
        metric_kwargs = {}
        if regression_task:
            metric_fn = get_scorer('r2') # [HACK] r2 is the fixed metric for regression
        elif metric_name.lower() == "d2" and not regression_task:     
            metric_fn = make_scorer(d2, response_method="predict_proba")
        elif "logodds" in metric_name.lower() and not regression_task:
            metric_type = metric_name.split("_")[-1]
            metric_fn = make_scorer(logodds_metric, response_method="predict_proba", metric=metric_type)
        elif metric_name.lower() == "loglikelihood_ratio" and not regression_task:
            metric_fn = make_scorer(loglikelihood_ratio, response_method="predict_proba")
        else:
            metric_fn = get_scorer(metric_name)
        
        results.update({f"score_train_{metric_name.replace('_', '-')}": metric_fn(clf, train_X, train_y),
                        f"score_test_{metric_name.replace('_', '-')}": metric_fn(clf, test_X, test_y)})
        
        # if an additional holdout dataset is provided then also estimate the score on it
        if holdout_data is not None and len(holdout_data)>0:
            for holdout_name, holdout_data_i in holdout_data.items():
                holdout_X, holdout_y = holdout_data_i[X_cols], holdout_data_i[y_col]
                # one hot encode the label if it is in string format
                if isinstance(holdout_y.iloc[0], str):
                    holdout_y = holdout_y.astype('category')
                    holdout_y = holdout_y.cat.codes

                results.update({f"score_holdout_{holdout_name.replace('_', '-')}_{metric_name.replace('_', '-')}": 
                                metric_fn(clf, holdout_X, holdout_y)})
                
                # Store the y_pred_probas[:1] and the the y_true for calculating other metrics in the future
                if not regression_task:
                    y_pred_probas_hold = clf.predict_proba(holdout_X)
                    y_pred_probas_hold = y_pred_probas_hold[:,-1]
                else:
                    y_pred_probas_hold = clf.predict(holdout_X)
                results.update({f"y_pred_probas_holdout_{holdout_name.replace('_', '-')}": y_pred_probas_hold.tolist(),
                                f"y_true_holdout_{holdout_name.replace('_', '-')}": holdout_y.tolist(),
                                f"y_true_probas_holdout_{holdout_name.replace('_', '-')}": holdout_data_i[f'probas_{y_col}'].tolist()})
                
    

    # SHAP explanations
    # shap_contrib_scores = None
#     if compute_shap:
#         preprocessing, best_model = clf[:-1], clf[-1]
#         # print("[D] best model = ", best_model)
#         data_train_processed = preprocessing.transform(train_X)
#         data_test_processed = preprocessing.transform(test_X)
#         all_data_processed = np.concatenate((data_train_processed,
#                                             data_test_processed), axis=0)
#         # transform the existing feature_names to include the one-hot encoded features
#         feature_names = train_X.columns.tolist()
#         new_feature_names = preprocessing.get_feature_names_out(feature_names)
#         n_feas = len(new_feature_names)
#         # remove preprocessor names from feature names
#         new_feature_names = [name.split("__")[-1] for name in new_feature_names]
#         explainer = shap.Explainer(best_model, 
#                                 data_train_processed,
#                                 feature_names=new_feature_names)
#         shap_values = explainer(all_data_processed)
#         base_shap_values = shap_values.base_values 
#         # get the model predicted probabilities to calculate C = probas - base for each sample
#         model_probas = best_model.predict_proba(all_data_processed) 
#         #  I verified that the shap values correspond to the second proba dim and not the first
#         model_probas = model_probas[:,1].squeeze()      
#         # calculate C = probas - base for each sample
#         logodds_adjusted = logit(model_probas) - base_shap_values
#         # now we expect the sum(shap_values) to be equal to logodds_adjusted for each sample
#         assert np.allclose(shap_values.values.sum(axis=1), logodds_adjusted), \
#             "sum(shap_values) != logodds_adjusted for some samples"
#         # scale shap values to positive values [0,inf] for each sample X
#         shap_val_mins = shap_values.values.min(axis=1)
#         shap_values_pos = (shap_values.values - shap_val_mins[:,np.newaxis])
#         # also apply these transforms to the RHS (logodds_centered) n_feas times
#         logodds_adjusted = (logodds_adjusted - n_feas*shap_val_mins)
#         # calculate shap value based contrib score for each feature
#         contribs = shap_values_pos / logodds_adjusted[:,np.newaxis]
        
#         contribs_avg = contribs.mean(axis=0) 

# #         fi = 37
# #         print('[D] f={} sum(contrib[f])[:5] = {} \t sum(contrib_avg)={}\
# #  \ncontribs[:5,f]     \t= {} \
# #  \nShap_scaled[:5,f]  \t= {} \
# #  \nlogodds_adjusted[:5] \t= {}'.format(new_feature_names[fi], contribs[:5].sum(axis=1), contribs_avg.sum(),
# #             contribs[:5,fi], shap_values_pos[:5,fi],
# #             logodds_adjusted[:5]))
# #         print("[D]", contribs.mean(), contribs_avg)
#         # calculate mean of absolute shaps for each feature
#         # contribs = np.abs(shap_values.values)
#         # shap_contrib_scores = np.abs(best_model.coef_).squeeze().tolist() # model coefficients
#         #min max scale the avg contribs to [0,1]
#         contribs_avg = (contribs_avg - contribs_avg.min())/(contribs_avg.max() - contribs_avg.min())
#         # contribs_avg = contribs_avg - contribs_avg.min()
#         #scale it to sum to 1
#         contribs_avg = contribs_avg / contribs_avg.sum()

#         shap_contrib_scores = [(fea_name, contribs_avg[i]) \
#                             for i, fea_name in enumerate(new_feature_names)]  #contribs[:,i].std()
#         results.update({"shap_contrib_scores": shap_contrib_scores})

    settings = {"model":model_name, "model_params":model_params,  "model_config":model}

    return results, settings