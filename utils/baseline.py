from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
import numpy as np
import shap
from matplotlib import pyplot as plt
from IPython.display import display
from utils.metrics import d2_metric_probas
from scipy.special import logit
import pickle

def run_lreg(data, compute_shap=False, 
             random_state=None):

    (data_train, target_train), (data_val, target_val), (data_test, target_test) = data
    
    cat_col_names_selector = selector(dtype_include=object)
    cont_col_names_selector = selector(dtype_exclude=object)
    
    categorical_preprocessor = OneHotEncoder(handle_unknown='ignore')
    continuous_preprocessor = StandardScaler()
    
    # select continuous columns
    cont_col_names = cont_col_names_selector(data_train)
    
    # select categorical columns
    cat_col_names = cat_col_names_selector(data_train)
    preprocessor = ColumnTransformer(
        [
            ("one-hot-encoder", categorical_preprocessor, cat_col_names),
            ("minmax_scaler", continuous_preprocessor, cont_col_names),
        ]
    )
    
    # TODO Refactoring needed
    n_classes = len(set(target_train))
    # binary labels
    if n_classes == 2: 
        model_name = 'logistic_regression'
        pipe = make_pipeline(preprocessor, 
                             LogisticRegression(max_iter=2000, random_state=random_state,
                                                penalty='l2',  solver='lbfgs'))
        parameters = {'logisticregression__C': [1/0.5,1,1/5, 1/10]}
        metric_name = "r2-pseudo"
        metric = make_scorer(d2_metric_probas, needs_proba=True)
        
    
    # multiclass labels
    elif n_classes <= 5:
        model_name = 'multinomial_logistic_regression' 
        pipe = make_pipeline(preprocessor, 
                             LogisticRegression(max_iter=2000, random_state=random_state, 
                                                multi_class='multinomial', 
                                                penalty='l2', solver='lbfgs'))
        parameters = {'logisticregression__C': [1/0.5,1,1/5,1/10]}
        metric_name = "r2-pseudo"
        metric = make_scorer(d2_metric_probas, needs_proba=True)
        
    # regression label
    else: # TODO test this
        model_name = 'linear_regression'
        pipe = make_pipeline(preprocessor, 
                             Ridge(random_state=random_state))
        parameters = {'ridge__alpha': [0.1,0.5,1,5]}
        metric_name = metric = "r2" 
    
    # Use GridSearchCV to find the optimal hyperparameters for the pipeline
    clf = GridSearchCV(pipe, param_grid=parameters, scoring=metric)
    
    # Train and fit logistic regression model
    clf.fit(data_train, target_train)
    
    # Predict using the trained model
    tr_acc = clf.score(data_train, target_train)
    vl_acc = clf.score(data_val, target_val)
    te_acc = clf.score(data_test, target_test)

    # SHAP explanations
    shap_contrib_scores = None
    if compute_shap:
        preprocessing, best_model = clf.best_estimator_[:-1], clf.best_estimator_[-1]
        # print("[D] best model = ", best_model)
        data_train_processed = preprocessing.transform(data_train)
        data_val_processed = preprocessing.transform(data_val)
        data_test_processed = preprocessing.transform(data_test)
        all_data_processed = np.concatenate((data_train_processed,
                                             data_val_processed, 
                                             data_test_processed), axis=0)
         # transform the existing feature_names to include the one-hot encoded features
        feature_names = data_train.columns.tolist()
        new_feature_names = preprocessing['columntransformer'].get_feature_names_out(feature_names)
        n_feas = len(new_feature_names)
        # remove preprocessor names from feature names
        new_feature_names = [name.split("__")[-1] for name in new_feature_names]
        explainer = shap.Explainer(best_model, 
                                   data_train_processed,
                                   feature_names=new_feature_names)
        shap_values = explainer(all_data_processed)

        # get the model predicted probabilities to calculate C = probas - base for each sample
        model_probas = best_model.predict_proba(all_data_processed) 
        #  I verified that the shap values correspond to the second proba dim and not the first
        model_probas = model_probas[:,1].squeeze()      
        # calculate C = probas - base for each sample
        logodds_adjusted = logit(model_probas) - shap_values.base_values 
        # scale shap values to positive values [0,inf] for each sample X
        shap_val_mins = shap_values.values.min(axis=1)
        shap_values_pos = (shap_values.values - shap_val_mins[:,np.newaxis])
        # also apply these transforms to the RHS (logodds_centered) n_feas times
        logodds_adjusted = (logodds_adjusted - n_feas*shap_val_mins)
        # calculate shap value based contrib score for each feature
        contribs = shap_values_pos / logodds_adjusted[:,np.newaxis]
        contribs_avg = contribs.mean(axis=0) 
        # min max scale the avg contribs to [0,1]
        contribs_avg = (contribs_avg - contribs_avg.min())/(contribs_avg.max() - contribs_avg.min())
        # scale it to sum to 1
        contribs_avg = contribs_avg / contribs_avg.sum()
#         fi = 37
#         print('[D] f={} sum(contrib[f])[:5] = {} \t sum(contrib_avg)={}\
#  \ncontribs[:5,f]     \t= {} \
#  \nShap_scaled[:5,f]  \t= {} \
#  \nlogodds_adjusted[:5] \t= {}'.format(new_feature_names[fi], contribs[:5].sum(axis=1), contribs_avg.sum(),
#             contribs[:5,fi], shap_values_pos[:5,fi],
#             logodds_adjusted[:5]))
#         print("[D]", contribs_avg)
        # calculate mean of absolute shaps for each feature
        # contribs = np.abs(shap_values.values)
        # shap_contrib_scores = np.abs(best_model.coef_).squeeze().tolist() # model coefficients
        shap_contrib_scores = [(fea_name, contribs_avg[i]) \
                               for i, fea_name in enumerate(new_feature_names)]  #contribs[:,i].std()

    results = {"train_metric":tr_acc, "val_metric":vl_acc, "test_metric":te_acc, 
               "shap_contrib_scores": shap_contrib_scores}
    settings = {"model":model_name, "metric":metric_name, "model_config":pipe}
    return results, settings




