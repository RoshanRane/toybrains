from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.svm import SVC
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
    shap_vals_tuples = None
    if compute_shap:
        preprocessing, best_model = clf.best_estimator_[:-1], clf.best_estimator_[-1]
        # print("[D] best model = ", best_model)
        data_train_processed = preprocessing.transform(data_train)
        data_test_processed = preprocessing.transform(data_test)
        
         # transform the existing feature_names to include the one-hot encoded features
        feature_names = data_train.columns.tolist()
        new_feature_names = preprocessing['columntransformer'].get_feature_names_out(feature_names)
        # remove preprocessor names from feature names
        new_feature_names = [name.split("__")[-1] for name in new_feature_names]
        explainer = shap.Explainer(best_model, 
                                   data_train_processed,
                                   feature_names=new_feature_names)
        shap_values = explainer(data_test_processed)

        shap_vals_avg = np.abs(shap_values.values).mean(0)
        # shap_vals_avg = np.abs(best_model.coef_).squeeze().tolist() # model coefficients
        shap_vals_tuples = sorted(list(zip(new_feature_names, shap_vals_avg)), key=lambda x: x[-1], reverse=True) 

    results = {"train_metric":tr_acc, "val_metric":vl_acc, "test_metric":te_acc, 
               "shap_values":shap_vals_tuples}
    settings = {"model":model_name, "metric":metric_name, "model_config":pipe}
    return results, settings




