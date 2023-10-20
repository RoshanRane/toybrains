from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
import numpy as np

from utils.metrics import d2_metric_probas


def run_lreg(data):

    (data_train, target_train), (data_val, target_val), (data_test, target_test) = data
    
    categorical_columns_selector = selector(dtype_include=object)
    continuous_columns_selector = selector(dtype_exclude=object)
    
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    continuous_preprocessor = StandardScaler()
    
    # select continuous columns
    continuous_columns = continuous_columns_selector(data_train)
    
    # select categorical columns
    categorical_columns = categorical_columns_selector(data_train)
    
    preprocessor = ColumnTransformer(
        [
            ("one-hot-encoder", categorical_preprocessor, categorical_columns),
            ("standard_scaler", continuous_preprocessor, continuous_columns),
        ]
    )
    
    # TODO Refactoring needed
    num = len(set(target_train))
    # binary labels
    if num == 2: 
        model_name = 'logistic_regression'
        pipe = make_pipeline(preprocessor, 
                             LogisticRegression(max_iter=1000, random_state=42))
        parameters = {'logisticregression__C': [0.1, 1, 10, 100]}
        metric_name = "r2-pseudo"
        metric = make_scorer(d2_metric_probas, needs_proba=True)
        
    
    # multiclass labels
    elif num < 5:
        model_name = 'multinomial_logistic_regression' 
        pipe = make_pipeline(preprocessor, 
                             LogisticRegression(max_iter=1000, random_state=42, 
                                                multi_class='multinomial', 
                                                solver='lbfgs'))
        parameters = {'logisticregression__C': [0.1, 1, 10, 100]}
        metric_name = "r2-pseudo"
        metric = make_scorer(d2_metric_probas, needs_proba=True)
        
    # regression label
    else:
        model_name = 'linear_regression'
        pipe = make_pipeline(preprocessor, 
                             LinearRegression())
        parameters = {'linearregression__fit_intercept': [True, False]}
        metric_name = metric = "r2" 
    
    # Use GridSearchCV to find the optimal hyperparameters for the pipeline
    clf = GridSearchCV(pipe, param_grid=parameters, scoring=metric)
    
    # Train and fit logistic regression model
    clf.fit(data_train, target_train)
    
    # Predict using the trained model
    tr_acc = clf.score(data_train, target_train)
    vl_acc = clf.score(data_val, target_val)
    te_acc = clf.score(data_test, target_test)
    
    results = {"train_metric":tr_acc, "val_metric":vl_acc, "test_metric":te_acc}
    settings = {"model":model_name, "metric":metric_name, "model_config":pipe}
    return results, settings




