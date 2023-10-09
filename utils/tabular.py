from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# helper

def _get_data(df, label, data_type='attr'):
    '''get tabula data using data_type criteria'''
    
    assert data_type in ['attr',  'cov', 'attr+cov', 'cov+attr'], \
"data type should be one of ['attr',  'cov', 'attr+cov', 'cov+attr']"
    assert label in df.columns, f"label {label} should be in dataframe"
    
    DF = df.copy()
    
    # set the target label
    target = list(DF['label'])
    
    # set the data using data_type
    columns = []
    if 'attr' in data_type:
        new_columns = DF.columns[DF.columns.str.startswith('gen')].tolist()
        columns += new_columns
    if 'cov' in data_type:
        new_columns = DF.columns[DF.columns.str.startswith('cov')].tolist()
        columns += new_columns
        if label in columns: columns.remove(label)
    
    data = DF[columns]
    
    return data, target

# function
# deviance function
def explained_deviance(y_true, y_pred_logits=None, y_pred_probas=None, 
                       returnloglikes=False):
    """Computes explained_deviance score to be comparable to explained_variance"""
    
    assert y_pred_logits is not None or y_pred_probas is not None, "Either the predicted probabilities \
(y_pred_probas) or the predicted logit values (y_pred_logits) should be provided. But neither of the two were provided."
    
    if y_pred_logits is not None and y_pred_probas is None:
        # check if binary or multiclass classification
        if y_pred_logits.ndim == 1: 
            y_pred_probas = expit(y_pred_logits)
        elif y_pred_logits.ndim == 2: 
            y_pred_probas = softmax(y_pred_logits)
        else: # invalid
            raise ValueError(f"logits passed seem to have incorrect shape of {y_pred_logits.shape}")
            
    if y_pred_probas.ndim == 1: y_pred_probas = np.stack([1-y_pred_probas, y_pred_probas], axis=-1)
    
    # compute a null model's predicted probability
    X_dummy = np.zeros(len(y_true))
    y_null_probas = DummyClassifier(strategy='prior').fit(X_dummy,y_true).predict_proba(X_dummy)
    #strategy : {"most_frequent", "prior", "stratified", "uniform",  "constant"}
    # suggestion from https://stackoverflow.com/a/53215317
    llf = -log_loss(y_true, y_pred_probas, normalize=False)
    llnull = -log_loss(y_true, y_null_probas, normalize=False)
    ### McFadden’s pseudo-R-squared: 1 - (llf / llnull)
    explained_deviance = 1 - (llf / llnull)
    ## Cox & Snell’s pseudo-R-squared: 1 - exp((llnull - llf)*(2/nobs))
    # explained_deviance = 1 - np.exp((llnull - llf) * (2 / len(y_pred_probas))) ## TODO, not implemented
    if returnloglikes:
        return explained_deviance, {'loglike_model':llf, 'loglike_null':llnull}
    else:
        return explained_deviance
    
def get_table_loader(dataset, label, data_type='attr', random_seed=42):
    '''
    get structural data return to data
    
    PARAMETER
    ---------
    dataset : tuple
        tuple of (DF_train, DF_val, DF_test)
        
    data_type : string, defuault : attr
        select the input type either 'attr',  'cov', 'attr+cov', 'cov+attr'
    
    seed : integer, default : 42
        random seed
    '''
    
    DF_train, DF_val, DF_test = dataset
    
    data_train, target_train = _get_data(df=DF_train, label=label, data_type=data_type)
    data_val, target_val = _get_data(df=DF_val, label=label, data_type=data_type)
    data_test, target_test = _get_data(df=DF_test, label=label, data_type=data_type)
    
    return (data_train, target_train, data_val, target_val, data_test, target_test)

def run_lreg(data):

    (data_train, target_train, data_val, target_val, data_test, target_test) = data
    
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
        scoring = "balanced_accuracy"
    
    # multiclass labels
    elif num < 5:
        model_name = 'multinomial_logistic_regression'
        pipe = make_pipeline(preprocessor, 
                             LogisticRegression(max_iter=1000, random_state=42, 
                                                multi_class='multinomial', solver='lbfgs'))
        parameters = {'logisticregression__C': [0.1, 1, 10, 100]}
        scoring = "balanced_accuracy"
        
    # regression label
    else:
        model_name = 'linear_regression'
        pipe = make_pipeline(preprocessor, 
                             LinearRegression())
        parameters = {'linearregression__fit_intercept': [True, False]}
        scoring = "r2"
    
    # Use GridSearchCV to find the optimal hyperparameters for the pipeline
    clf = GridSearchCV(pipe, param_grid=parameters, scoring=scoring)
    
    # Train and fit logistic regression model
    clf.fit(data_train, target_train)
    
    # Predict using the trained model
    tr_acc = clf.score(data_train, target_train)
    vl_acc = clf.score(data_val, target_val)
    te_acc = clf.score(data_test, target_test)
    
    results = {"train_metric":tr_acc, "val_metric":vl_acc, "test_metric":te_acc}
    
    return results, model_name, scoring, pipe




