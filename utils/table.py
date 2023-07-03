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

def _get_columns(df, data_type):
    '''
    generate the columns
    '''
    DF = df.copy()
    
    DF_columns = []
    
    if 'x' in data_type:
        New_columns = DF.columns[DF.columns.str.startswith('gen')].tolist()
        DF_columns += New_columns
    
    if 'c' in data_type:
        New_columns = DF.columns[DF.columns.str.startswith('cov')].tolist()
        DF_columns += New_columns
    
    return DF_columns

def _get_data(df, data_type='x'):
    '''
    get strucutral data using data_type criteria
    '''
    
    assert data_type in ['x', 'cx', 'xc'], "data type should be either x, or cx, or xc"
    
    DF = df.copy()
    
    # set the target label
    
    target = list(DF['label'])
    
    # set the data using data_type
    
    columns = _get_columns(DF, data_type)
    
    data = DF[columns]
    
    return data, target

def get_table_loader(dataset, data_type='x', seed=42):
    '''
    get structural data return to data
    
    PARAMETER
    ---------
    dataset : tuple
        tuple of (DF_train, DF_val, DF_test)
        
    data_type : string, defuault : x
        select the input type either x, c, or xc
    
    seed : integer, default : 42
        random seed
    '''
    
    DF_train, DF_val, DF_test = dataset
    
    data_train, target_train = _get_data(df = DF_train, data_type = data_type)
    data_val, target_val = _get_data(df = DF_val, data_type = data_type)
    data_test, target_test = _get_data(df = DF_test, data_type = data_type)
    
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
    
    num = len(set(target_train))
    
    if num == 2:
        pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=1000, random_state=42))
        parameters = {'logisticregression__C': [0.1, 1, 10, 100]}
    
    # if num == 4:
    #     pipe = make_pipeline(preprocessor, OneVsRestClassifier(SVC()))
    #     parameters = {'onevsrestclassifier': [0.1, 1, 10, 100]}
        
    if num > 4:
        pipe = make_pipeline(preprocessor, LinearRegression())
        parameters = {'linearregression__fit_intercept': [True, False]}
    
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