import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# function

def split_dataset(data_csv, label, CV=None, trial=0, random_seed=42, debug=False):
    '''
    generate the dataset
    
    PARAMETERS
    ----------
    raw_csv_path : string
        csv path
        
    label : string
        labels, 'lblbin_stop-smidl-bvol', 'lblbin_stop-smidl-vthick', 'lblbin_bvol-vthick'
    
    random_seed : integer, default : 42
        random seed number
    
    NOTE
    ----
    (TODO) change function to class
    (TODO) Refactoring support K-fold or stratified
    
    '''
    seed = random_seed
    # load the raw csv
    
    if isinstance(data_csv, str) and os.path.isfile(data_csv):
        DF = pd.read_csv(data_csv)
    elif isinstance(data_csv, pd.DataFrame):
        DF = data_csv.copy()
    else:
        raise ValueError(f"data_csv provided {data_csv} is neither a pandas Dataframe nor a path to a table.")
    # define target label 
    DF['label'] = DF[label]
    #TODO hardcoded dtype change to categorical - refactor
    # if 'cov_age' == label:
    #     DF['label'] = DF['cov_age']
    # else:
    #     DF['label'] = pd.factorize(DF[label])[0]
    
    # split dataset into 80% for remaining and 20% for test
    DF_training, DF_test = train_test_split(DF, test_size=0.2, random_state=seed)
    
    if CV == None:
        # split remaining 80% split it into 80% for training and 20% for validation
        DF_train, DF_val = train_test_split(DF_training, test_size=0.2, random_state=seed)
    else:
        assert CV >= 2, "CV must be at least 2"
        assert trial <= CV, "trial should be in CV number"
        
        skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=seed)
        # TODO remove hardcoded 
        if label == 'cov_age':
            n_grp = 1000
            DF_training['grp'] = pd.cut(DF_training['label'], n_grp, labels=False)
            target = DF_training.grp
            for trial_no, (train_idx, val_idx) in enumerate(skf.split(target, target)):
                if trial_no == trial:
                    DF_train = DF_training.iloc[train_idx]
                    DF_val = DF_training.iloc[val_idx]
                    break
        else:
            target = DF_training['label']
            for trial_no, (train_idx, val_idx) in enumerate(skf.split(DF_training, target)):
                if trial_no == trial:
                    DF_train = DF_training.iloc[train_idx]
                    DF_val = DF_training.iloc[val_idx]
                    break
    
    # reset the index
    DF_train.reset_index(inplace=True, drop=True)
    DF_val.reset_index(inplace=True, drop=True)
    DF_test.reset_index(inplace=True, drop=True)
    
    # print the number of rows in each dataframe
    if debug:
        print(f"Full dataset:   {len(DF)}\n"
              f"Train:  {len(DF_train)}\n"
              f"Val:    {len(DF_val)}\n"
              f"Test:   {len(DF_test)}")
    
    return DF_train, DF_val, DF_test