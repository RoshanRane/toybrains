import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# function

def generate_dataset(raw_csv_path, label, CV=None, trial=0, random_seed=42, debug=False):
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
    # set random seed
    seed = random_seed
    
    # set raw csv path
    raw_csv_path = raw_csv_path
    
    # set target label
    label = label
    
    # load the raw csv
    DF = pd.read_csv(raw_csv_path)
    
    # assign target label
    if 'cov_age' == label:
        DF['label'] = DF['cov_age']
    else:
        DF['label'] = pd.factorize(DF[label])[0]
    
    # split dataset into 80% for remaining and 20% for test
    DF_training, DF_test = train_test_split(DF, test_size=0.2, random_state=seed)
    
    if CV == None:
        # split remaining 80% split it into 90% for training and 10% for validation
        DF_train, DF_val = train_test_split(DF_training, test_size=0.1, random_state=seed)
    else:
        assert CV >= 2, "CV must be at least 2"
        assert trial <= CV, "trial should be in CV number"
        
        skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=seed)
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
        print(f"Raw:   {len(DF)}\n"
              f"Train:  {len(DF_train)}\n"
              f"Val:    {len(DF_val)}\n"
              f"Test:   {len(DF_test)}")
    
    return DF_train, DF_val, DF_test