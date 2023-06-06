import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# function

def generate_dataset(raw_csv_path, label, random_seed=42, debug=False):
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
    (TODO) Refactoring `torch.utils.data.random_split` or support K-fold or stratified
    
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
    
    DF['label'] = DF[label].astype(int)
    
    # split dataset into 80% for training and 20% for remaining
    
    DF_train, DF_remaining = train_test_split(DF, test_size=0.2, random_state=seed)
    
    # split remaining 20% into 10% for validation and 10% for test
    
    DF_val, DF_test = train_test_split(DF_remaining, test_size=0.5, random_state=seed)
    
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