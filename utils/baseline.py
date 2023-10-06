# standard python packages
import os, sys, shutil
import random
from glob import glob
from datetime import datetime
from collections import Counter
from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import subprocess
from joblib import Parallel, delayed

# add custom imports
from utils.dataset import generate_dataset
from utils.tabular import get_table_loader, run_lreg

sys.path.insert(0, "../")
sns.set_theme(style="ticks", palette="pastel")

# functions

## TODO make it a class with 2 methods: run and plot
# run one using settings
def run(
    label,
    feature,
    trial,
    raw_csv_path,
    CV,
    OUT_DIR,
    common,
    random_seed,
    debug,
):
    '''
    run one baesline: label X feature X trial
    ''' 
    # split the dataset for training, validation, and test from raw dataset
    dataset = generate_dataset(raw_csv_path, label, CV, trial, random_seed, debug)
        
    # load the dataset
    data = get_table_loader(dataset=dataset, label=label, data_type=feature, random_seed=random_seed)
    if debug: print(f'Inputs: {data[0].columns}')

    # run logistic regression and linear regression for tabular dataset
    output, pipe = run_lreg(data)
    train_metric, val_metric, test_metric = output
    num = pipe[0]
    if num == 2: model_name = 'logistic_regression'
    if num == 4: model_name = 'multinomial_logistic_regression'
    if num > 4: model_name = 'linear_regression'
        
    if debug:
        print(f"Train metric: {train_metric:>8.4f} "
              f"Validation metric: {val_metric:>8.4f} "
              f"Test metric: {test_metric:>8.4f}")
        
    result = {
        "inp" : feature,
        "out" : label,
        "trial" : trial,
        "model" : model_name,
        "model_config" : pipe,
        "train_metric" : train_metric,
        "val_metric" : val_metric,
        "test_metric" : test_metric
    }
        
    result.update(common)
    df = pd.DataFrame([result])
    df.to_csv(f"{str(OUT_DIR)}/run_bsl_{label}_{feature}_{trial}_of_{CV}_{model_name}.csv", index=False)

    
# run baseline on both attributes and covariates
def run_baseline(
                DATA_DIR, 
                OUT_DIR,
                CV = 10,
                N_JOBS = 5,
                random_seed = 42,
                debug = False,
):
    ''' run baseline 
    
    labels x data_type are fixed. More details can be found in NOTE below
    
    PARAMETERS
    ----------
    DATA_DIR : string
         dataset directory containing the .csv table and the images/*.jpg
    OUT_DIR : output directory where the run.csv results will be stored
    CV : int, default : 10
        number of cross validation
    random_seed : int, default : 42
        random state for reproducibility
    debug : boolean, default : False
        debug mode
    
    NOTE
    ----
    support [input features] X [output labels] X [cross validation]
    
    input features : a, a+c, c
    output labels : lblbin_shp, lblbin_shp-vol, lblbin_shp-vent, cov_sex, cov_age, cov_site
    model : logistic regression, multiple logistic regression, linear regression
    '''
    start_time = datetime.now()
    
    # recreate output dirs
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.makedirs(OUT_DIR)
    
    raw_csv_path = glob(DATA_DIR+"/*.csv")
    assert len(raw_csv_path)==1, f"{len(raw_csv_path)} dataset tables found in {DATA_DIR}" 
    raw_csv_path = raw_csv_path[0]
    DF = pd.read_csv(raw_csv_path)
    
    common = {
        "dataset" : DATA_DIR,
        "type" : "baseline",
        "n_samples" : len(DF),
        "CV" : CV,
    }
    labels = DF.columns[DF.columns.str.startswith('lbl')].tolist() + DF.columns[DF.columns.str.startswith('cov')].tolist()
    features = ['a', 'a+c', 'c']
    
    # generate the different settings
    all_settings = []
    for lbl in labels:
        for ftr in features:
            for trl in range(CV):
                all_settings.append((lbl,ftr,trl))
    print(f'running a total of {len(all_settings)} different settings of [input features] x [output labels] x [cross validation]')
    
    with Parallel(n_jobs=N_JOBS) as parallel:
        parallel(delayed(run)(
            label=label,
            feature=feature,
            trial=trial,
            raw_csv_path=raw_csv_path,
            CV=CV,
            OUT_DIR=OUT_DIR,
            common=common,
            random_seed=random_seed,
            debug=debug) for label, feature, trial in tqdm(all_settings))
    
    # merge run_*.csv into one run.csv
    df = pd.concat([pd.read_csv(csv) for csv in glob(f"{str(OUT_DIR)}/run_*.csv")], ignore_index=True)
    df = df.sort_values(["dataset", "inp", "out", "type", "trial", "model"]) # sort
    # (TODO) Reorder columns for readability
    df.to_csv(f"{str(OUT_DIR)}/run.csv", index=False)
    
    # delete the temp csv files
    os.system(f"rm {str(OUT_DIR)}/run_*.csv")
    
    runtime = str(datetime.now()-start_time).split(".")[0]
    print(f'TOTAL RUNTIME: {runtime}')
    return df
    

# visualization
def viz_baseline(run_results):
    ''' vizualization output of baseline models
    '''
    if isinstance(run_results, str) and os.path.exists(run_results):
        RUN = pd.read_csv(run_results)
    elif isinstance(run_results, pd.DataFrame):
        RUN = run_results
    else:
        raise ValueError(f"{run_results} is neither a path to the results csv nor a pandas dataframe")
    
    viz_df = RUN.copy(deep=True)
    
    x = 'test_metric'
    y = 'out'
    hue = 'inp'
    hue_order = ['a', 'c', 'a+c']
    datasets = list(viz_df['dataset'].unique())
    num_rows = len(datasets)
    join = False

    # setup the figure properties
    sns.set(style='whitegrid', context='paper')
    fig, axes = plt.subplots(num_rows, 1,
                             sharex=True, sharey=True,
                             dpi=120, figsize=(7,5))

    if num_rows == 1: axes = [axes]
    plt.xlim([-0.1, 1.1])

    for ax, dataset in zip(axes, datasets):
        dfn = viz_df.query(f"dataset == '{dataset}'")

        # plotting details
        palette = sns.color_palette()
        dodge, scale, errwidth, capsize = 0.4, 0.4, 0.9, 0.08
    
        ax = sns.pointplot(y=y, x=x, 
                           hue=hue, hue_order=hue_order,
                           join=join, data=dfn, ax=ax,
                           errorbar='sd', errwidth=errwidth, capsize=capsize,
                           dodge=dodge, scale=scale, palette=palette)
    
        ax.legend_.remove()
        ax.set_title(f"{dataset}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    
        # draw the chance line in the legend
        ax.axvline(x=0.5, label="chance", c='gray', ls='--', lw=1.5)

    # set custom x-axis tick positions and labels
    x_tick_positions = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    x_tick_labels = ['0', '20', '40', '60', '80', '100']

    for ax in axes:
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels)

    for ax in axes[:-1]:
        ax.set_xlabel("")

    # add labels for the last subplot
    axes[-1].set_xlabel(f"{x} - Accuracy (%)")

    legend_handles = []

    legend_labels = ["Attributes (A)", "Covariates (C)", "A+C"]
    for i, label in enumerate(legend_labels):
        legend_handles.append(plt.Line2D([0], [0], marker='', linestyle='-', color=palette[i], label=label))

    # add the legend outside the subplots
    plt.legend(handles=legend_handles, loc='upper right', title='Inputs', fontsize=7, bbox_to_anchor=(1.0, 0.4))
    
    plt.suptitle("Baseline Analysis Plot")
    plt.tight_layout()
    plt.show()
    # plt.savefig("figures/results_bl.pdf", bbox_inches='tight')

    
# summary
def results_summary(
    df,
    col=None,
    cmap='YlOrBr',
):
    ''' summary 
    
    PARAMETER
    ---------
    DF : pandas.dataframe
        run.csv
        
    col : None or string
        target columns if None then display all the metric
    '''
    assert col in [None, 'train_metric', 'val_metric', 'test_metric']
    
    desc = pd.DataFrame(df.groupby(['out', 'inp', 'dataset'])['train_metric', 'val_metric', 'test_metric'].describe())
    if col is None:
        desc = desc[
            [('train_metric', 'mean'), ('train_metric', 'std'), ('train_metric', 'min'), ('train_metric', 'max'),
             ('val_metric', 'mean'), ('val_metric', 'std'), ('val_metric', 'min'), ('val_metric', 'max'),
             ('test_metric', 'mean'), ('test_metric', 'std'), ('test_metric', 'min'), ('test_metric', 'max')]]
    else:
        desc['average'] = desc[(col, 'mean')].values
        desc['standard_deviation'] = desc[(col, 'std')].values
        desc['min'] = desc[(col, 'min')].values
        desc['max'] = desc[('train_metric', 'max')].values
        desc = desc[['average', 'standard_deviation', 'min', 'max']]
        
    return desc.style.background_gradient(cmap=cmap)