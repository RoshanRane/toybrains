# standard python packages
import os, sys
import random
from glob import glob
from datetime import datetime
from collections import Counter
from tqdm import tqdm
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

## create toybrains dataset as a function
def run_toybrains(n_samples, data_dir, config=None, debug=False):
    ''' run toybrains
    
    NOTE
    ----
    ! python create_toybrains.py --dir $data_dir -n $n_samples -c $config
    '''
    cmd = [
        "python",            
        "create_toybrains.py",
        "-n", str(n_samples),
        "--dir", str(data_dir),
        "-c", str(config),
    ]
    if debug: cmd += ["--debug"]
    subprocess.run(cmd)

## create toybrains dataset using data_dict
def generate_toybrains_list(data_dict, debug=False):
    ''' create a toybrains using parallel
    
    PARAMETER
    ---------
    data_dict : dictionary
        data_dict
        
    debug : boolean, default : False
        debug mode
    
    OUTPUT
    ------
    path_list : list
        list of csv path, element is a string
    '''
    
    def generate_toybrains(data_dir, args, debug):
        n_samples, config, img = args['n_samples'], args['config'], args['img']
        
        if img:
            if debug: print(f"Save toybrains dataset (N={n_samples}) in 'dataset/{data_dir}'")
            # compatible with CLI and notebook
            run_toybrains(n_samples, data_dir, config, debug)
            # ! python create_toybrains.py --dir $data_dir -n $n_samples -c $config
            if debug: print(f'summary can be found in {csv_path}\n')
            
        csv_path = f"dataset/{data_dir}/toybrains_n{n_samples}.csv"
        return csv_path
    
    n_jobs = len(data_dict) if len(data_dict) < 10 else 10
    # parallelize the loop
    path_list = Parallel(n_jobs=n_jobs)(delayed(generate_toybrains)(
        data_dir, args, debug
    ) for data_dir, args in data_dict.items())
    
    return path_list

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
    raw_csv_path,
    DATA_DIR,
    DATA_N,
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
    raw_csv_path : string
        toybrains_n*.csv path
    DATA_DIR : string
        raw_csv_path input directory
    DATA_N : string
        n samples
    OUT_DIR : string path or pathlib.PosixPath
        run.csv output directory
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
    
    common = {
        "dataset" : DATA_DIR,
        "type" : "baseline",
        "n_samples" : DATA_N,
        "CV" : CV,
    }
    
    DF = pd.read_csv(raw_csv_path)
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
    
## run toybrains dataset baseline pipeline
def run_baseline_pipeline(
    data_dict,
    CV=10,
    N_JOBS=5,
    random_seed=42,
    debug=False
):
    ''' run baseline pipeline
    
    baseline on attributes and covariates
    
    PARAMETERS
    ----------
    data_dict : dictionary
        toybrains data dictionary
        
    CV : int
        cross validation
        
    N_JOBS : int
        N jobs
        
    random_seed : int
        random seed
    
    debug : boolean
        debug mode
        
    OUTPUT
    ------
    out_path_list : list
        list of run.csv absolute path
    
    NOTE
    ----
    output saved in the directory called dataset directory key
    '''
    
    start_time = datetime.now()
    # generate the toybrains dataset
    csv_path_list = generate_toybrains_list(data_dict, debug=debug)
    out_path_list = []
    # loop for data_dict
    for (DATA_DIR, DATA), raw_csv_path in zip(data_dict.items(), csv_path_list):
        # TODO debug mode only try one
        DATA_N = DATA['n_samples']
        CONFIG = DATA['config']
        OUT_DIR = Path.cwd() / 'results' / DATA_DIR / start_time.strftime("%Y%m%d-%H%M")
        OUT_DIR.mkdir(parents = True, exist_ok = True)
        out_path = OUT_DIR / "run.csv"
        out_path_list += [str(out_path.resolve())]
        
        print(f"{'#'*95}\nRunning Baseline on\nDATA DIR: {DATA_DIR}\nN SAMPLES: {DATA_N}\nOUTPUT DIR: {OUT_DIR}\n{'#'*95}")
        run_baseline(
            raw_csv_path = raw_csv_path,
            DATA_DIR = DATA_DIR,
            DATA_N = DATA_N,
            OUT_DIR = OUT_DIR,
            CV = CV,
            N_JOBS = N_JOBS,
            random_seed = random_seed,
            debug = debug,
        )
    
    runtime = str(datetime.now()-start_time).split(".")[0]
    print(f'TOTAL PIPELINE RUNTIME: {runtime}')
    return out_path_list

# visualization
def viz_baseline(
    output_list,
):
    ''' vizualization output
    
    PARAMETER
    ---------
    output_list : list
        run.csv of the output_list
    
    TODO add parameters
    
    '''
    num = len(output_list)
    if len(output_list) == 1:
        RUN = pd.read_csv(output_list[0])
    if len(output_list) != 1:
        DFS = [pd.read_csv(output_csv) for output_csv in output_list]
        RUN = pd.concat(DFS)
    
    viz_df = RUN.copy(deep=True)
    
    x = 'test_metric'
    y = 'out'
    hue = 'inp'
    y_order = ['lblbin_shp', 'lblbin_shp-vent', 'lblbin_shp-vol', 'cov_sex', 'cov_site', 'cov_age']
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
    
        ax = sns.pointplot(y=y, x=x, order=y_order,
                           hue=hue, hue_order=hue_order,
                           join=join, data=dfn, ax=ax,
                           errorbar='sd', errwidth=errwidth, capsize=capsize,
                           dodge=dodge, scale=scale, palette=palette)
    
        ax.legend_.remove()
        ax.set_title(f"{dataset}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    
        # draw the chance line in the legend
        chance = [0, 0.25, 0.5, 0.5, 0.5, 0.5]
        for z, ch in enumerate(chance):
            # print(f'z={z}, ch={ch}'
            # print(f'ymin={z/len(chance)}, ymax={(z+1)/(len(chance))}')
            ax.axvline(x=ch, ymin=z/len(chance), ymax=(z+1)/(len(chance)), label="chance", c='gray', ls='--', lw=1.5)
        ax.axhline(y=3.5, xmin=0.29, xmax=0.5, c='gray', ls='--', lw=1.5)
        ax.axhline(y=4.5, xmin=0.09, xmax=0.29, c='gray', ls='--', lw=1.5)

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

    return viz_df
    
# summary
def summary(
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