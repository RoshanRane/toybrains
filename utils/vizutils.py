# standard python packages
import os, sys
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.image as mpimg

import json
import matplotlib
from  matplotlib.ticker import FuncFormatter
from matplotlib.colors import is_color_like


def show_images(img_files, n_rows=1, n_cols=10, title=''):
    assert isinstance(img_files[0], (str,np.ndarray, np.generic)), "img_files \
should either me a string path to the image files or numpy arrays"   
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(n_cols,n_rows+0.5),  # +0.5 for title
                                sharex=True, sharey=True)
    axes = axes.ravel() if not isinstance(axes, matplotlib.axes.Axes) else [axes]
    
    if title is not None and title != '': fig.suptitle(title, fontsize=12)

    for i, img in enumerate(img_files):
        if i<len(axes):
            if isinstance(img, str):
                img = mpimg.imread(img)
            ax = axes[i]
            ax.imshow(img)
            ax.axis('off')

    
    plt.tight_layout()
    # plt.show()


def plot_col_counts(df, title=''):
    
    df_copy = df.reindex(sorted(df.columns), axis=1).copy()
    cols = df_copy.columns 
    # convert each column to its appropriate dtype and 
    # then decide the type of plot to use for it
    fs=10
    plottypes = {}
    
    for col in cols:
        # don't plot the probas columns
        if col.startswith('probas_'): continue 
        if df_copy[col].nunique()==2:
            plottypes.update({col:'pie'})
        elif pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df[col].round(1) # round the floats to 1 decimal
            # use hist plot if there are more than 10 states  
            if df_copy[col].nunique()<10: ## hardcoded
                plottypes.update({col:'bar'})
            else:
                plottypes.update({col:'hist'})
            # # if all values are int then covert col to int dtype
            # if df_copy[col].dropna().apply(float.is_integer).all():
            #     df_copy[col] = df_copy[col].astype(int)
        else: # not pd.dtypes.is_numeric_dtype(df_copy[col]):
            plottypes.update({col:'bar'})
            
    # re-sort dataframe cols by dtypes
    cols = df_copy.dtypes.sort_values().index 
    # decide the number of rows and columns in the plot
    subplot_ncols = 5 if len(cols)>=5 else len(cols)
    subplot_nrows = len(cols)//subplot_ncols
    subplot_overflows = len(cols)%subplot_ncols
    if subplot_overflows!=0: subplot_nrows+=1
    # create subplots set attributes
    f,axes = plt.subplots(subplot_nrows, subplot_ncols, 
                          figsize=(1+2*subplot_ncols,1+2*subplot_nrows))
    axes = axes.ravel() if not isinstance(axes, matplotlib.axes.Axes) else [axes]
    if title: f.suptitle(title, fontsize=fs+2)
    # f.supylabel("Count", fontsize=fs)

    for i, ax in enumerate(axes):
        # print('[D]',col, plottypes[col])
        if i >= subplot_ncols*(len(cols)//subplot_ncols) + subplot_overflows:
            ax.axis('off')
            continue
            
        col = cols[i]
        plottype = plottypes[col]
        
        df_copy = df_copy.sort_values(by=col)
        if plottype == 'bar':
            # check if the attribute represents colors then use the same color names for the bar plot
            if isinstance(df_copy[col].iloc[0], str) and is_color_like(df_copy[col].iloc[0].split('-')[-1]):
                sns.countplot(data=df_copy, x=col, ax=ax, 
                             order=sorted(df_copy[col].unique()))
            else:
                sns.countplot(data=df_copy, x=col, ax=ax)
            # format the xtick labels 
            if pd.api.types.is_integer_dtype(df_copy[col]):
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
            elif pd.api.types.is_string_dtype(df_copy[col]):
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                
        elif plottype == 'hist':
            bins = df_copy[col].nunique()//5
            sns.histplot(data=df_copy, x=col, ax=ax, kde=False, bins=bins, ) #multiple='fill'
        elif plottype == 'pie':
            cnt = df_copy[col].value_counts().sort_index()
            ax.pie(cnt, labels=cnt.index,
                    colors=sns.color_palette('pastel'), autopct='%.0f%%')
            
        ax.set_ylabel(col, fontsize=fs)
        ax.set_xlabel(None)

    plt.tight_layout()
    
    
def plot_col_dists(df, attr_cols, cov_cols, title=''):
    
    # drop columns with 'probas_' prefix
    df = df.copy().drop(columns=[c for c in df.columns if c.startswith('probas_')])
    attr_cols = [c for c in attr_cols if c in df.columns]
    cov_cols = [c for c in cov_cols if c in df.columns]
    # subsample a max of 1000 samples to speed up the planning
    if len(df)>1000: df = df.sample(1000)

    # remove duplicates and sort
    cov_cols = sorted(list(set(cov_cols)))
    attr_cols = sorted(list(set(attr_cols)))

    # convert all columns to numerical
    for col in attr_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            vals =  {v:i for i,v in enumerate(sorted(df[col].unique()))}
            df[col] = df[col].map(vals)
            
    # if the covariates are continuous then bin them
    for col in cov_cols:
        if np.issubdtype(df[col].dtype, np.number) and df[col].nunique()>5:
            df[col] = pd.cut(df[col], bins=3, precision=0)
    
    subplot_nrows = len(cov_cols)
    subplot_ncols = len(attr_cols)
    fs=12
    # create subplots set attributes
    f,axes = plt.subplots(subplot_nrows, subplot_ncols, 
                          figsize=(2+1.5*subplot_ncols, 1+1.5*subplot_nrows),
                          sharex="col", sharey=True)
    axes = axes if axes.ndim>1 else axes.reshape(1,-1)
    if title: f.suptitle(title, fontsize=fs+4, y=1.01)
    
    for i, axis_row in enumerate(axes):
        for j, ax in enumerate(axis_row):
            draw_legend=False if j!=0 else True
            cov, attr = cov_cols[i], attr_cols[j]
            # if the attr is categorical then encode it to numerical
            if not pd.api.types.is_numeric_dtype(df[attr]):
                df[attr] = df[attr].astype('category').cat.codes
                df[attr] = df[attr].astype(int)
                
            if np.issubdtype(df[attr].dtype, str):
                df[attr] = df[attr].map({v:i for i,v in enumerate(sorted(df[attr].unique()))})

            # sort df by the covariate and attribute
            g = sns.kdeplot(df.sort_values(by=[cov]), 
                            x=attr, hue=cov, 
                            ax=ax, fill=True, legend=draw_legend,
                            bw_adjust=3, gridsize=10, cut=0, # adjust the smoothing
                            warn_singular=False)
            if draw_legend: 
                # make 2 cols if there are many lagend labels
                ncol = 2 if len(df[cov].unique())>3 else 1
                sns.move_legend(g, loc="upper left", 
                                bbox_to_anchor=(-1.5,1.), ncol=ncol,
                                frameon=True, 
                                title_fontproperties={'size':fs, 
                                                      'weight':'heavy'})
            # set xlabel and ylabel at the top and leftside of the plots resp.
            ax.set_ylabel(None)
            ax.set_xlabel(attr.replace('gen_',''), fontsize=fs) if i==len(axes)-1 else ax.set_xlabel(None)
            # fix the density range to 0, 0.5 to make them comparable
            ax.set_ylim(0,0.35)
            if i==0: ax.set_title(attr.replace('gen_',''), fontsize=fs)
            
    f.supylabel("Covariates & labels", fontsize=fs+2)
    
    plt.tight_layout()



#########################################        Baseline models        ########################################
def show_contrib_table(dfs_results, 
                       avg_over_trials=True,
                       filter_rows={}, filter_cols=[], 
                       percentages=True, 
                       color=None):
    '''reorganize the generated baseline results and display it as a pretty table with style:
    average the results across trials after grouping by ['dataset','out', 'inp']'''
    if isinstance(dfs_results, (list, tuple)): 
        dfs = pd.concat(dfs_results).copy()
    else:
        dfs = dfs_results.copy()
    
    # make the dataset name shorter for prettiness
    dfs = dfs.dropna(subset=["dataset"])
    dfs['dataset'] = dfs['dataset'].apply(lambda x: x.replace('/train/','').rstrip('/').split('/')[-1])
    # append model parameters info to the model name
    dfs['model'] = dfs.apply(lambda row: f"{row['model']}({row['model_params']})", axis=1)
    grp_by = ['out','inp','model','type','dataset']
    if not avg_over_trials: grp_by.append('trial')

    if filter_rows:
        for col, vals in filter_rows.items():
            include_nans = False
            # if None is provided then select NaN values
            for v in vals: 
                if v is None: 
                    include_nans = True
                    vals.remove(v)
            if include_nans:
                dfs = dfs[dfs[col].isna() | dfs[col].isin(vals)]
            else:
                dfs = dfs[dfs[col].isin(vals)]

    if filter_cols:
        dfs = dfs[grp_by+filter_cols]
        
    filter_cols = dfs.filter(regex='^score_').columns.tolist() + dfs.filter(regex='^shap__').columns.tolist()
    desc = dfs[grp_by+filter_cols].groupby(grp_by).mean()

    # format to percentages
    if percentages:
        func = lambda s: int(s*100) if pd.notnull(s) else -1
        desc = desc.map(func)
        print("All scores are converted to percentages (%)")

    return desc.style.bar(vmin=0, vmax=100, color=color) 



def viz_contribs_shap(df_results, top_n=4,  
                      show_individual_trials=False):
    
    if isinstance(df_results, (list, tuple)):
        df_results = pd.concat(df_results)
    
    # first, process all SHAP values and collect them in a dataframe
    plot_df = []
    for i, row in df_results.iterrows():
        shap_vals = row["shap_contrib_scores"]
        if not pd.isnull(shap_vals):
            # print("[D]", shap_vals)
            shap_vals = dict(eval(shap_vals))
            run_name = "{}<--{} trial{}\n Acc = {:.0f}%".format(row['out'], 
                                                                row['inp'].replace('attr_','').replace(' ','-'),
                                                                row['trial'], 
                                                                row['test_metric']*100)
            plot_dfi = pd.Series(shap_vals, name=run_name)
            # make the shap values sum to 1
            plot_dfi = plot_dfi/plot_dfi.sum()
            # additionally save other columns for later use
            plot_dfi = pd.concat([plot_dfi, pd.Series({
                "model":"{}<--{}".format(row['out'], 
                                            row['inp'].replace('attr_','').replace(' ','-')), 
                'dataset':row['dataset'],
                'out':row['out'],
                'inp':row['inp'],
                'trial':row['trial'],
                'accuracy':row['test_metric']}, name=run_name)])
            plot_df.append(plot_dfi)
    plot_df = pd.DataFrame(plot_df)
    # sort by column name alphabetically
    plot_df = plot_df.reindex(sorted(plot_df.columns), axis=1)

    # (1) Plot the |SHAP| scores in each indivudual trial for all features
    if show_individual_trials:
        for dataset, plot_dfi in plot_df.groupby("dataset"):
            for out, plot_dfii in plot_df.groupby("out"):
                plot_dfii = plot_dfi.drop(columns=['model', 'dataset', 'out', 'inp', 'trial', 'accuracy'])
                # TODO change the cmap such that each category of attributes are grouped together in same color
                ax = plot_dfii.plot(kind='barh', stacked=True, figsize=(12,5), cmap='tab20b')
                ax.set_title(f"Dataset: {os.path.basename(dataset)}\nPredicted label = {out}")
                plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), 
                        ncol=2, prop={'size': 7})  # Set legend size to 6
                plt.tight_layout()
                plt.show()

    # (2) plot the average |SHAP| across all trials for only the top N features
    n_datasets = len(plot_df['dataset'].unique())
    n_outs = len(plot_df['out'].unique())
    fig, axes, = plt.subplots(n_outs, n_datasets, figsize=(3+3*n_datasets, 2+3*n_outs),
                                sharex=True, sharey='row')
    if n_datasets==1: axes = [[ax] for ax in axes]

    
    fig.suptitle(f"avg. |SHAP| proportions of the top {top_n} performers across all trials")
    fig.supylabel("Predicted target")
    fig.supxlabel("Dataset iters")
    # drop the inputs with superset
    plot_df = plot_df[~plot_df['inp'].str.contains('superset')]

    for col_i, (dataset, plot_dfi) in enumerate(plot_df.groupby("dataset")):
        for row_i, (inp_out, plot_dfii) in enumerate(plot_dfi.groupby("model")):
            plot_dfii_feas = plot_dfii.drop(columns=['model', 'dataset', 
                                                     'out', 'inp', 
                                                     'trial', 'accuracy']).mean(
                                                     ).sort_values(ascending=False)
            plot_dfii_feas = plot_dfii_feas[:top_n] if top_n>0 else plot_dfii_feas
            top_performers = list(plot_dfii_feas.index)
            
            ax = axes[row_i][col_i]
            # display(plot_dfii[top_performers])
            plot_dfii[top_performers] = plot_dfii[top_performers].apply(np.abs)
            ax = sns.barplot(data=plot_dfii[['accuracy']+top_performers],
                            order=['accuracy']+sorted(top_performers),
                            color='tab:blue',
                            orient='h', ax=ax)
            if row_i==0: ax.set_title(os.path.basename(dataset))
            if col_i==0: ax.set_ylabel(plot_dfii['out'].iloc[0])
            ax.set_xlim(0,1)
            for i in ax.containers:
                ax.bar_label(i,fmt='%.2f', label_type='edge', padding=10)

    plt.tight_layout()
    plt.show()
    
    return plot_df



def viz_contribs_univariate(df):
    df_plt = df.reset_index()
    # rename input to X and output to y & create a new column X=>y for the plot 
    df_plt = df_plt.rename(columns={'inp': 'Input $X$', 'out': 'Output'})
    # replace attr_ with '$L_{}$', cov_ with '$C_{}$', lbl_ with '$y_{}$' in input & outputs 
    df_plt['Input $X$'] = df_plt['Input $X$'].str.replace('_', '-').str.replace('attr-', '$L_{').str.replace('lbl-', '$y_{').str.replace('cov-', '$C_{').add('}$') 
    df_plt['Output'] = df_plt['Output'].str.replace('_', '-').str.replace('lbl-', '$y_{').str.replace('cov-', '$C_{').add('}$')   
    df_plt['$f: X \\Rightarrow y$'] = df_plt.apply(lambda row: f"{row['Input $X$']} $\\Rightarrow$ {row['Output']}", axis=1)
    # create a column with info about dataset and model
    df_plt['Dataset & Model'] = df_plt.apply(lambda row:  f"{os.path.basename(row['dataset']).replace('toybrains_', '')} Model({row['model']})", axis=1)

    sns.set(style="darkgrid")
    g = sns.catplot(data=df_plt, kind='bar', 
                    x='score_test_balanced-accuracy', 
                    y='$f: X \\Rightarrow y$', 
                    hue='Input $X$', col='Dataset & Model', 
                    errorbar='ci',
                    aspect=1.5, height=3+0.1*len(df_plt['$f: X \\Rightarrow y$'].unique()))

    for ax in g.axes.flat:
        ax.set_xlabel('Balanced Accuracy')
        ax.set_xlim(0.0, 1)
        ax.axvline(0.5, color='grey', lw=0.5, linestyle='--') 

    return g


#############################################   Backend   #############################################
def viz_rep_metrics(metrics_scores, title=''):
    n_cols = 3
    n_rows = int(np.ceil(len(metrics_scores)/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(2*n_cols, 5*n_rows),
                                sharex=False, sharey=True)
    if title: plt.suptitle(title)
    axes = axes.flatten()

    metrics_scores = list(metrics_scores.items())
    for i, ax in enumerate(axes):
        if i<len(metrics_scores):
            metric_name, metric_dict = metrics_scores[i]
            ax.set_title(f"Metric = {metric_name}")
            sns.barplot(data=metric_dict, orient='h', ax=ax)
            if i >= (n_rows-1)*n_cols: ax.set_xlabel('Metric score')
            if i%n_cols == 0: ax.set_ylabel('Attributes')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def load_results_from_logdir(logsdir, 
                             best_ckpt_by='loss', 
                             best_ckpt_metric_should_be='min'):
    '''Load all DeepRepViz log/checkpoints present in the directory'''
    deeprepvizlog = {}
    assert os.path.isdir(logsdir)

    # (1) go through all checkpoints and find the best checkpoint
    best_ckpt_idx = -1
    best_ckpt_score = np.inf if best_ckpt_metric_should_be=='min' else 0
    for i, ckpt_dir in enumerate(sorted(glob(logsdir+'/*/*/'))):
        ckpt_name = os.path.basename(os.path.normpath(ckpt_dir))
        # (1) read the metadata.tsv columns
        metadata = pd.read_csv(ckpt_dir+'metadata.tsv', sep='\t')
        print(metadata.columns)
        # if "IDs" not in deeprepvizlog:
        #     deeprepvizlog["IDs"]    = metadata["IDs"].values
        #     deeprepvizlog["labels"] = metadata["labels"].values
        ckpt_values.update({col:metadata[col].values for col in metadata if col not in ["IDs","labels"]})
        # (3) load metrics
        with open(ckpt_dir+'metametadata.json') as fp:
            metametadata = json.load(fp)
            ckpt_values.update(metametadata)
            score = metametadata['metrics'][best_ckpt_by]
            if (best_ckpt_metric_should_be=='min' and score<best_ckpt_score) or \
                    (best_ckpt_metric_should_be=='max' and score>best_ckpt_score):
                best_ckpt_score = score
                best_ckpt_idx = i 
                
    
    deeprepvizlog['best_ckpt_idx']=best_ckpt_idx
    deeprepvizlog['checkpoints']=checkpoints

    # if shortname=='': 
    #     logsdir_folder = dirname(dirname(normpath(logsdir)))
    #     shortname = logsdir_folder if 'deeprepvizlog' not in logsdir_folder else logsdir
    deeprepvizlogs.update({logsdir: deeprepvizlog})

