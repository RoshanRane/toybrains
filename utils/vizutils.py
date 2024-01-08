# standard python packages
import os, sys
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.image as mpimg
from sklearn import datasets, linear_model
from tqdm.notebook import tqdm
import random
import math
import json
import matplotlib
from  matplotlib.ticker import FuncFormatter
from matplotlib.colors import is_color_like
from colordict import ColorDict, rgb_to_hex


def show_images(img_files, n_rows=1):
    assert isinstance(img_files[0], (str,np.ndarray, np.generic)), "img_files \
should either me a string path to the image files or numpy arrays"   
    n_cols = 10
    f, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols,n_rows), 
                           sharex=True, sharey=True)
    # f.suptitle("Toy brains dataset:")
    axes = axes.ravel() if not isinstance(axes, matplotlib.axes.Axes) else [axes]

    for i, img in enumerate(img_files):
        if i<len(axes):
            if isinstance(img, str):
                img = mpimg.imread(img)
            ax = axes[i]
            ax.imshow(img)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_col_counts(df, title=''):
    
    df_copy = df.reindex(sorted(df.columns), axis=1).copy()
    cols = df_copy.columns 
    # convert each column to its appropriate dtype and 
    # then decide the type of plot to use for it
    plottypes = {}
    
    for col in cols:
        if df_copy[col].nunique()==2:
            plottypes.update({col:'pie'})
        elif df_copy[col].dtype.name == 'object' or isinstance(df_copy[col].iloc[0], str):
            plottypes.update({col:'bar'})
        elif 'float' in df_copy[col].dtype.name:
            # round the floats to 2 decimal
            df_copy[col] = df[col].round(1)
            # use hist plot if there are more than 10 states  
            if df_copy[col].nunique()<10:
                plottypes.update({col:'bar'})
            else:
                plottypes.update({col:'hist'})
            # if all values are int then covert col to int dtype
            if df_copy[col].dropna().apply(float.is_integer).all():
                df_copy[col] = df_copy[col].astype(int)
            
    # re-sort dataframe cols by dtypes
    cols = df_copy.dtypes.sort_values().index 
    # decide the number of rows and columns in the plot
    subplot_ncols = 5 if len(cols)>=5 else len(cols)
    subplot_nrows = len(cols)//subplot_ncols
    subplot_overflows = len(cols)%subplot_ncols
    if subplot_overflows!=0: subplot_nrows+=1
    # create subplots set attributes
    f,axes = plt.subplots(subplot_nrows, subplot_ncols, 
                          figsize=(4*subplot_ncols,3*subplot_nrows))
    axes = axes.ravel() if not isinstance(axes, matplotlib.axes.Axes) else [axes]
    if title: f.suptitle(title, fontsize=16)
    f.supylabel("Count")

    for i, ax in enumerate(axes):
        # print('[D]',col, plottypes[col])
        if i >= subplot_ncols*(len(cols)//subplot_ncols) + subplot_overflows:
            ax.axis('off')
            continue
            
        col = cols[i]
        plottype = plottypes[col]
            
        if plottype == 'bar':
            # check if the attribute represents colors then use the same color names for the bar plot
            if isinstance(df_copy[col].iloc[0], str) and is_color_like(df_copy[col].iloc[0].split('-')[-1]):
                colormap = ColorDict()
                colors = [rgb_to_hex(colormap[c.split('-')[-1]]) for c in df_copy[col].sort_values().unique().tolist()]

                sns.countplot(data=df_copy, x=col, ax=ax, 
                             order=df_copy[col].sort_values().unique())
            else:
                sns.countplot(data=df_copy, x=col, ax=ax)
            # format the xtick labels 
            if 'int' in df_copy[col].dtype.name:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
            elif isinstance(df_copy[col].iloc[0], str):
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                
        elif plottype == 'hist':
            bins = df_copy[col].nunique()//5
            sns.histplot(data=df_copy, x=col, ax=ax, kde=True, bins=bins) #multiple='fill'
        elif plottype == 'pie':
            cnt = df_copy[col].value_counts().sort_index()
            ax.pie(cnt, labels=cnt.index,
                    colors=sns.color_palette('pastel'), autopct='%.0f%%')
            
        ax.set_title(col)
        ax.set_xlabel(None)

    plt.tight_layout()
    
    
def plot_col_dists(df, attr_cols, cov_cols, title=''):
    
    df = df.copy()
    cov_cols = sorted(cov_cols)
    attr_cols = sorted(attr_cols)
    subplot_nrows = len(cov_cols)
    subplot_ncols = len(attr_cols)
    fs=12
    # convert all columns to numerical
    for col in attr_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            vals =  {v:i for i,v in enumerate(sorted(df[col].unique()))}
            df[col] = df[col].map(vals)
            
    # if the covariates are continuous then bin them
    for col in cov_cols:
        if df[col].nunique()>5:
            df[col] = pd.cut(df[col], bins=3, precision=0)
    
    # create subplots set attributes
    f,axes = plt.subplots(subplot_nrows, subplot_ncols, 
                          figsize=(2+1.5*subplot_ncols, 1.5*subplot_nrows),
                          sharex="col", sharey="col")
    if title: f.suptitle(title, fontsize=fs+4)
    
    for i, axis_row in enumerate(axes):
        for j, ax in enumerate(axis_row):
            cov, attr = cov_cols[i], attr_cols[j]
            g = sns.kdeplot(df, x=attr, hue=cov, ax=ax, fill=True, legend=(j==0))
            if j==0: 
                # make 2 cols if there are many lagend labels
                ncol=2 if len(ax.legend_.legendHandles)>3 else 1
                sns.move_legend(g, loc="upper left", 
                                bbox_to_anchor=(-1.5,1.), ncol=ncol,
                                frameon=True, 
                                title_fontproperties={'size':fs, 
                                                      'weight':'heavy'})
            # set xlabel and ylabel at the top and leftside of the plots resp.
            ax.set_ylabel(None)
            ax.set_xlabel(attr.replace('gen_',''), fontsize=fs) if i==len(axes)-1 else ax.set_xlabel(None)
            # turn off the density ticks
            ax.set_yticklabels([])
            if i==0: ax.set_title(attr.replace('gen_',''), fontsize=fs)
            
    f.supylabel("Covariates & labels", fontsize=fs+2)
    
    plt.tight_layout()



#### BASELINE modelling with gen. attributes ###

def viz_baseline_results(run_results):
    ''' vizualization output of baseline models
    '''
    if not isinstance(run_results, list): run_results = [run_results]
    dfs = []
    for run in run_results:
        if isinstance(run, pd.DataFrame):
            dfs.append(run.copy())
        elif isinstance(run, str) and os.path.exists(run):
            dfs.append(pd.read_csv(run))
        else:
            raise ValueError(f"{run} is neither a path to the results csv nor a pandas dataframe")
    viz_df = pd.concat(dfs, ignore_index=True)
    
    x="$R2$ / Pseudo-$R2$ (%) on test data"
    y="Predicted Output"
    row="Output type"
    col="Dataset"
    hue="Input features"
    viz_df = viz_df.rename(columns={'test_metric':x, 'out':y, 
                                    'inp':hue, 'dataset':col})
    # remove the full path from the dataset names to prevent long titles
    viz_df[col] = viz_df[col].apply(lambda x:x.split('/')[-1]) 
    # separate covariates and labels
    map_out_type = {'cov':'Covariates','lbl':'Label'}
    viz_df[row] = viz_df[y].apply(lambda x: x.split('_')[0])
    # adjust the hue for the legend
    viz_df[hue] = viz_df[hue].apply(lambda x: x.replace(', ',',\n'))
    unique_inps = viz_df[hue].sort_values().unique()
    unique_cols = viz_df[col].sort_values().unique()
    unique_y    = viz_df[y].sort_values().unique()
    # use a different color set for attrs and lbls and conf in inputs
    palette_attr = sns.color_palette("mako", len(unique_inps))
    palette_covs = sns.color_palette("husl", len(unique_inps))
    palette = {}
    m,n=0,0
    for cat in unique_inps:
        if cat[:5]=='attr_':
            palette.update({cat:palette_attr[m]})
            m+=1
        else:
            palette.update({cat:palette_covs[n]})
            n+=1
    
    height = 1+len(unique_y)//2
    g = sns.catplot(data=viz_df, kind='bar',
                    col=col, row=row,
                    # row_order=['lbl','cov'], # show labels before covariates
                    x=x, y=y, hue=hue, 
                    palette=palette, #hue_order=hue_order, 
                    errorbar=('ci', 95), 
                    legend=True, legend_out=True,
                    sharey='row', height=height, aspect=2.0)
    for ax in g.axes.ravel():
        for i in ax.containers:
            ax.bar_label(i, fmt='%.2f', label_type='edge', padding=10)
    
    # adjust subplot titles
    g.set_titles("{col_var}: {col_name}") 
    for i, ax in enumerate(g.axes.ravel()):
        if i==0:
            title = ax.get_title()
        elif i<len(unique_cols): 
            ax.set_title(ax.get_title().replace(title,'.. '))
        else: 
            ax.set_title('')
    # adjust legend
    g._legend.set_frame_on(True)
    # set custom x-axis tick positions and labels
    g.set(xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            xticklabels = ['0%', '20%', '40%', '60%', '80%', '100%']) 
    plt.show()
# plt.savefig("figures/results_bl.pdf", bbox_inches='tight')

def viz_baseline_results_summary(dfs):
    dfs = pd.concat(dfs)
    dfs['Dataset iters'] = dfs['dataset'].apply(lambda x: int(x.split('_')[-1][-1]))
    dfs['inp'] = dfs['inp'].apply(lambda x: x.replace(', ',',\n'))
    n_cols = len(dfs['out'].unique())
    fig, axes = plt.subplots(len(dfs['out'].unique()),1,  
                            figsize=(5, 2*n_cols),
                            sharex=True, sharey=True)
    axes = axes.flatten()
    fig.supylabel("Predicted target")
    # fig.supxlabel("Dataset iters")
    for i, out in enumerate(dfs['out'].unique()):
        ax = axes[i]
        df = dfs[(dfs['out']==out)]
        ax.set_ylabel(out)
        sns.lineplot(x='Dataset iters', y='test_metric', hue="inp", 
                     data=df, ax=ax)
        if i==0:
            ax.legend(title=r'Accuracy ($R^2$ or $D^2$)', 
                       loc="upper left", bbox_to_anchor=(1, 1))
        else:
            ax.get_legend().remove()    
    plt.tight_layout()
    plt.show()


def viz_baseline_results_shap(df_results,  
                              top_n=4,  
                              show_individual_trials=False):
    if isinstance(df_results, (list, tuple)):
        df_results = pd.concat(df_results)
    
    # first, process all SHAP values and collect them in a dataframe
    plot_df = []
    for i, row in df_results.iterrows():
        shap_vals = row["shap_values"]
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
    fig, axes, = plt.subplots(n_outs, n_datasets, figsize=(3+3*n_datasets, 1+3*n_outs),
                                sharex=True, sharey='row')
    fig.suptitle(f"avg. |SHAP| proportions of the top {top_n} performers across all trials")
    fig.supylabel("Predicted target")
    fig.supxlabel("Dataset iters")
    # drop the inputs with superset
    plot_df = plot_df[~plot_df['inp'].str.contains('superset')]

    for col_i, (dataset, plot_dfi) in enumerate(plot_df.groupby("dataset")):
        for row_i, (inp_out, plot_dfii) in enumerate(plot_dfi.groupby("model")):
            top_performers = list(plot_dfii.drop(
                columns=['model', 'dataset', 'out', 'inp', 'trial', 'accuracy']).mean().sort_values(
                    ascending=False)[:top_n].apply(
                        lambda x: '{:.0f}%'.format(x*100)).index)
            
            ax = axes[row_i][col_i]
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