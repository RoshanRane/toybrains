# standard python packages
import os, sys
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.image as mpimg
from tqdm.auto import tqdm

import json
import matplotlib
from  matplotlib.ticker import FuncFormatter
from matplotlib.colors import is_color_like
from colordict import ColorDict, rgb_to_hex
import shap


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



#########################################        Baseline models        ########################################
def show_contrib_table(dfs_results, 
                       avg_over_trials=True,
                       filter_rows={}, filter_cols=[], 
                       color=None):
    '''reorganize the generated baseline results and display it as a pretty table with style:
    average the results across trials after grouping by ['dataset','out', 'inp']'''
    if isinstance(dfs_results, (list, tuple)): dfs = pd.concat(dfs_results).copy()
    
    # make the dataset name shorter for pretty-ness
    dfs['dataset'] = dfs['dataset'].apply(lambda x: os.path.basename(x.rstrip('/')))

    grp_by = ['out','inp','dataset']
    if not avg_over_trials: grp_by.append('trial')

    if filter_rows:
        for col, vals in filter_rows.items():
            dfs = dfs[dfs[col].isin(vals)]

    if filter_cols:
        dfs = dfs[grp_by+filter_cols]
        
    filter_cols = dfs.filter(regex='^score_').columns.tolist() + dfs.filter(regex='^shap__').columns.tolist()
    desc = dfs[grp_by+filter_cols].groupby(grp_by).mean()

    # format to percentages
    func = lambda s: int(s*100) if pd.notnull(s) else -1
    desc = desc.map(func)
    print("All results are shown in percentage (%)")

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







#################################       lblmidr-consite       #################################
def viz_contrib_table(data, X_axes=['X->y','c->X','c->y'], 
                      metric_col='score_test_r2',
                      show_SHAP=False, err_style='bars',
                      y_label_suffix=''):
    for rel in X_axes:
        assert rel in ['X->y','c->X','c->y'], f"invalid rel {rel}"
        
   
    if isinstance(data, pd.io.formats.style.Styler):
        data = data.data
    df = data.copy().reset_index()
    # shorten the 'dataset' name for the plot labels
    df['dataset'] = df['dataset'].apply(lambda x: x.split('_')[-1])

    # get the iterations of yX, cX, and cy as separate columns
    df[['c->y','c->X','X->y']] = df['dataset'].str.split('-', expand=True)
    for col in ['X->y','c->X','c->y']:
        # print(col, df[col].values[0])
        df[col] = df[col].str[-1].astype(int)

    # rename the test_metric column to 'Model pred. contrib score'
    y=f'Model-based contrib score {y_label_suffix}'
    df = df.rename(columns={metric_col:y})
    
    col = 'inp'
    col_order = ['attr_all', 'attr_shape-midr_curv, shape-midr_vol-rad', 
                 'attr_brain-int_fill', 'cov_all'] if 'attr_all' in  df[col].unique() else df[col].unique()
    if show_SHAP:
        y = 'SHAP contrib score'
        col = 'SHAP(attr)'
        col_order = df.filter(regex='shap__').columns.tolist()
        # select only the 'inp'=attr_all rows
        df = df[df['inp']=='attr_all']
        # stack the SHAP cols into a single column 'SHAP' for compatibility with seaborn relplot
        df = df.melt(id_vars=['dataset','inp','c->y','c->X','X->y'], 
                        value_vars=col_order, 
                    var_name=col, value_name=y)
    # display(df)
    for x in X_axes:
        if x=='X->y':
            hue='c->X'
            size='c->y'
        elif x=='c->X':
            hue='X->y'
            size='c->y'
        elif x=='c->y':
            hue='X->y'
            size='c->X'
        else:
            assert False, f"invalid col {x}"

        sns.set(style="darkgrid")
        g = sns.relplot(data=df, kind='line', 
                x=x,y=y, hue=hue, style=size, size=size,
                err_style=err_style,
                # hue='inp', height=30, aspect=0.5,
                col_wrap=2, col=col, palette='brg_r',
                col_order=col_order,
                height=5, aspect=1.5,
            )
        # turn on frame of the legend
        g._legend.set_frame_on(True)
        # set y lim to 0-100
        # g.set(ylim=(0, 100))
        g.fig.suptitle(f"{y} as we iteratively increase {x}", fontsize=20)
        g.fig.subplots_adjust(top=0.9)


def viz_contrib_table_2(df_original, 
                        metric_name='r2', 
                        cmap=None, 
                        title=''):
    
    def get_yX_cX_cX(dataset_suffixes):
        cy, cX, yX = zip(*dataset_suffixes)
        # sanity check the index format
        assert (cy[0].startswith('cy')) and (cX[0].startswith('cX')) and (yX[0].startswith('yX')), "This plot function expects the  dataset names (df.index)\
to be of format '*_cyiii-cXjjj-yXkkk' where cy, cX, and yX are the c->y, c->X, and X->y relations respectively,\
and iii, jjj & kkk are the strength of this relation in percentage ranging in 0-100."
        cy = np.array([int(i[2:]) for i in cy])
        cX = np.array([int(i[2:]) for i in cX])
        yX = np.array([int(i[2:]) for i in yX])
        return cy, cX, yX
    
    df = df_original.copy()
    assert df.index.name.lower() == 'dataset', f"index of the provided df should be 'dataset' but it is {df.index.name}"
    # shorten the 'dataset' name for the plot labels to only contain its suffix with cy, cX, yX
    df.index = df.index.map(lambda x: x.split('_')[-1])
    # sort the X axis by 100*(X<-y) + 10*(X<-c * c->y)/2 + cX
    cy, cX, yX = get_yX_cX_cX(df.index.str.split('-', expand=True))
    sort_order = yX + (cy*cX)/1000 + cX/10000 + cy/1000 
    df = df.iloc[sort_order.argsort()]

    # plot with seaborn lineplot
    sns.set_style("ticks")
    f, ax = plt.subplots(figsize=(25, 8))
    g = sns.lineplot(df, ax=ax, 
                     dashes=False, markers=True, alpha=0.9, linewidth=2,
                     palette=cmap)

    # make the plot pretty and readable
    ax.set_ylabel(f"{metric_name.replace('-', ' ').replace('_',' ').title()} score", fontsize=15)
    ax.set_xlabel(r'Increasing confound signal [$X \leftarrow c \to y$]'+'\n'+r'   &   True signal  [$X \leftarrow y$] ', fontsize=15)

    
    # on the x-axis ticks show the total X<-y and the total X<-c->y
    last_Xy = -1
    cy, cX, yX  = get_yX_cX_cX([xtick.get_text().split('-') for xtick in ax.get_xticklabels()])
    poses = list(ax.get_xticks())
    new_xticklabels = []
    majorticks = []
    for cy_i, cX_i, yX_i, pos_i in zip(cy, cX, yX, poses):
        # add a major tick label every time the total_Xy changes
        if cX_i == 0 and cy_i == 0:
            majorticks.append(pos_i)
            new_xtick = f'Xy={yX_i:03d}%      cy={cy_i:03d}%   cX={cX_i:03d}%'
        else:
            new_xtick = f'cy={cy_i:03d}%   cX={cX_i:03d}%'
        new_xticklabels.append(new_xtick)
    majorticks.append(poses[-1])

    # print(ax.get_xticklabels(), new_xticklabels)       
    ax.set_xticks(poses, new_xticklabels, rotation=90)

    # vertical lines to show transition of X<-c->y
    for x_line in majorticks:
        ax.axvline(x_line-0.9, color='grey', ls='--', lw=1, alpha=0.5)
        ax.vlines( x_line-0.9, 0, -0.45, color='grey', ls='--', lw=1,
                clip_on=False,
                transform=ax.get_xaxis_transform())
    
    ax.set_xlim(-1, poses[-1]+1)
    
    ymin, ymax = ax.get_ylim()
    ysteps = (ymax-ymin)/3
    y_lines = [ymin+ysteps, ymin+2*ysteps, ymin+3*ysteps, ymax]
    for y_line in y_lines:
        ax.axhline(y_line, color='grey', ls='--', lw=0.8, alpha=0.5)

    if title:
        ax.set_title(title, fontsize=20)

    sns.move_legend(ax, "upper right", bbox_to_anchor=(1.0, 1.1), frameon=True, ncol=3, title='')
    plt.setp(g.get_legend().get_texts(), fontsize='20')  # for legend text
