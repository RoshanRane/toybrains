import os,sys
from glob import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap 
import seaborn as sns
import re
from math import log

UTILS_PATH = os.path.abspath('../')
if UTILS_PATH not in sys.path: sys.path.append(UTILS_PATH)
from utils.genutils_all import break_dataset_name


####################################################################################################

def _get_rename_cols_dict(metric, true_names, con_names):
    rename_cols = {f"score_holdout_test-all_{metric.replace('_','-')}": 'A_total', 
                   f"score_holdout_test-none_{metric.replace('_','-')}": 'A_none'}
    # rename true_attrs to A_true
    rename_cols.update({f"score_holdout_test-{true.replace('_','-')}_{metric.replace('_','-')}": f'A_true{i+1}' for i, true in enumerate(true_names)})
    # rename confounders to con_1(dtype), .. con_2(dtype)
    rename_cols.update({f"score_holdout_test-{con.replace('_','-')}_{metric.replace('_','-')}": f'A_con{i+1}({con.split('_')[-1]})' for i, con in enumerate(con_names)})
    
    return rename_cols

####################################################################################################

def show_scores_barplot(df, metric='balanced_accuracy', effect_mul=1):
    n_models = len(df.index.get_level_values('model').unique())
    n_ood_tests = len(df.columns)
    n_dataset_variations = df.index.get_level_values('dataset').nunique()//(n_models)
    f, axes = plt.subplots(1, n_models, figsize=(2+5*n_models, 1+n_ood_tests+n_dataset_variations//4), 
                            sharex=True, sharey=True)

    if n_models==1: axes = [axes]
    
    for ax, (model, df_result) in zip(axes, df.reset_index('model').groupby('model')):
        
        df_result = df_result.droplevel(['out', 'inp', 'type']).T.sort_index().drop('model')

        # change the dataset name to only have the effect sizes
        eff_sizes = [] 
        cXs, cys, Xys = [], [], []
        for d in df_result.columns:
            _,_, cX, cy, Xy, _ = break_dataset_name(d, effect_mul=effect_mul)            
            eff_sizes.append(f"cX={int(cX):3d} cy={int(cy):3d} Xy={int(Xy):3d}")
            cXs.append(cX)
            cys.append(cy)
            Xys.append(Xy)

        df_result.columns = eff_sizes
        # resort the columns by the strength of cy, then Xy and finally cX
        df_result = df_result.reindex(columns=np.array(eff_sizes)[np.lexsort((cXs, cys, Xys))])

        # set one cmap cycle for each unique value of cy and Xy
        col_to_color = []
        color_idx = -1
        # col_to_edgecolor = []
        # edge_idx = -1
        cy_last, Xy_last = -1, -1
        for d in df_result.columns:
            cX, cy, Xy = [float(x) for x in re.findall(r"\d+", d)]
            if Xy!=Xy_last:
                color_idx += 1
            # if cy!=cy_last:
            #     edge_idx += 1
            cy_last, Xy_last = cy, Xy

            cmap = get_cmap('tab10')
            cmap_edgecolor = get_cmap('Dark2')
            col_to_color.append(cmap(color_idx, alpha=0.1+0.9*cX*cy/5000)) # hardcoded for cX in [0,100]
            # col_to_edgecolor.append(cmap_edgecolor(edge_idx)) 
        # center
        if 'accuracy' in metric.lower(): 
            # scale to 0 to 100 % and center to the chance level (50%) 
            if df_result.max().max() <= 1.0: df_result = df_result * 100
            df_result = df_result - 50
            xmin = -10
            xmax = 50
        elif 'r2' in metric.lower():
            # scale to 0 to 100 %
            if df_result.max().max() <= 1.0: df_result = df_result * 100
            df_result = df_result - 0
            xmin = -10
            xmax = 100
        else:
            xmin = None
            xmax = None
            

        # remove the metric name from the column names
        df_result.index = df_result.index.str.replace(f"_{metric}", '')

        df_result.plot(kind='barh', ax=ax, color=col_to_color, edgecolor='grey', linewidth=1)

        # hdls, lbls = ax.get_legend_handles_labels()
        ax.set_xlim(xmin, xmax)
        ax.set_title(f"{metric} scores on the test datasets (OOD)\nModel: {model}")
        # remove legend in all but the last plot
        if ax != axes[-1]: ax.get_legend().remove()
        else: ax.legend(title='Configured effect sizes (%)', bbox_to_anchor=(1,1), frameon=True)

    plt.tight_layout()
    plt.show()

####################################################################################################
def show_scores_lineplot(df, 
                        con_names, true_names,
                        metric='balanced_accuracy', 
                        ax=None, 
                        title='', adjust_xticks=True,
                        subtract_A_total=True, show_A_sum=False, show_A_none=True, 
                        force_neg_to_zero=False):

    fs = 12 
    # sort the datasets such that they are ordered first by the changes in cy, then cX, and then Xy 
    sort_vals = []
    proba_to_eff = lambda p: log(p/(1-p))
    Xy_betas = []
    cy_gammas = []
    for d in df.index:
        cX, cy, Xy = break_dataset_name(d)[2:-1]
        sort_val = Xy + cy/100 + cX/10000
        sort_vals.append(sort_val)
        Xy_betas.append(proba_to_eff(Xy/100))
        cy_gammas.append(proba_to_eff(cy/100))
        
    # save a column with the true beta / gamma values
    df.loc[:,'true_beta'] = Xy_betas
    df.loc[:,'true_gamma'] = cy_gammas

    df = df.iloc[np.array(sort_vals).argsort()]

    # rename the columns to [A_total, A_true, A_confi] 
    rename_cols_dict = _get_rename_cols_dict(metric, true_names, con_names)
    df = df.rename(columns=rename_cols_dict)

    # subtract A_total from A_true and A_confi 
    if subtract_A_total:
        for col in df.filter(like=f'A_').columns:
            if col != 'A_total':
                df[col] = (df['A_total'] - df[col]).abs()
                if 'accuracy' in metric:
                    df[col] = df[col] + 50
                if 'logodds_mse' in metric:
                    if 'A_true' in col :
                        df[col] = df[col]/(df['A_total']) #df['true_beta']
                    elif 'A_con' in col:
                        df[col] = df[col]/(df['A_total']) #df['true_gamma'] 
    
    # df_disp = df.filter(regex='^(A_true.*|A_con.*|true_beta|true_gamma)$').replace([np.inf, -np.inf], 0)
    # display(df_disp.style.format("{:.2f}").bar())

    # force negative values to zero
    if force_neg_to_zero:
        for col in df.filter(like=f'A_').columns:
            df[col] = df[col].apply(lambda x: x if x>0 else 0)
            
    if show_A_sum:
        df['A_trues + A_cons'] = df[[f'A_true{i+1}' for i in range(len(true_names))] + [f'A_con{i+1}({con.split("_")[-1]})' for i, con in enumerate(con_names)]].sum(axis=1)
        # for accuracy remove 50% from each A to only show the above chance levels
        if 'accuracy' in metric:
            df['A_trues + A_cons'] = df['A_trues + A_cons'] - 50*(len(true_names) + len(con_names) - 1)
    
    # if there are multiple models, plot separateluy for each model
    for bl_model, dfi in df.groupby('model'):

        # create and plot a seaborn lineplot
        sns.set_style('ticks')
        f, ax = plt.subplots(figsize=(25, 8))

        # set A_total to blue, A_true to green, A_confi to shades of red and A_none to grey
        hue_order = []
        colors = []
        if show_A_none: 
            hue_order += ['A_none']
            colors += ['grey']
        if show_A_sum:
            hue_order += ['A_trues + A_cons']
            colors += ['goldenrod']


        hue_order += ['A_total'] + [f'A_true{i+1}' for i in range(len(true_names))] + [f'A_con{i+1}({con.split("_")[-1]})' for i, con in enumerate(con_names)]
        colors += ['blue'] + sns.color_palette('Greens', len(true_names)) + sns.color_palette('Reds', len(con_names)) 

        g = sns.lineplot(df, ax=ax, 
                        dashes=False, markers=True, alpha=0.8, linewidth=2.5,
                        hue_order=hue_order,
                        palette=colors)

        # make the plot pretty and readable
        ax.set_ylabel(f"{metric.replace('-', ' ').replace('_',' ').title()}", fontsize=fs)
        ax.set_xlabel(r'Increasing confound signal [$X \leftarrow c \to y$]'+'\n'+r'   &   True signal  [$X \rightarrow y$] ', fontsize=fs)
        ymin, ymax = ax.get_ylim()
        y_lines = [y_line for y_line in [0,25,75,100] if ymin<=y_line<=ymax]
        for y_line in y_lines:
            ax.axhline(y_line, color='grey', ls='--', lw=0.8, alpha=0.5)

        # the xticks are the dataset names - draw lines when the true effect Xy changes
        if adjust_xticks:

            cX_cy_Xy_tuples  = np.array([break_dataset_name(xtick.get_text())[2:-1] for xtick in ax.get_xticklabels()]).astype(int)
            poses = list(ax.get_xticks())
            new_xticklabels = []
            majorticks = []

            last_Xy = -1
            for i, pos_i in enumerate(poses):
                cX_i, cy_i, Xy_i = cX_cy_Xy_tuples[i]
                # add a major tick label every time the total_Xy increases
                if Xy_i != last_Xy:
                    majorticks.append(pos_i)
                    last_Xy = Xy_i
                    new_xtick = f'Xy={Xy_i:03d}%      cy={cy_i:03d}%   cX={cX_i:03d}%'
                else:
                    new_xtick = f'cy={cy_i:03d}%   cX={cX_i:03d}%'
                    # new_xtick = f'Xy={Xy_i:03d}%      cy={cy_i:03d}%   cX={cX_i:03d}%'
                new_xticklabels.append(new_xtick)
            majorticks.append(poses[-1]+2)

            # print(ax.get_xticklabels(), new_xticklabels)       
            ax.set_xticks(poses, new_xticklabels, rotation=90)

            # vertical lines to show transition of X<-c->y
            for x_line in majorticks:
                ax.axvline(x_line-0.5, color='grey', ls='--', lw=1, alpha=0.5)
                ax.vlines( x_line-0.5, 0, -0.45, color='grey', ls='--', lw=1,
                        clip_on=False,
                        transform=ax.get_xaxis_transform())
        
            ax.set_xlim(-1, poses[-1]+1)

        title_text = f"Metric = {metric.replace('_',' ').title()}   Model = {bl_model}"
        if title: title_text += f"\n{title}"
        ax.set_title(title_text, fontsize=fs+4)

        sns.move_legend(ax, "upper right", bbox_to_anchor=(1.0, 1.15), frameon=True, ncol=3, title='')
        plt.setp(g.get_legend().get_texts(), fontsize=str(fs+4))  # for legend text


####################################################################################################
    
def show_scores_decomp(results, 
                    true_names, 
                    metric='D2', 
                    center_metric=False):

    '''plot the results separately for each dataset group (unique cons) for different variations of cX, cy, Xy'''
    sns.set_style("whitegrid")
    df = results.copy()

    # beautify the dataset names
    df['dataset'] = df['dataset'].apply(lambda x: os.path.basename(x.replace('/train/', '')))

    df[['ncons', 'cons', 'cX', 'cy', 'Xy', 'dataset_suffix']] = df['dataset'].apply(break_dataset_name).apply(pd.Series)
    dataset_prefix = df['dataset'].iloc[0].split('cX')[0]
    df['dataset'] = df['dataset'].str.replace(dataset_prefix, '')
    # replace dataset names with just cX, cy, Xy values

    ncons = df['ncons'].unique()[0]
    con_names = np.unique(df['cons'].values)[0]
    assert len(con_names) == 1, f"Function accepts one dataset group (unique cons) at a time. The datasets passed have con names = {cons}"
    # clean the column names and only select one metric
    rename_cols_dict = _get_rename_cols_dict(metric, true_names, con_names)
    df = df.rename(columns=rename_cols_dict)
    df = df[['dataset', 'model', 'cX', 'cy', 'Xy', *rename_cols_dict.values()]]

    # determine if the metric is in [0,1.0] range or scaled to [0,100]
    if 'accuracy' in metric:
        min_val = 0
        max_val = 50 if df['A_total'].max()>1.0 else 0.5 
        # subtract 50% from all accuracy cols
        if center_metric:
            print(f"Centering the metric {metric} by subtracting {max_val}")
            for col in df.filter(like='A_').columns:
                df[col] = df[col] - max_val

    elif 'r2' in metric.lower() or 'd2' in metric.lower():
        min_val = 0
        max_val = 100 if df['A_total'].max()>1.0 else 1.0 
        print(f"Centering the metric {metric} by forcing negative values to zero")
        for col in df.filter(like='A_').columns:
            df[col] = df[col].apply(lambda x: x if x>min_val else min_val)

    n_dataset_vers = df['dataset'].nunique()
    n_models = df['model'].nunique()

    f, axes = plt.subplots(n_dataset_vers, n_models, 
                        figsize=(2+4*n_models, 1+n_dataset_vers//4), sharex=True, sharey='row')
    if n_dataset_vers==1: axes = [[axes]]
    if n_models==1: axes = axes.reshape(-1,1)

    # sort the dataset names by the strength of Xy, then cy, and finally cX
    df = df.sort_values(by=['Xy', 'cy', 'cX'])
    for i, (d, dfi) in enumerate(df.groupby('dataset', sort=False)):
        ax_row = axes[i] if n_dataset_vers>1 else axes
        dfi = dfi.set_index('dataset')

        for j, (model, dfij) in enumerate(dfi.groupby('model')):
            first_subplot = True if (i==0 and j==0) else False
            last_subplot = True if (i==n_dataset_vers-1 and j==n_models-1) else False

            ax = ax_row[j]
            # print(f"[{i},{j}] Plotting cons={cons} dataset={d} model={model}")
            # display(dfij)
            dfij = dfij.drop(columns=['model'])

            # 1) First plot a horizontal bar of the total ('all') performance in this dataset
            dfij_all = pd.Series({'A_total': dfij['A_total'].mean(), 
                                  'std': dfij['A_total'].std()*2}).to_frame(name=d).T # show 2x std dev as the error bar
            dfij_all.plot.barh(ax=ax, color='slategrey', alpha=1, xerr='std', 
                            position=0.0, width=0.4, legend=first_subplot)

            # 2) Then plot the decomposed performance of each covariate on each dataset
            # take an average over all the trials
            dfij_decomp = dfij.drop(columns=['A_total', 'A_none']).mean(axis=0).to_frame(name=d).T.dropna(axis=1)
            # group all the remaining covariates into 'A_cons', 'cov', 'others'
            # TODO 'others' for any other covs  
            # dfij_decomp['others'] = dfij_decomp.drop([for c in con_names+true_names], axis=1).sum() 
            # assert dfij_decomp['others'] is not pd.NA, f"Error in calculating the 'others' column. Found {dfij_decomp['others']}"
            dfij_decomp = dfij_decomp.filter(regex='(A_true.*|A_con.*)')
            dfij_decomp.plot.barh(stacked=True, ax=ax,
                                position=1.0, alpha=0.9, width=0.6, legend=first_subplot)

            if first_subplot: ax.set_title(dataset_prefix.strip('_'))
            
            ax.set_xlim(min_val, max_val)
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.spines.bottom.set_visible(False)

            # add legend to the topmost axis
            if first_subplot:
                leg = ax.legend(title=f'{metric} decomposed into', 
                        loc='best', 
                        ncols=4, frameon=True)
                leg.remove()
            if last_subplot:
                ax.add_artist(leg)


####################################################################################################

def plot_py_cond_dist(nc, set_name):

    fig, axes = plt.subplots(2,2, figsize=(8,6), sharex=True, sharey=True)
    axes = axes.ravel()

    data = {'cX':np.empty(0, dtype=int), 'cy':np.empty(0, dtype=int), 'Xy':np.empty(0, dtype=int), 
            'probas':np.empty(0), 
            'con_name':np.empty(0)}

    datasets = sorted(glob(f'dataset/toybrains_*{set_name}*_cX*_cy*_Xy*/train/*.csv'))

    for dataset in datasets:
        dataset_name = os.path.basename(dataset).replace('.csv','')
        ncons, cons, cX, cy, Xy, _ = break_dataset_name(dataset_name)
        # print('[D]', dataset_name, cons, cX, cy, Xy)

        try:
            df = pd.read_csv(dataset).set_index('subjectID')
            # plot the distribution of the label lbl_y
            # plt.hist(df['lbl_y'], bins=20, alpha=0.5, label='lbl_y')
            # plot the distribution of the probas
            probas = np.array([eval(i) for i in df['probas_lbl_y'].values])
            n_obs = probas.shape[0]

            data['con_name'] = np.append(data['con_name'], np.full(n_obs, cons[0]))
            data['cX'] = np.append(data['cX'], np.full(n_obs, cX).astype(int))
            data['cy'] = np.append(data['cy'], np.full(n_obs, cy).astype(int))
            data['Xy'] = np.append(data['Xy'], np.full(n_obs, Xy).astype(int))
            data['probas'] = np.append(data['probas'], probas[:,1])
        except Exception as e:
            print(f"[ERROR] reading {dataset}. Skipping .... \n{e}")
            continue
        
    df = pd.DataFrame(data)
    hue_max = df['Xy'].max() 
    legend_kwargs = dict(loc="upper center", bbox_to_anchor=(0.5, 1.25), 
                        ncols=4, fontsize=6, frameon=True)
    fig.suptitle(f"con = {cons}           n(covs --> lbl_y) = {nc}")
    g = sns.kdeplot(df, x='probas', hue='Xy', 
                    cut=1, fill=False, common_norm=False, hue_norm=(0,hue_max), 
                    ax=axes[0])
    # move the legend of each axis to the top to look like the title of the plot
    sns.move_legend(g, **legend_kwargs)
    g = sns.kdeplot(df, x='probas', hue='cy', 
                    cut=1, fill=False, common_norm=False, hue_norm=(0,hue_max), 
                    ax=axes[1], palette='flare')  
    sns.move_legend(g, **legend_kwargs)
    g = sns.kdeplot(df, x='probas', hue='cX', 
                    cut=1, fill=False, common_norm=False, hue_norm=(0,hue_max), 
                    ax=axes[2], palette='mako_r')
    sns.move_legend(g, **legend_kwargs)
    # unique tuple of [Xy effects, Xcy effect]
    df['Xy & cy'] = df.apply(lambda x: str(sorted([x['Xy'], x['cy']])), axis=1)
    unique_combos = df['Xy & cy'].unique()
    order = np.argsort([np.multiply(*eval(i)) for i in unique_combos])

    g = sns.kdeplot(df, x='probas', hue='Xy & cy',
                    cut=1, fill=False, common_norm=False, 
                    hue_order=unique_combos[order],
                    ax=axes[3], palette='viridis_r')
    sns.move_legend(g, **legend_kwargs)

    plt.tight_layout()
    plt.show()