import os, sys
from glob import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pprint


PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR) 
from create_toybrains import ToyBrainsData

def apply_tweak_rules(rules, tweak_rules):
    applied_rules = False
    for rule in tweak_rules:
        cov1, state1, cov2, key, change_fn, limit = rule
        val = rules[cov1][state1][cov2][key]
        if val < limit:
            rules[cov1][state1][cov2][key] = change_fn(val)
            applied_rules = True
    return rules, applied_rules


def create_config_file(config_fname, covars, rules, 
                       show_dag_probas=False, 
                       return_baseline_results=False,
                       gen_images=0):
    n_samples = 1000 if gen_images<=0 else gen_images
    with open(config_fname, 'w') as f:
        f.write('# List of all covariates\n\
COVARS           = {}\n\
# Rules about which covariate-state influences which generative variables\n\
RULES_COV_TO_GEN = {}\n'.format(pprint.pformat(covars), pprint.pformat(rules)))

    # 4) Test that the config file meets the expectation using `ToyBrainsData(config=...).show_current_config()`

    toy = ToyBrainsData(config=config_fname)
    if show_dag_probas:
        print(f"Config file: {config_fname}")
        print(toy.show_current_config())
    df = toy.generate_dataset_table(n_samples=n_samples, 
                                    outdir_suffix=f"n_{os.path.basename(config_fname).replace('.py','')}")

    
    if gen_images > 0:
        toy.generate_dataset_images(n_jobs=10)

    if return_baseline_results:
        df_results = toy.fit_baseline_models(
        input_feature_types=["attr_subsets", "cov_subsets"], 
        debug=False)
        return df_results
    
# visualization
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
                    # show labels before covariates
                    row_order=['lbl','cov'], 
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
    dfs['dataset_iter'] = dfs['dataset'].apply(lambda x: int(x.split('_')[-1][-1]))
    n_cols = len(dfs['out'].unique())
    fig, axes = plt.subplots(len(dfs['out'].unique()),1,  
                            figsize=(6, 2*n_cols),
                            sharex=True, sharey=True)
    axes = axes.flatten()
    for i, out in enumerate(dfs['out'].unique()):
        ax = axes[i]
        df = dfs[(dfs['out']==out)]
        ax.set_ylabel(f'Prediction target: {out}')
        sns.lineplot(x='dataset_iter', y='test_metric', hue="inp", 
                     data=df, ax=ax)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()


def viz_baseline_results_shap(df_results, labels=[], all_results=False):
    for out, df_results_i  in df_results.groupby("out"):
        if labels==[] or out in labels:
            plot_df = []
            for i, row in df_results_i.iterrows():
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
                    plot_df.append(plot_dfi)
                    
            plot_df = pd.DataFrame(plot_df)
            # sort by column name alphabetically
            plot_df = plot_df.reindex(sorted(plot_df.columns), axis=1)
            if all_results:
                # TODO change the cmap such that each category of attributes are grouped together in same color
                ax = plot_df.plot(kind='barh', stacked=True, figsize=(12,5), cmap='tab20b')
                plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), 
                        ncol=2, prop={'size': 7})  # Set legend size to 6
                plt.tight_layout()
                plt.show()

            # plot the average shap values across all trials
            plot_df = plot_df.reset_index(names='Model')
            plot_df[['Model', 'Accuracy']] = plot_df['Model'].str.split("\n Acc = ", expand=True)
            plot_df['Accuracy'] = plot_df['Accuracy'].str.replace("%", "").astype(float)/100
            plot_df[['Model', 'Trial']] = plot_df['Model'].str.split(" ", expand=True)
            
            fig, axes, = plt.subplots(1,2, figsize=(7,2),
                                      sharex=True, sharey=True)
            plt.suptitle(f"{df_results['dataset'][0]}: top 4 performers avg. |SHAP| proportion across trials:")
            axes = axes.flatten()
            for i, (out, plot_dfi) in enumerate(plot_df.groupby("Model")):
                top_performers = list(plot_dfi.drop(
                    columns=['Model', 'Trial', 'Accuracy']).mean().sort_values(
                        ascending=False)[:4].apply(
                            lambda x: '{:.0f}%'.format(x*100)).index)
                ax = axes[i]
                ax = sns.barplot(data=plot_dfi[['Accuracy']+top_performers],
                                 order=['Accuracy']+sorted(top_performers),
                                 orient='h', ax=ax)
                ax.set_title(out)
                ax.set_xlim(0,1)
                for i in ax.containers:
                    ax.bar_label(i,fmt='%.2f', label_type='edge', padding=10)

            plt.tight_layout()
            plt.show()



