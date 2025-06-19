import os, sys
from glob import glob
import re
import numpy as np
import pandas as pd
from copy import deepcopy
import itertools

import pprint
from IPython.display import display

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR) 
    
from create_toybrains import ToyBrainsData

# import itertools
# def apply_tweak_rules(rules_og, tweak_rules, i):
#     rules = deepcopy(rules_og)
#     for rule in (tweak_rules):
#         cov1, state1, cov2, key, values_list = rule
#         rules[cov1][state1][cov2][key] = values_list[i]
#     return rules

# def print_tweaked_rules(RULES, tweak_rules, iters):
#     rules = deepcopy(RULES)
#     for i in range(iters):
#         print(f"{'-'*50}\ni={i}")
#         rules = apply_tweak_rules(rules, tweak_rules, i)
#         for tweak_rule_k in tweak_rules:
#           key0, key1, key2, key3, _ = tweak_rule_k
#           print(f"\t {key0} = {key1} \t--> {key2}:\t {key3} = {[round(v,2) for v in rules[key0][key1][key2][key3]]}")


###################################################################################################

def write_config_file(config_fname, covars, rules, 
                      config_write_kwargs={}):

    opt_lists = ''
    for k,v in config_write_kwargs.items():
        opt_lists += f"{k.upper()} = {v}\n"

    with open(config_fname, 'w') as f:
        pp = pprint.PrettyPrinter(indent=2, width=200, compact=True, sort_dicts=True)
        f.write('{}# List of all covariates\n\
COVARS           = {}\n\
# Rules about which covariate-state influences which generative variables\n\
RULES_COV_TO_GEN = {}\n'.format(opt_lists,
                                pp.pformat(covars),
                                pp.pformat(rules)))


###################################################################################################
        
def _gen_toybrains_dataset(
                split_name,  
                config_fname, covars, rules, config_write_kwargs,
                n_samples, show_dag, show_probas, 
                outdir_suffix, overwrite_existing, gen_images,
                n_jobs, verbose):
    ## 1) Write the config file
    # for test datasets, create a temporary config file with the split_name name appended and later delete it
    if split_name!="train":
        config_fname = config_fname.replace(".py", f"_{split_name}.py")
    
    # write the config file if it doesn't already exist
    if os.path.exists(config_fname) and not overwrite_existing:
        print(f"Config file '{config_fname}' already exists. Not overwriting it.")
    else:
        write_config_file(config_fname, covars, rules, config_write_kwargs=config_write_kwargs)

    ## 2) instantiate the toybrains class
    toy = ToyBrainsData(config=config_fname, verbose=verbose)
    # 3) show the generative graph used for sampling
    if show_dag==True:
        print(f"Config file: {config_fname}")
        display(toy.draw_dag())
    if show_probas is not None and len(show_probas)>0:
        print(f"Config file: {config_fname}")
        display(toy.show_current_config(subset=show_probas))

    ## 3) Generate the dataset by sampling the gen. variables in a table
    # check if the dataset has already been generated
    dataset_table_file = glob(f"{toy.OUT_DIR}*{outdir_suffix}/{split_name}/*{outdir_suffix}.csv")
    if len(dataset_table_file)>0 and not overwrite_existing:
        print(f"Dataset table '{dataset_table_file[0]}' already exists. Not overwriting it.")
    else:
        if verbose>0: print(f"Generating dataset table with {n_samples} samples and storing \
at {toy.OUT_DIR}toybrains_n*_{outdir_suffix} ...")
        _ = toy.generate_dataset_table(n_samples=n_samples,
                                       split=split_name, 
                                       outdir_suffix=outdir_suffix)
    ## 4) generate the images
    if gen_images:
        if os.path.exists(f"{toy.DATASET_DIR}/images") and not overwrite_existing:
            print(f"Images directory {toy.DATASET_DIR}/images already exists. Not overwriting it.")
        else:
            toy.generate_dataset_images(n_jobs=n_jobs, verbose=verbose) 

    # remove the config files after generating the data for test datasets
    # if split_name!="train": os.remove(config_fname)
    return toy


def gen_toybrains_dataset(config_fname, covars, rules,
                       config_write_kwargs={"LATENTS_DIRECT"    : [], 
                                            "CONFOUNDERS"       : [], 
                                            "MEDIATORS"         : [],
                                            },
                       n_samples=1000, 
                       n_samples_test=0, n_samples_test_ood=0, 
                       lbl_name='lbl_y',
                       show_probas=None, show_dag=False, 
                       return_baseline_results=False,
                       baseline_models=[("LR",{})],  
                       baseline_metrics=["r2", 'balanced_accuracy'], trials=10,
                       gen_images=False,
                       n_jobs=-1, verbose=1, 
                       overwrite_existing=False):    
    # 1) Training data
    # prepare the name of the config file
    outdir_suffix = os.path.basename(config_fname).replace('.py','')
    # append the number of samples to the outdir_suffix manually
    outdir_suffix = f"n{n_samples}_" + outdir_suffix

    split_name='train'
    toy = _gen_toybrains_dataset(split_name, 
                                config_fname, covars, rules, config_write_kwargs=config_write_kwargs, 
                                n_samples=n_samples, show_dag=show_dag, 
                                show_probas=show_probas, 
                                outdir_suffix=outdir_suffix, 
                                overwrite_existing=overwrite_existing, 
                                gen_images=gen_images,
                                n_jobs=n_jobs, verbose=verbose)
        
    # 2) Test data
    if n_samples_test > 0:
        split_name='test_all'
        _gen_toybrains_dataset(split_name, 
                                config_fname, covars, rules, 
                                config_write_kwargs={}, 
                                n_samples=n_samples_test, show_dag=False, 
                                show_probas=None, 
                                outdir_suffix=outdir_suffix, 
                                overwrite_existing=overwrite_existing, 
                                gen_images=gen_images,
                                n_jobs=n_jobs, verbose=verbose)

    # 3) OOD Test data
    # generate the OOD test datasets by creating new config files with relavant edges of the
    # generative graph (cov-->gen / cov-->lbl-->gen) enabled at a time 
    if n_samples_test_ood > 0:
        if  len(config_write_kwargs['LATENTS_DIRECT'])==0 and len(config_write_kwargs['CONFOUNDERS'])==0 and len(config_write_kwargs['MEDIATORS'])==0:
            print("[WARN] No covariates provided either in 'LATENTS_DIRECT', 'CONFOUNDERS', or 'MEDIATORS' to generate OOD test datasets.\
 Only generating the 'test_none' OOD test data...")
        
        
        ### create an OOD test data by disabling every direct latent, confounder, and mediator

        # Generate all subsets excluding the full set (already generated as test_all) and the empty set (will be generated next as test_none)
        all_covs = config_write_kwargs['CONFOUNDERS'] + config_write_kwargs['MEDIATORS'] + config_write_kwargs['LATENTS_DIRECT']
        cov_subsets = list(itertools.chain.from_iterable(
            itertools.combinations(all_covs, r) for r in range(1, len(all_covs))  # start with atleast 1 and stop before len(all_covs)
            ))

        for cov_subset in cov_subsets: # also one more (called none) with all disabled at the same time.            
            cov_subset = sorted(cov_subset)
            # ### Create 2 version of OOD test data: 
            # (a) excluding only this covariate (b) including only this covariate
            # for OOD_type in ['with']: # ['with', 'without']
            split_name=f'test_with_'
            for i, cov in enumerate(cov_subset):
                if i>0: split_name += '--'
                split_name += f'{cov}'
            rules_cov = deepcopy(rules)
            rules_cov_out = deepcopy(rules)
            covars_copy = deepcopy(covars)

            # iterate through all edges in the DAG 
            for src_node,v in rules_cov.items():
                
                flag_disable_edge = (
                    # for the with case (b) set cov-->L to 0 for all covariates except the requested one
                    (src_node not in cov_subset) or
                    # # for the none case, set all cov-->L to 0 (uniform dist.)
                    ('none' in cov_subset)
                    # # for the without case (a) only set cov-->L to 0 of the requested covariate and
                    # (OOD_type=='without' and src_node==cov) or 
                )
                if flag_disable_edge: 
                    # (algorithm note) the disabling of cov-->y must be done carefully such that the variance of y is not affected
                    # (i) if cov== ldir (direct effect from a latent in L), 
                    # add a dummy copy of the latent variable that has the same properties as the ldir variable (dtype, n_states) so that the variance of y remains unchanged
                    if src_node in config_write_kwargs['LATENTS_DIRECT']:
                        # create a new dummy covariate with exact same rules as ldir and delete exisiting rule of ldir
                        rules_cov_out[f'cov_dummy_{src_node}'] = rules_cov[src_node]
                        del rules_cov_out[src_node]
                        # also add it in the COVARS list to avoid the sanity check errors in ToybrainsData
                        covars_copy.update({f'cov_dummy_{src_node}': {'states': list(v.keys())}})
                        
                    # for confounds / mediators just disable the c --> L path. This removes the path but keeps the variance of y unchanged
                    elif src_node in config_write_kwargs['CONFOUNDERS']+config_write_kwargs['MEDIATORS']:
                        for src_node_state, vi in v.items():
                            for dest_node, proba_args in vi.items():
                                # we don't disable cov-->y since modifying it can change the variance of y in the test data
                                if ('lbl' not in dest_node) and (dest_node != cov): 
                                    rules_cov_out[src_node][src_node_state][dest_node] = {'amt': 0}

            _gen_toybrains_dataset(split_name, config_fname, covars_copy, rules_cov_out,
                                    config_write_kwargs={},
                                    n_samples=n_samples_test_ood, show_dag=False, 
                                    show_probas=None,
                                    outdir_suffix=outdir_suffix, 
                                    overwrite_existing=overwrite_existing, 
                                    gen_images=gen_images,
                                    n_jobs=n_jobs, verbose=verbose)

        # generate one more test data 'test_none' with all cov-->L effects disabled
        split_name='test_none'
        rules_cov = deepcopy(rules)
        rules_cov_out = deepcopy(rules)
        covars_copy = deepcopy(covars)

        for src_node,v in rules_cov.items():
            if src_node in config_write_kwargs['LATENTS_DIRECT']:
                # create a new dummy covariate with exact same rules as ldir and delete exisiting rule of ldir
                rules_cov_out[f'cov_dummy_{src_node}'] = rules_cov[src_node]
                del rules_cov_out[src_node]
                # also add it in the COVARS list to avoid the sanity check errors in ToybrainsData
                covars_copy.update({f'cov_dummy_{src_node}': {'states': list(v.keys())}})

            elif src_node in config_write_kwargs['CONFOUNDERS']+config_write_kwargs['MEDIATORS']:
                for src_node_state, vi in v.items():
                    for dest_node, proba_args in vi.items():
                        if ('lbl' not in dest_node): 
                            rules_cov_out[src_node][src_node_state][dest_node] = {'amt': 0}
        
        _gen_toybrains_dataset(split_name, config_fname, covars_copy, rules_cov_out,
                                config_write_kwargs={},
                                n_samples=n_samples_test_ood, show_dag=False, 
                                show_probas=None,
                                outdir_suffix=outdir_suffix, 
                                overwrite_existing=overwrite_existing, 
                                gen_images=gen_images,
                                n_jobs=n_jobs, verbose=verbose)

        # Parallel(n_jobs=3, verbose=verbose)(
        #     delayed(gen_test_data)(split_name, rules_cov, 
        #                             n_samples_test_ood, 
        #                             outdir_suffix, overwrite_existing, verbose)
        #     for split_name, rules_cov in ood_rules.items())

    if return_baseline_results:

        baselines_file = f"{toy.DATASET_DIR}../baseline_results.csv"
        if os.path.exists(baselines_file) and not overwrite_existing:
            print(f"Baseline results file '{baselines_file}' already exists. Not overwriting it.")
        else:

            df_results_all = []
            for model, model_params in baseline_models:
                if verbose>0: print(f"Estimating ground truth associations using {model}({model_params}) model...")
                # if OOD test datasets are available then just estimate the ground truth association using them
                test_data_glob = toy.DATASET_DIR.replace('/train/', '/test_*')
                re_pattern = r"_n(\d+)"
                test_data_glob = re.sub(re_pattern, "_n*", test_data_glob)
                test_datasets =  {data_dir.rstrip('/').split('/')[-1]: data_dir \
for data_dir in glob(test_data_glob)}
                if verbose>0: print(f"holdout datasets used for baselining: {list(test_datasets.keys())}")
                
                contrib_estimator_args =  dict(
                        holdout_data=test_datasets,
                        output_labels_prefix=['lbl'], 
                        model_name=model, model_params=model_params,
                        outer_CV=trials, n_jobs=1,
                        metrics=baseline_metrics,
                        verbose=verbose)
                
                # check if there are OOD test datasets available to estimate the ground truth associations
                if n_samples_test_ood > 0 or len(test_datasets)>1:  
                    df_results = toy.fit_contrib_estimators(
                        input_feature_sets=["attr_all"],
                        **contrib_estimator_args)
                # # if not, estimate the ground truth associations using a subset of input features 
                else:
                    df_results = toy.fit_contrib_estimators(
                        input_feature_sets=["attr_all", "attr_subsets", "cov_all"],
                        **contrib_estimator_args)
                    
                df_results_all.append(df_results)

            df_results_all = pd.concat(df_results_all) if len(df_results_all)>1 else df_results_all[0]
            df_results_all.to_csv(baselines_file, index=False)

        return df_results_all


