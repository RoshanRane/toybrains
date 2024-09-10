import os, sys
from glob import glob
import re
import numpy as np
import pandas as pd
from copy import deepcopy

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
                      cov_cons=[], cov_true=[]):

    opt_lists = ''
    if len(cov_cons)>0: opt_lists += f"COVARS_CONS = {cov_cons}\n"
    if len(cov_true)>0: opt_lists += f"COVARS_TRUE = {cov_true}\n"

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
                split,  
                config_fname, covars, rules, cov_cons, cov_true,
                n_samples, show_dag, show_probas, save_probas,
                outdir_suffix, overwrite_existing, gen_images,
                n_jobs, verbose):
    ## 1) Write the config file
    # for test datasets, create a temporary config file with the split name appended and later delete it
    if split!="train":
        config_fname = config_fname.replace(".py", f"_{split}.py")
    
    # write the config file if it doesn't already exist
    if os.path.exists(config_fname) and not overwrite_existing:
        print(f"Config file '{config_fname}' already exists. Not overwriting it.")
    else:
        write_config_file(config_fname, covars, rules, cov_cons, cov_true)

    ## 2) instantiate the toybrains class
    toy = ToyBrainsData(config=config_fname, save_probas=save_probas, verbose=verbose)
    # 3) show the generative graph used for sampling
    if show_dag==True:
        print(f"Config file: {config_fname}")
        display(toy.draw_dag())
    if show_probas is not None and len(show_probas)>0:
        print(f"Config file: {config_fname}")
        display(toy.show_current_config(subset=show_probas))

    ## 3) Generate the dataset by sampling the gen. variables in a table
    # check if the dataset has already been generated
    dataset_table_file = glob(f"{toy.OUT_DIR}*{outdir_suffix}/{split}/*{outdir_suffix}.csv")
    if len(dataset_table_file)>0 and not overwrite_existing:
        print(f"Dataset table '{dataset_table_file[0]}' already exists. Not overwriting it.")
    else:
        if verbose>0: print(f"Generating dataset table with {n_samples} samples and storing \
at {toy.OUT_DIR}toybrains_n*_{outdir_suffix} ...")
        _ = toy.generate_dataset_table(n_samples=n_samples,
                                       split=split, 
                                       outdir_suffix=outdir_suffix)
    ## 4) generate the images
    if gen_images:
        if os.path.exists(f"{toy.DATASET_DIR}/images") and not overwrite_existing:
            print(f"Images directory {toy.DATASET_DIR}/images already exists. Not overwriting it.")
        else:
            toy.generate_dataset_images(n_jobs=n_jobs, verbose=verbose) 

    # remove the config files after generating the data for test datasets
    if split!="train": os.remove(config_fname)
    return toy


def gen_toybrains_dataset(config_fname, covars, rules,
                       cov_cons=[], cov_true=[],
                       n_samples=1000, 
                       n_samples_test=0, n_samples_test_ood=0, 
                       lbl_name='lbl_y',
                       show_probas=None, save_probas=False, show_dag=False, 
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

    split='train'
    toy = _gen_toybrains_dataset(split, 
                                config_fname, covars, rules, cov_cons, cov_true,
                                n_samples=n_samples, show_dag=show_dag, 
                                show_probas=show_probas, save_probas=save_probas,
                                outdir_suffix=outdir_suffix, 
                                overwrite_existing=overwrite_existing, 
                                gen_images=gen_images,
                                n_jobs=n_jobs, verbose=verbose)
        
    # 2) Test data
    if n_samples_test > 0:
        split='test_all'
        _gen_toybrains_dataset(split, 
                                config_fname, covars, rules, 
                                cov_cons=[], cov_true=[],
                                n_samples=n_samples_test, show_dag=False, 
                                show_probas=None, save_probas=save_probas,
                                outdir_suffix=outdir_suffix, 
                                overwrite_existing=overwrite_existing, 
                                gen_images=gen_images,
                                n_jobs=n_jobs, verbose=verbose)

    # 3) OOD Test data
    # generate the OOD test datasets by creating new config files with relavant edges of the
    # generative graph (cov-->gen / cov-->lbl-->gen) enabled at a time 
    if n_samples_test_ood > 0:
        # add a 'none' to generate one test data with all edges disabled
        if len(cov_true)==0 and len(cov_cons)==0:
            print("[WARN] No covariates provided either in cov_true or in cov_cons to generate OOD test datasets.\
 Only generating the 'test_none' OOD test data...")
        covars_list = cov_true + cov_cons + ['none']
        # collect all the rules in a dict
        # ood_rules={}
        for _, cov in enumerate(covars_list):
            split=f'test_{cov}'
            # set all the other relations except cov-->gen / cov-->lbl-->gen to a uniform distribution
            rules_cov = deepcopy(rules)
            for src_node,v in rules_cov.items():
              for src_node_state, vi in v.items():
                for dest_node, proba_args in vi.items():
                    if (src_node != cov) or (cov=='none'): # for none, set all the rules to uniform distribution
                        rules_cov[src_node][src_node_state][dest_node] = {'amt': 0}

            _gen_toybrains_dataset(split, config_fname, covars, rules_cov,
                                    cov_cons=[], cov_true=[],
                                    n_samples=n_samples_test_ood, show_dag=False, 
                                    show_probas=None, save_probas=save_probas,
                                    outdir_suffix=outdir_suffix, 
                                    overwrite_existing=overwrite_existing, 
                                    gen_images=gen_images,
                                    n_jobs=n_jobs, verbose=verbose)
        # generate in parallel
            # ood_rules.update({split: rules_cov})

        # Parallel(n_jobs=3, verbose=verbose)(
        #     delayed(gen_test_data)(split, rules_cov, 
        #                             n_samples_test_ood, 
        #                             outdir_suffix, overwrite_existing, verbose)
        #     for split, rules_cov in ood_rules.items())

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
                
                # check if there are other test datasets than just 'test_all'
                if n_samples_test_ood > 0 or len(test_datasets)>1: 
                    df_results = toy.fit_contrib_estimators(
                        input_feature_sets=["attr_all"],
                        **contrib_estimator_args)
                # # if not, then use subsets of the attribute and estimate the ground truth associations configured
                # else:
                #     df_results = toy.fit_contrib_estimators(
                #         input_feature_sets=["attr_all", "attr_subsets", "cov_all"],
                #         **contrib_estimator_args)
                    
                df_results_all.append(df_results)

            df_results_all = pd.concat(df_results_all) if len(df_results_all)>1 else df_results_all[0]
            df_results_all.to_csv(baselines_file, index=False)

        return df_results_all


