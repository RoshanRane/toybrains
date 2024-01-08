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


