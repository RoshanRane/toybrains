import os, sys
from copy import deepcopy
import pprint
from IPython.display import display

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR) 
from create_toybrains import ToyBrainsData

def apply_tweak_rules(rules_og, tweak_rules):
    rules = deepcopy(rules_og)
    for rule in (tweak_rules):
        cov1, state1, cov2, key, change_fn = rule
        val = rules[cov1][state1][cov2][key]
        rules[cov1][state1][cov2][key] = change_fn(val)
    return rules

def test_tweak_rules(RULES, tweak_rules, iters=5):
    rules = deepcopy(RULES)
    for i in range(iters):
        if i>0: # don't tweak the rules in iter 0
            rules = apply_tweak_rules(rules, tweak_rules)
        print(f"{'-'*50}\ni={i}")
        for k in range(len(tweak_rules)):
            key0, key1, key2, key3, _ = tweak_rules[k]
            print(f"\t {key0} = {key1} \t--> {key2}:\t {key3} = {rules[key0][key1][key2][key3]}")


def create_config_file(config_fname, covars, rules,
                       n_samples=1000, 
                       show_dag_probas=False, 
                       return_baseline_results=False,
                       baseline_metrics=["r2"], trials=10,
                       gen_images=False,
                       n_jobs=-1, verbose=1,
                       overwrite_existing=False):
    
    # write the config file
    if os.path.exists(config_fname) and not overwrite_existing:
        print(f"Config file '{config_fname}' already exists. Not overwriting it.")
    else:
        with open(config_fname, 'w') as f:
            pp = pprint.PrettyPrinter(indent=2, width=200, compact=True, sort_dicts=True)
            f.write('# List of all covariates\n\
COVARS           = {}\n\
# Rules about which covariate-state influences which generative variables\n\
RULES_COV_TO_GEN = {}\n'.format(pp.pformat(covars), 
                                pp.pformat(rules)))

    # 4) Test that the config file meets the expectation using `ToyBrainsData(config=...).show_current_config()`
    toy = ToyBrainsData(config=config_fname)
    outdir_suffix = f"n_{os.path.basename(config_fname).replace('.py','')}"
    toy.DATASET_DIR = f"{toy.OUT_DIR}_{outdir_suffix}"
    dataset_table_file = f"{toy.DATASET_DIR}/toybrains_{outdir_suffix}.csv"
    if os.path.exists(dataset_table_file) and not overwrite_existing:
        print(f"Dataset table '{dataset_table_file}' already exists. Not overwriting it.")
    else:
        df = toy.generate_dataset_table(n_samples=n_samples, verbose=verbose, 
                                        outdir_suffix=outdir_suffix)

    if show_dag_probas:
        print(f"Config file: {config_fname}")
        display(toy.show_current_config())
    
    images_dir = f"{toy.DATASET_DIR}/images"
    if gen_images:
        if os.path.exists(images_dir) and not overwrite_existing:
            print(f"Images directory '{images_dir}' already exists. Not overwriting it.")
        else:
            toy.generate_dataset_images(n_jobs=n_jobs, verbose=1) # print the image gen logs always

    if return_baseline_results:
        baselines_file = f"{toy.DATASET_DIR}/baseline_results.csv"
        if os.path.exists(baselines_file) and not overwrite_existing:
            print(f"Baseline results file '{baselines_file}' already exists. Not overwriting it.")
        else:
            df_results = toy.fit_contrib_estimators(
                input_feature_sets=["attr_all", "attr_subsets", "cov_all"], 
                output_labels=["lbls"], outer_CV=trials, inner_CV=5, n_jobs=n_jobs,
                metrics=baseline_metrics,
                debug=False,
                verbose=verbose)
            df_results.to_csv(baselines_file, index=False)

        return df_results


