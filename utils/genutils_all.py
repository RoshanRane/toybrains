import os, sys
import numpy as np
import pandas as pd
from scipy import stats
from math import log
from pprint import pprint

#####################################################################################

def make_dataset_name(lat_dir, cons, 
                    cX, cy, Xy, 
                    suffix='',
                    effect_mul=lambda e: e):
                    
    basefilename = f"con{len(cons)}_{lat_dir.replace('_','-')}"
    # append the names of the confounders to the file name
    for ci in cons:
        basefilename += "_" + ci.replace('_','-')
    if suffix != '': suffix = '_' + suffix
    return f"{basefilename}_cX{int(effect_mul(cX)):03d}_cy{int(effect_mul(cy)):03d}_Xy{int(effect_mul(Xy)):03d}{suffix}"


def break_dataset_name(dataset_name, effect_mul=1):
    # check if the dataset name contains the prefixes 'toybrains_n*_'. if yes, remove it first
    if dataset_name.startswith('toybrains_n'):
        parts = dataset_name.split('_')[2:]
    else:
        parts = dataset_name.split('_')
    n_covs = int(parts[0].replace('con',''))
    ldir = parts[1]
    ldir = '_'.join(ldir.rsplit('-', 1)) # replace the last '-' with '_' for latents
    cons = parts[2:2+n_covs]
    cons = [ci.replace('-', '_') for ci in cons]
    cX = int(parts[2+n_covs+0].replace('cX',''))/effect_mul
    cy = int(parts[2+n_covs+1].replace('cy',''))/effect_mul
    Xy = int(parts[2+n_covs+2].replace('Xy',''))/effect_mul
    suffix = parts[2+n_covs+3] if len(parts) > 4+n_covs else ''
    
    assert n_covs+5 <= len(parts) <= n_covs+6, f"dataset name {dataset_name} has more or less '_' than expected {n_covs+4}"
    
    return n_covs, ldir, cons, cX, cy, Xy, suffix

#####################################################################################

def sample_covars(n_covs=100, ctype_probs={'cont':0.34, 'cat2': 0.33, 'catn': 0.33}):


    covs = {}
    for ci in range(n_covs):
        # choose one of the three datatypes {categorical-binary, categorical-multiclass, or continuous} with 0.3 probability
        ctypes, probs = zip(*ctype_probs.items())
        ctype = np.random.choice(ctypes, p=probs)
        if ctype == 'cat2':
            states = ['s0', 's1']
        elif ctype == 'catn':
            n = np.random.randint(3, 6)
            states = [f's{i}' for i in range(n)]
            ctype = ctype.replace('n', str(n)) 
        elif ctype == 'cont':
            step = np.random.randint(1,5)/10
            start = np.random.choice([-1, 0]) if step<=0.3 else 0
            states = np.arange(start, 1+step, step)
            states = (states/states.max()).round(1).astype(str).tolist()
        else:
            raise ValueError(f"unknown covariate type {ctype}")

        covs.update({f'cov_{ci}_{ctype}': states})
    return covs

#####################################################################################

def sample_influential_covars(covs, attrs, 
                             n_covs_enc, n_covs_lbl=5, 
                             lat_direct='shape-midr_vol', 
                             verbose=0):    
    
    assert n_covs_lbl <= len(covs) 
    assert n_covs_enc <= n_covs_lbl 

    if lat_direct is None: lat_direct = np.random.choice(attrs).item()
    
    # select the n=n_covs_lbl covariates that influence the label y 
    covs_remaining = list(covs.keys())
    covs_lbl = np.random.choice(covs_remaining, 
                                size=n_covs_lbl, replace=False).tolist()
    if verbose>0: print(f"Other covariates that influence the label  (c-->y) = {covs_lbl}")

    # second, select n=n_covs_enc covariates out of the covs_lbl to also influence attributes (confounders) 
    covs_enc = np.random.choice(covs_lbl, 
                                size=n_covs_enc, replace=False).tolist()
    if verbose>0: print(f"Covariates chosen as confounder        (L<--c-->y) = {covs_enc}")

    # finally, select n=(n_attrs - n_covs_enc - 1 (lat_direct)) covariates to influence each of the remaining attributes 
    covs_remaining = list(set(covs_remaining) - set(covs_lbl))
    n_covs_attrs = len(attrs) - n_covs_enc - 1
    covs_attrs = np.random.choice(covs_remaining, size=n_covs_attrs, replace=False).tolist()
    
    # select one unique attribute per covariate
    covs_attrs += covs_enc # also the confounders get an attribute
    attrs_available = attrs.copy()
    attrs_available.remove(lat_direct)
    assert len(attrs_available) == len(covs_attrs), f"number of attributes avialable = {len(attrs_available)} \
is not equal to the number of covs that need to be assigned = {len(covs_attrs)}"
    matching_attrs = np.random.choice(attrs_available, size=len(attrs_available), replace=False).tolist()
    # match one cov to one attribute
    covs_to_attrs = dict(zip(covs_attrs, matching_attrs))
    if verbose>0: print(f"Covariates that influence img. attributes  (L<--c) = {covs_to_attrs}")

    # check that all the sampling was done correctly as expected
    assert len(covs_lbl) == n_covs_lbl
    assert len(covs_enc) == n_covs_enc
    assert len(covs_to_attrs) + 1 == len(attrs), f"number of covariates assigned to attributes = {n_covs_attrs}\
+{n_covs_enc}+1 is not equal to the number of attributes = {len(attrs)}"

    return covs_to_attrs, covs_lbl, covs_enc, lat_direct

#####################################################################################
#####################################################################################

def _map_multinomial_cats(A, B):

    # Ensure A and B are lists
    A, B = list(A), list(B)
    nA, nB = len(A), len(B)
    assert nA >= 2, "A must have at least 2 categories"
    assert nB >= 2, "B must have at least 2 categories"

    # if the categories are exactly the same then just map one to one
    if nA == nB:
        mapping = {A[i]: [B[i]] for i in range(nA)}
    elif nA > nB:
        # bin the categories in A into nB bins
        inds = np.digitize(np.arange(nA), bins=np.linspace(0, nA, nB+1)) - 1
        # print('inds A>B', inds)
        mapping = {A[i]: [B[inds[i]]] for i in range(nA)}
    else: # nB > nA
        # bin the categories in B into nA bins
        inds = np.digitize(np.arange(nB), bins=np.linspace(0, nB, nA+1)) - 1
        # print('inds B>A', inds) 
        # each category in A maps to only 1 category in B
        # mapping = {A[inds[i]]: [B[i]] for i in range(nB)} # 
        # each category in A can have more than 1 category in B
        mapping = {}
        for i in range(nB):
            a = A[inds[i]]
            if a not in mapping:
                mapping[a] = [B[i]]
            else:
                mapping[a].append(B[i])
                
    return mapping

#####################################################################################

def get_rule_for_cov(cov_name, ci_states, 
                   attr, attr_states_n, 
                   effect_size=1,
                   verbose=0):
    # scale the effect size to be between 0 and 100 and inverse of the effect of the p/1+p applied in logistic regression
    assert 0<=effect_size<=5, f"please provide an effect size between 0 and 5 so that it can be \
later adjusted by function sigmoid(effect_size) to get an appropriate probability score. Provided effect size = {effect_size}" 
    # get the datatype of the cov by checking ci_states are float or string
    if isinstance(ci_states[0], (int,float)):
        ci_type = 'cont'  
    else:
        try: # try to convert the states to float 
            if isinstance(eval(ci_states[0]), (int,float)):
                ci_type = 'cont' 
        except: # if the strings dont convert to int or float then they are categorical
                ci_type = f'cat{len(ci_states)}'

    # covariates have their datatype in their name so do an extra check if this dtype and the infered dtype match
    assert not cov_name.startswith('cov_') or (
        ci_type=='cont' and cov_name.endswith('_cont')) or (
        'cat' in ci_type and '_cat' in cov_name), f"cov_name={cov_name} (states={ci_states}) and ci_type={ci_type} do not match"

    # the weights or beta values are computed differently for continuous and categorical covariates 
    weights = []
    if 'cont' in ci_type:
        ci_states_flt = np.array(ci_states).astype(float)
        # standardize the values range and then make mean = 0 and std.dev = effect_size
        ci_states_norm = (ci_states_flt - ci_states_flt.mean())/ci_states_flt.std()
        ws = effect_size * np.absolute(ci_states_norm) # the negative effects are applied to the other class
        weights = ws.tolist()

    elif 'cat' in ci_type:
        # to make sure that the mean effect and variance is the same for catogorical and continuous covariates
        # we scale the effect size by 2.5 times (average of 3x for std.dev and 2x for variance)
        w = 2.5 * (effect_size)
        # when any of the categories occur (onehot(cat[i])==1.0) the effect size should apply
        for ci_state in ci_states:
            weights.append(w)

    # assume every covariate ci and attr are categorical and find a mapping between the states
    # of the covariate and the attribute
    mapping = _map_multinomial_cats(ci_states, list(range(attr_states_n)))

    if verbose>1: print(f"Mapping between covariate (type={ci_type}, k={len(ci_states)}) \
and attributes (name={attr}, k={attr_states_n}) = {mapping}")

    # now generate the rules mapping the cov --> gen.attr. or another cov.
    rule = {}
    for ci_state, weight in zip(ci_states, weights):
        attr_state_idx = mapping[ci_state]
        # set the weight of all other categories to log(1/(len-1)) and the current category to + weight 
        neg_weight = log(1/(attr_states_n-len(attr_state_idx))) 
        weights = np.full(attr_states_n, neg_weight)
        weights[attr_state_idx] = log(1/len(attr_state_idx)) + weight # e^(log(1/len) + weight) = 1/len * e^weight
        rule.update({ci_state: {attr: {'amt': weights.round(2).tolist()}}})     

    if verbose>0:
        print(f"Rule for \t covariate '{cov_name} (n_states={len(ci_states)})' --> attribute '{attr}(n_states={attr_states_n})'")
        pprint(rule)

    return rule

#####################################################################################

# def poisson_pmf(n_states, mode='left', lmd_offset=0):
#     '''
#     The higher the lambda, the more the distribution skews to the right
#     The lower the lambda, the more the distribution becomes a normal distribution
#     '''
#     assert n_states>=2, "number of states should be atleast 2"
#     assert lmd_offset>=0, "lmd_offset should be greater than or equal to 0"
    
#     if mode in ['left', 'right']: # left skewed
#         lmd = n_states//2 * (lmd_offset+1)
#     elif mode == 'center': # center gaussian
#         lmd = n_states//2
#     else:
#         raise ValueError("mode should be either 'left' or 'center' or 'right'")
#     # lambda cannot be negative or very small
#     weights = stats.poisson.pmf(np.arange(n_states), mu=lmd)
    
#     if mode == 'right': # right skewed
#         weights = weights[::-1]
#     elif mode == 'center':
#         weights = 2*weights
#     weights/=weights.sum()

#     return weights.round(2).tolist()
