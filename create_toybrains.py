import os, shutil, sys
from glob import glob
import re
import numpy as np
import random
import pandas as pd
from scipy.special import expit, softmax
import matplotlib
from matplotlib import pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid") #'dark_background','seaborn-v0_8-muted', 'seaborn-v0_8-notebook',
import seaborn as sns
import math
from PIL import Image, ImageDraw
from colordict import ColorDict
from joblib import Parallel, delayed  
from tqdm import tqdm
import argparse
import importlib
from datetime import datetime
from tabulate import tabulate
from slugify import slugify
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
import skimage.io as io
# add custom imports
from utils.fitmodel import fitmodel
from utils.confounds import CounterBalance
import utils.metrics
# from utils.vizutils import plot_col_dists
import graphviz
# from causalgraphicalmodels import CausalGraphicalModel # Does not work with python 3.10

#################################  Helper functions  ###############################################

class PROB_VAR:
    '''Class to init a probabilistic variable that has states with a probability 
     distribution which can be modified and sampled from'''
    def __init__(self, name, states, 
                link_fn=softmax, basis_fn=lambda z: z): # alternative is log lambda z: math.log(z) if z!=0 else 0
        # link function set to the logistic function by default
        # for other options refer to https://www.statsmodels.org/stable/glm.html#link-functions
        self.name = name
        self.states = np.array(states)
        self.k = len(states)
        self._link_fn = link_fn
        self._basis_fn = basis_fn
        self.reset_weights()
        
    def _apply_link_safely(self, z):
        ''' The activation function that converts the weights into probabilities is applied with some safety checks and adjustments'''
        probas = self._link_fn(z)
        # force very low probas and -inf to 0.0
        probas[probas<1e-5] = 0.0 
        # force very high probas and +inf to 1.0
        probas[probas>1-1e-5] = 1.0
        # if the error is larger than 0.1 then force the probability to sum to one with a warning
        if abs(1.0 - probas.sum())>0.1: 
            print(f"[WARN] The probabilities of the states of {self.name} do not sum to 1.0. \
            The sum of the probabilities computed from weight {z.tolist()} = {probas.sum()}.\
            The probabilities are being forced to sum to 1.")
            probas = probas/probas.sum()
        else:  # adjust minor rounding errors 
            probas = probas/probas.sum()

        return probas
        
    def set_weights(self, weights):
        # if weights are directly provided then just set them
        assert len(weights)==self.k, f"provided len(weights={weights}) are not equal to configured len(states={self.states})."
        self.weights = np.array(weights)
        

    def bump_up_weight(self, idxs=None, amt=1):
        '''bump up the existing weight at 'idx' by 'amt' '''
        # if no idx given then it implies all idxs
        if idxs is None: 
            idxs = np.arange(self.k)
        elif isinstance(idxs, (int,float)): # and not a list
            idxs = [idxs]
        if isinstance(amt, (int,float)):
            amt = [amt]*len(idxs)
        assert len(amt)==self.k, f"len(amt={amt}) are not equal to len(states={self.states})"

        for i,idx in enumerate(idxs):
            try:
                self.weights[idx] += amt[i]
            except IndexError as e:
                print(f"\n[IndexError] index={idx} is out-of-bound for variable '{self.name}' \
with n={self.k} states {self.states} and weights {self.weights}")
                raise e
        # self._smooth_weights()
        
        return self
        
    def sample(self):
        probas = self._get_probas()
        sample = np.random.choice(self.states, p=probas).item()
        self.last_sample = sample
        return sample
    
    def reset_weights(self):
        self.weights = np.zeros(self.k)
        self.last_sample = None

    def _get_probas(self):
        # if weights are not set at all then set them to be uniform before computing the probas
        if self.weights.sum()==0: 
            probas = np.full(self.k, 1/self.k) # sample from a uniform distribution
        else:
            # apply basis func (like GAMs) to the weights only if provided
            if self._basis_fn is None:
                z = self.weight
            else:
                z = np.vectorize(self._basis_fn)(self.weights)

            probas = self._apply_link_safely(z)

        return probas
        
#     def _smooth_weights(self): 
#         """Smooths the self.weights numpy array by taking the 
#         average of its neighbouring values within a specified window.
#         Args:
#         - window (int): the window size for smoothing
#         """
#         # Pad the array with ones for the sliding window
#         # for odd len arrays pad differently as opposed to even lenght arrays
#         #  
#         window=2 #TODO currently only works with 2 
#         pad = (window//2-1, window//2)
#         arr = np.pad(self.weights, pad, mode='edge')
#         # Create a 2D array of sliding windows
#         shape = arr.shape[:-1] + (arr.shape[-1]-window+1, window)
#         strides = arr.strides + (arr.strides[-1],)
#         windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        
#         # Take the average of each sliding window to smooth the array
#         self.weights = np.mean(windows, axis=1)
#         # remove the paddings
#         # windows  = windows[pad[0]-1:-pad[1]]
#         return self


#######################################################################################################################
##################################               Main Class             ###############################################
#######################################################################################################################

class ToyBrainsData:
    
    def __init__(self,  
                 config=None, 
                 out_dir="dataset/", 
                 img_size=64, seed=None, 
                 verbose=0,
                 save_probas=False):
        
        self.I = img_size
        self.OUT_DIR = out_dir
        self.verbose = verbose
        self.save_probas = save_probas
        # forcefully set a random seed
        if self.verbose>2: seed = 42
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # the center of the image for drawing reference
        self.ctr = (self.I/2, self.I/2) #np.random.randint(self.I/2-2,self.I/2+2, size=2)
        
        # initialize all the generative properties for the images and the labels and covariates
        self._setup_genvars_covars()
        
        # Import the covariates and covariates-to-image-attribute relationships
        if config is not None:
            self._load_config(config)
        else:
            self.COVARS = {}
            self.RULES_COV_TO_GEN = {}
            if self.verbose>0: print("[WARN] No config file provided when instantiating a ToyBrainsData() object.")
            
        self._init_df()
        # store a dict of results tables (run.csv) on the dataset
        self.results = {"baseline_results":None,
                        "supervised_results":None,
                        "unsupervised_results":None,
                       }
        self.IMAGES_ARR = {}

        # check the metrics defined in utils/metrics.py
        self.my_custom_scorers = [func for func in dir(utils.metrics) if '_'!=func[0]] 

    ### INIT METHODS
    def _init_df(self):
        # Initialize a table to store all dataset attributes
        columns = sorted(list(self.COVARS.keys())) + sorted(list(self.GENVARS.keys()))
        if self.save_probas: columns += [f'probas_{var}' for var in columns]                
        self.df =  pd.DataFrame(columns=columns)
        self.df.index.name = "subjectID"  
    
    
    def _load_config(self, config_file):
        # if user provided '.py' in the config filename argument then remove it
        # dynamically import the config file using its relative path
        if config_file[-3:]=='.py': config_file = config_file[:-3]
        config_dir = os.path.dirname((config_file+'.py'))
        if config_dir not in sys.path:
            sys.path.append(config_dir)
        config = importlib.import_module(os.path.basename(config_file))

        assert hasattr(config, 'COVARS')
        self.COVARS = {cov: PROB_VAR(name=cov, **states) for cov, states in config.COVARS.items()}
        assert hasattr(config, 'RULES_COV_TO_GEN')
        # sanity checks
        for cov, rules in config.RULES_COV_TO_GEN.items():
            assert cov in self.COVARS.keys() or cov in self.GENVARS.keys(), f"[CONFIG file Sanity check] In the rules RULES_COV_TO_GEN, the covariate {cov} has not been previously defined in COVARS\
and it is not a generative attribute either."
            if cov in self.COVARS.keys(): # only check for covars and not for genvars
                for cov_state, updates in rules.items():
                    if not isinstance(cov_state, tuple):
                        assert cov_state in self.COVARS[cov].states, f"[CONFIG file Sanity check] In the rules RULES_COV_TO_GEN, the {cov}={cov_state} has not been previously defined in COVARS"
                    else: # continuous states
                        for cov_state_i in cov_state:
                            assert cov_state_i in self.COVARS[cov].states, f"[CONFIG file Sanity check] In the rules RULES_COV_TO_GEN, the {cov}={cov_state_i} has not been previously defined in COVARS"
                
        self.RULES_COV_TO_GEN = config.RULES_COV_TO_GEN

        # compute the nodes and edges of the configuration for forming a Causal Graphical Model
        nodes, edges = set(), set()
        for c, c_states in self.RULES_COV_TO_GEN.items():
            nodes.add(c)
            for _, ats in c_states.items():
                for at in ats.keys():
                    nodes.add(at)
                    edges.add((c,at))
                    
        self.CGM_nodes = sorted(list(nodes))
        self.CGM_edges = sorted(list(edges))    
        
            
    def _reset_vars(self):
        [gen_var.reset_weights() for _, gen_var in self.GENVARS.items()]
        [gen_var.reset_weights() for _, gen_var in self.COVARS.items()]


    def _get_rule(self, cov_name, cov_state):
        '''Function handles situations in the config of RULES_COV_TO_GEN such as when
         user provides a tuple of values as cov_state (continuous) instead of a single cov_state
         Return None if no rule is defined for the covariate = cov_state'''
        rules = self.RULES_COV_TO_GEN[cov_name] 
        if cov_state in rules.keys():
            return rules[cov_state]
        else:
            for cov_rule in rules.keys():
                if isinstance(cov_rule, tuple):
                    if cov_state in cov_rule:
                        return rules[cov_rule]
                    else:
                        raise ValueError(f"[ERROR] The covariate {cov_name} has a state {cov_state} \
that has no defined rules in the config file.")
            
    def _adjust_covar_dists_and_sample(self, fix_covar={}):
        # separate the covars into parent and child nodes of the causal links
        child_covs = set()
        # generative attributes of the images can also be parents of some covars like the label y
        # if they exist, they must be be sampled here already to sample the child nodes
        parent_gens = set()
        for cov_name, cov_state in self.RULES_COV_TO_GEN.items():
            if cov_name in self.GENVARS.keys():
                parent_gens.add(cov_name)
            for cov_state, rules in cov_state.items():
                for child_node, child_node_rule in rules.items():
                    if child_node in self.COVARS.keys():
                        child_covs.add(child_node)

        child_covs = list(child_covs)
        parent_gens = list(parent_gens)
        parent_gen_vars = {gen_name: self.GENVARS[gen_name] for gen_name in parent_gens}

        # first, sample the parent covars and parent genvars
        covars = {}
        for cov_name, covar in list(self.COVARS.items()) + list(parent_gen_vars.items()):
            if cov_name not in child_covs:
                if cov_name in fix_covar: # if predefined just use it (for viz purpose)
                    cov_state = fix_covar[cov_name]
                else:
                    cov_state = covar.sample()
                covars.update({cov_name: cov_state})
                # adjust the probability of the child covars if there is a rule
                rules = self._get_rule(cov_name, cov_state)
                if rules is not None:
                    # if fix_covar is provided (visualization reason) then only adjust the weights univariately w.r.t to only this covars 
                    if len(fix_covar)==0 or (cov_name in fix_covar):
                        for var, rule in rules.items():
                            if var in child_covs:
                                # print(f"[D][adjust_covar_dists_and_sample] {cov_name}(state={cov_state}) bumps weights of {var} with rule {rule['amt']}")
                                self.COVARS[var].bump_up_weight(**rule)
                
        # now, sample the child covars with the adjusted dists
        for cov_name, covar in self.COVARS.items():
            if cov_name in child_covs:
                covars.update({cov_name: covar.sample()})
        
        return covars
        
    def _adjust_genvar_dists(self, covars): 
        """Configure the relationship between covariates and the generative attributes"""
        ### model `Covariates -> image attributes` distribution
        for cov_name, cov_state in covars.items(): 
            try: 
                rules = self._get_rule(cov_name, cov_state)
            except Exception as e:
                # print unconfigured covs for debugging
                if self.verbose>0 and ('lbl' not in cov_name): 
                    print(f"[WARN] No rules have been defined for covariate = {cov_name} with state = {cov_state}. skipping .. \n{e}")
                continue
            if rules is not None:
                for var, rule in rules.items():
                    # ignore rules that are for cov->cov relation as they have already been handled in adjust_covar_dists_and_sample()
                    if var in self.GENVARS:
                        # print("[D][adjust_genvar_dists()] bumping weights of ", var, "with rule", rule)
                        self.GENVARS[var].bump_up_weight(**rule)
              
    def update_tweaks(self, RULES_COV_TO_GEN_TWEAKED):
        for key, states in RULES_COV_TO_GEN_TWEAKED.items():
            for state, new_values in states.items():
                 self.RULES_COV_TO_GEN[key][state] = new_values          
        
##########################################      methods for CONFIG FILE           ###################################################
#####################################################################################################################################

    def show_all_states(self):
        # generative image attributes
        print("\nImage attributes (fixed):")
        data = []
        for var_name, var in self.GENVARS.items():
            data.append([var_name, str(var.states),  str(var.weights)])
        print (tabulate(data, headers=["Name", "States", "Weights"]))
            
        # covariates
        print("\nExemplary covariates & labels (customizable):")
        data = []
        for var_name, var in self.COVARS.items():
            data.append([var_name, str(var.states),  str(var.weights)])
        print (tabulate(data, headers=["Name", "States", "Weights"]))
                
    # def get_config_causal_graph(self): 
    #     return CausalGraphicalModel(nodes=self.CGM_nodes, 
    #                                 edges=self.CGM_edges)

    def show_current_config(self, subset=['all']):
        """
        show_attr_probas = True : shows how the sampling probability distribution of the different
                                  image attributes change for each covariate/label state. This feature
                                   is used to verify that all the intended weight changes were applied.
        """
        # convert to one-to-many dict of source node to destination nodes
        src_to_dst_map = {}
        for src, dst in self.CGM_edges:
            if src not in src_to_dst_map: # create a new entry
                src_to_dst_map.update({src:[dst]})
            else: # append to list of dst nodes
                src_to_dst_map[src] += [dst]
        
        # show one plot for each src node to all it's destination node
        for cov_name, dst_nodes in sorted(src_to_dst_map.items()):
            cov = self.COVARS[cov_name] if cov_name in self.COVARS else self.GENVARS[cov_name]
            # hack: reduce plotting time by only taking few states when there are more than 10 states
            cov_states = cov.states if len(cov.states)<=10 else  cov.states[::len(cov.states)//10]
            # for every possible state of the covariate, collect the final weights dist. of each dst node
            df_temps = []
            for cov_state in cov_states:
                fix_covar = {cov_name : cov_state}
                
                df_temp = pd.DataFrame(index=range(1000)) 
                for node_name in dst_nodes:
                    # reset all other values to only observe the (univariate) relationship of src_node --> dst_node
                    self._reset_vars() 
                    # print('[D] prob ({} ({}) --> {})'.format(cov_name, cov_state, node_name))
                    # init_weights = self.GENVARS[node_name].weights if node_name in self.GENVARS else self.COVARS[node_name].weights
                    # print('[D] init:', np.array(init_weights).round(2).tolist())
                    if node_name in self.GENVARS:
                        # only trigger the change in weights caused when covariate = cov_state
                        # do not set any other covariate state
                        self._adjust_genvar_dists(covars=fix_covar)
                        # print('[D] update dst_node in genvar:', np.array(self.GENVARS[node_name].weights).round(2).tolist())
                        node = self.GENVARS[node_name]
                    elif node_name in self.COVARS:
                        self._adjust_covar_dists_and_sample(fix_covar=fix_covar)
                        # print('[D] update dst_node in covar:', np.array(self.COVARS[node_name].weights).round(2).tolist())
                        node = self.COVARS[node_name]
                        
                    k = list(range(node.k))
                    df_temp.loc[k, node_name] = node.states
                    # normalize weights to get p-distribution 
                    node_probas = node._get_probas()*100 
                    # print('[D] final:', np.array(node_probas).astype(int).tolist())
                    df_temp.loc[k, f"{node_name}_probas"] = node_probas
                    
                # add the cov state to the whole df                     
                df_temp = df_temp.dropna(how='all')
                df_temp[cov_name] = cov_state
                # if cov_states[0] in [True, False]:
                #     df_temp[cov_name] = df_temp[cov_name].astype(bool)
                df_temps.append(df_temp)
            df = pd.concat(df_temps, ignore_index=True)
            # display(df)

            if ('all' in subset) or (cov_name in subset):
                # create figure
                sns.set_style("whitegrid")
                fs = 12
                f,axes = plt.subplots(1, len(dst_nodes), 
                                        figsize=(2+2*len(dst_nodes), 2.5),
                                        sharex=False, sharey=True)
                axes = axes.ravel() if not isinstance(axes, matplotlib.axes.Axes) else [axes]
            
                for i, (ax, node) in enumerate(zip(axes, dst_nodes)):
                    legend = (i==(len(axes)-1))
                    # just print the values that will be plot
                    # add 1% random noise to the probas of different cov_name states for better visualization
                    df[f"{node}_probas"] = df[f"{node}_probas"] + np.random.uniform(-1,1, size=len(df))
                    g = sns.lineplot(df, hue=cov_name, 
                                        alpha=0.8,
                                        x=node, y=f"{node}_probas",  
                                        ax=ax, legend=legend)
                    # for line in g.get_lines():
                    #     ax.fill_between(line.get_xdata(), [0], color='blue', alpha=.25)
                    if legend: 
                        # make 2 cols if there are many lagend labels
                        sns.move_legend(g, "upper left", bbox_to_anchor=(1,1.2), frameon=True) 
                        
                    # set xlabel and ylabel  and title
                    ax.set_title(r"{}  $\longmapsto$".format(cov_name)+'\n'+node, fontsize=fs)
                    ax.set_ylabel("Proba. dist. (%)", fontsize=fs)
                    ax.set_xlabel(None)
                    ax.set_ylim([0,100]) # probability
                    
                    # set the ticks to be same as the unique values in the list
                    states = df[node].sort_values().unique().tolist() 
                    new_tick_labels, new_ticks = [], []
                    for i, tick in enumerate(states):
                        if isinstance(tick, str):
                            new_ticks.append(i)
                            # trim the xtick labels if it is too long
                            if len(tick)>5: tick = tick[:5]
                            new_tick_labels.append(tick)
                        elif isinstance(tick, float):
                            if tick.is_integer(): tick = int(tick) 
                            new_ticks.append(tick)
                            new_tick_labels.append(str(tick))
                        elif isinstance(tick, bool):
                            new_ticks.append(tick)
                            new_tick_labels.append(str(bool(tick)))
                    # print("ticklabels",new_tick_labels, [t.get_text() for t in ax.get_xticklabels()])
                    ax.set_xticks(new_ticks, labels=new_tick_labels) 
            
                plt.tight_layout()
                plt.show()   
    

    def draw_dag(self):
        """dot file representation of the Causal Graphical Model (CGM)."""
        dot = graphviz.Digraph(graph_attr={'rankdir': ''})
        # separate out the source nodes from the destination nodes
        src_nodes, dst_nodes = zip(*self.CGM_edges)
        src_nodes = sorted(list(set(src_nodes)))
        dst_nodes = sorted(list(set(dst_nodes)))
        # middle nodes will be in both src_nodes and dst_nodes. 
        bridge_nodes = [n for n in dst_nodes if n in src_nodes]
        # keep these only in the src_nodes list
        dst_nodes    = [n for n in dst_nodes if n not in src_nodes]
        # categorize all nodes (attrib vars) into groups for easy reading
        src_grps = sorted(list(set([node.split("_")[0] for node in src_nodes])))
        # dont do it for bridge nodes
        dst_grps = sorted(list(set([node.split("_")[0] for node in dst_nodes])))

        grp_to_rgb_map_src = {grp:(30, (60+i*100)%255, 200) for i, grp in enumerate(src_grps)} 
        grp_to_rgb_map_dst = {grp:(200, (60+i*75)%255,  30) for i, grp in enumerate(dst_grps)}
        grp_to_rgb_map = {**grp_to_rgb_map_src, **grp_to_rgb_map_dst}
        
        def get_color_hex(grp, alpha=100):
            r,g,b = grp_to_rgb_map[grp]
            return f'#{r:x}{g:x}{b:x}{alpha:x}'
        
        # add all source nodes
        for node in src_nodes:
            grp = node.split("_")[0]
            settings = {"shape": "ellipse", "group":grp, 
                        "style":"filled", "fillcolor": get_color_hex(grp), 
                        # "color":color_hex,"penwidth":"2"
                        }
            dot.node(node, node, settings)
            
        # add destination nodes
        for grp in dst_grps:
            # add each destination grp as a parent node and each sub category as child node
            alpha = 100
            r,g,b = grp_to_rgb_map[grp]
            color_hex = f'#{r:x}{g:x}{b:x}{alpha:x}'
            
            settings = {"shape": "ellipse", "group":grp, 
                        "style":"filled", "fillcolor": get_color_hex(grp), 
                        } 
            
            with dot.subgraph(name=f'cluster_dst') as dot_dst:
                dot_dst.attr(rank="dst", style="invis")
                with dot_dst.subgraph(name=f'cluster_dst_{grp}') as dot_c:
                    dot_c.attr(label=grp, labelloc='b', style="dashed")
                    for node in dst_nodes:
                        if node.split("_")[0] == grp:
                            dot_c.node(node, "_".join(node.split("_")[1:]), **settings)
            
        for a, b in self.CGM_edges:
            # set the arrow color same as the color of the attrib variable
            grp = b.split("_")[0]
            dot.edge(a, b, _attributes={"color": get_color_hex(grp, alpha=200), 
                                        "style":"bold",
                                        # "penwidth":"2",
                                        "arrowhead":"vee"})

        return dot
        
##########################################  methods for GENERATING DATASET    #######################################################
#####################################################################################################################################

    def _setup_genvars_covars(self):
        # define all the generative properties for the images
        self.GENVARS = {
            # 1. brain_vol created as a ellipse with a minor and major radius
            # ranging between 1633 to 2261 [(S/2-12)*(S/2-6) to (S/2-8)*(S/2-2)]
            'brain-vol_radminor': PROB_VAR('brain-vol_radminor', 
                                       np.arange(self.I/2 - 12, self.I/2 - 8 + 1, dtype=int)),
            'brain-vol_radmajor': PROB_VAR('brain-vol_radmajor', 
                                       np.arange(self.I/2 - 6, self.I/2 - 2 + 1, dtype=int)),                        
            # 2. brain_thick: the thickness of the blue border around the brain ranging between 1 to 4
            'brain_thick':        PROB_VAR('brain_thick', np.arange(1,4+1, dtype=int)), 
            # 3. the intensity or brightness of the brain region ranging between 'greyness1' (210) to 'greyness5' (170)
            'brain-int_fill':     PROB_VAR('brain-int_fill', [210,200,190,180,170]), 
            # 4. the intensity or brightness of the ventricles and brain borders ranging between 'blueness1' to 'blueness3' 
            'brain-int_border':   PROB_VAR('brain-int_border', ['0-mediumslateblue','1-slateblue','2-darkslateblue','3-darkblue']),
            # ventricle (the 2 touching arcs in the center) thickness ranging between 1 to 4
            'vent_thick':         PROB_VAR('vent_thick', np.arange(1,4+1, dtype=int)),
            # 'vent_curv' (TODO) curvature of the ventricles ranging between ..
        }
                                
        # also add the gen props of the 5 shapes 
        self.SHAPE_POS = {'shape-top': (self.I*.5, self.I*.22),
                          'shape-midr':(self.I*.7, self.I*.4),
                          'shape-midl':(self.I*.3, self.I*.4),
                          'shape-botr':(self.I*.6, self.I*.7),
                          'shape-botl':(self.I*.4, self.I*.7)}
                                          
        for shape_pos in self.SHAPE_POS.keys():
            
            self.GENVARS.update({
                # color of the regular polygon from shades of green to shades of red
                f'{shape_pos}_int' : PROB_VAR(f'{shape_pos}_int',
                                                ['0-indianred','1-salmon','2-lightsalmon',
                                                '3-palegoldenrod','4-lightgreen','5-darkgreen']), 
                # the final volume of the circle inside which the regular polygon is drawn
                # 4 steps of radius [2,3,4,5] times 4 steps of curvature (sides of the regular polygon) from [3,4,6,12]
                f'{shape_pos}_vol': PROB_VAR(f'{shape_pos}_vol',
                                            np.arange(0, 16, dtype=int)),
                # TODO # rot = np.random.randint(0,360)
            })
        
        
        
    def generate_dataset_table(self, n_samples, 
                               outdir_suffix='', 
                               outdir_append_sample_size=True, split='train'):
        self._init_df()
        
        if self.verbose>2:
            print(f"Generative parameter {' '*7}|{' '*7} States \n{'-'*60}")
            for name, var in self.GENVARS.items():
                print(f"{name} {' '*(25-len(name))} {var.states}")
        
        for subID in tqdm(range(n_samples)):
            # first reset all generative image attributes to have uniform distribution
            self._reset_vars()
            # (1) sample the covariates, labels, or any parent gen.attrs. for this data point
            covars = self._adjust_covar_dists_and_sample()
            #  and adjust the image attribute probability distributions
            self._adjust_genvar_dists(covars) 
            # (2) sample the image attributes conditional on the sampled labels and covariates
            genvars = {}
            for var_name, gen_var in self.GENVARS.items():
                if gen_var.last_sample is None:
                    genvars.update({var_name: gen_var.sample()})
                else: # might already be sampled during adjust_covar_dists_and_sample() so reuse it
                    genvars.update({var_name: gen_var.last_sample})
            # combine the gen params 'brain-vol_radmajor' and 'brain-vol_radminor' into one 'brain_vol'
            genvars.update({'brain-vol': math.pi * genvars['brain-vol_radmajor'] * genvars['brain-vol_radminor']})
            # TODO calculate the exact volume of the regular polygon shapes and store that too? 

            # (3) store the covars and then the generative attributes
            for k,v in covars.items():
                self.df.at[f'{subID:05}', k] = v
            for k,v in genvars.items():
                self.df.at[f'{subID:05}', k] = v

            if self.save_probas:
                # also store the probability of the states of the genvars and the covars
                all_vars = list(self.GENVARS.items()) + list(self.COVARS.items())
                for k,v in all_vars:
                    self.df.at[f'{subID:05}', f'probas_{k}'] = v._get_probas().round(2).tolist()
        
        # create the output folder and save the table
        # add sample size to output dir name and create the folders
        n_samples_regex = re.compile(r"n\d+")
        outdir_name = 'toybrains'
        # append sample size only if it is not already in the outdir_suffix
        if outdir_append_sample_size and not n_samples_regex.search(outdir_suffix): 
            outdir_name += f"_n{n_samples}"
        if outdir_suffix:
            outdir_name += f'_{outdir_suffix}'
            
        self.DATASET_DIR = f"{self.OUT_DIR}{outdir_name}/{split}/"
        # delete previous folder if they already exist
        shutil.rmtree(self.DATASET_DIR, ignore_errors=True)
        os.makedirs(self.DATASET_DIR)
        # save the dataframe in the dataset folder
        self.df.to_csv(f"{self.DATASET_DIR}/{outdir_name}.csv")
        
        return self.df
    
        
    def generate_dataset_images(self, n_jobs=10,
                                verbose=1):
        """Use the self.df and create the images and store them"""
        n_samples = len(self.df)
        if verbose>0: print("Generating n={} toybrain images".format(n_samples))
        shutil.rmtree(f"{self.DATASET_DIR}/images", ignore_errors=True)
        os.makedirs(f"{self.DATASET_DIR}/images")
        # os.makedirs(f"{self.OUT_DIR}/masks", exist_ok=True)
        
        # Generate images and save them 
        Parallel(n_jobs=n_jobs)(
                            delayed(
                                self._gen_image)(subidx) 
                                    for subidx in tqdm(range(n_samples)))
            
            
    def _gen_image(self, df_idx):
        subject = self.df.iloc[df_idx]
        genvars = {}
        for key,val in subject.items():
            if key[:4] not in ['lbl_','cov_']:
                # convert to int for ImageDraw package
                if isinstance(val, float): val = int(val)
                genvars.update({key: val})
                
        # Create a new image of size 64x64 with a black background and a draw object on it
        image = Image.new('RGB',(self.I,self.I),(0,0,0))
        draw = ImageDraw.Draw(image)
        # Draw the brain 
        # (a) Draw an outer ellipse of the image
        x0,y0 = (self.ctr[0]-genvars['brain-vol_radminor'],
                 self.ctr[1]-genvars['brain-vol_radmajor'])
        x1,y1 = (self.ctr[0]+genvars['brain-vol_radminor']-1,
                 self.ctr[1]+genvars['brain-vol_radmajor']-1)

        draw.ellipse((x0,y0,x1,y1), 
                     fill=self.get_color_val(genvars['brain-int_fill']), 
                     width=genvars['brain_thick'],
                     outline=self.get_color_val(genvars['brain-int_border'])
                    )
        # create the brain mask
        # brain_mask = (np.array(image).sum(axis=-1) > 0) #TODO save it?

        # (b) Draw ventricles as 2 touching arcs 
        xy_l = (self.I*.3, self.I*.3, self.I*.5, self.I*.5)
        xy_r = (self.I*.5, self.I*.3, self.I*.7, self.I*.5)
        draw.arc(xy_l, start=+310, end=+90, 
                 fill=self.get_color_val(genvars['brain-int_border']), 
                 width=genvars['vent_thick'])
        draw.arc(xy_r, start=-290, end=-110, 
                 fill=self.get_color_val(genvars['brain-int_border']), 
                 width=genvars['vent_thick'])

        # (c) draw 5 shapes (triangle, square, pentagon, hexagon, ..)
        # with different size, color and rotations
        for shape_pos, (x,y) in self.SHAPE_POS.items():
            vol = genvars[f'{shape_pos}_vol']
            rad = 2 + vol//4 # radius of the regular polygon that fit best in the brain region are in range [2,5]
            # number of sides in the regular polygon from 3 (triangle) to 12 (dodecagon) which is almost a circle
            # the higher the number of sides the higher the total volume of the shape
            n_sides = [3,4,7,12][vol%4]
            
            draw.regular_polygon((x, y, rad), 
                                 n_sides=n_sides, 
                                 rotation=np.random.randint(0,360),
                                 fill=self.get_color_val(genvars[f'{shape_pos}_int']), 
                                 outline=self.get_color_val(genvars['brain-int_border']))
        # (d) save the image
        image.save(f"{self.DATASET_DIR}/images/{subject.name}.jpg")
        
    
    def generate_dataset(self, n_samples, n_jobs=10, outdir_suffix='n'):
        """Creates toy dataset and save to disk."""
        # first generate dataset table and update self.df
        self.generate_dataset_table(n_samples,
                                     outdir_suffix=outdir_suffix)
        self.generate_dataset_images(n_jobs=n_jobs)
            
            
    def load_generated_dataset(self, dataset_path):
        dataset_path = dataset_path + '/train/'
        data_table = glob(f'{dataset_path}/toybrains_*.csv')
        assert len(data_table)==1, f"It is expected to find only one dataset table. However,\
 at the provided location {dataset_path} the following dataset tables were found = '{data_table}'. "
        self.DATASET_DIR = dataset_path
        self.df = pd.read_csv(data_table[0])
         
            
    def get_color_val(self, color):
        if isinstance(color, str):
            # if there is a number tag on the color then remove it
            if '-' in color: color = color.split('-')[-1]
            val = [int(c) for c in ColorDict()[color]]
        else:
            val = [int(color) for i in range(3)]
        # print(color, tuple(val))
        return tuple(val)
    
    
    def area_of_regular_polygon(self, n, r):
        # Calculate the length of each side of the polygon
        side_length = 2 * r * math.sin(math.pi / n)
        # Calculate the area of the polygon
        area = 0.5 * n * side_length * r
        return area

##########################################                                  #############################################
##########################################  methods for BASELINE MODEL FIT  #############################################
##########################################                                  #############################################
    
    # run baseline on both attributes and covariates
    def fit_contrib_estimators(self,
                            input_feature_sets=["attr_all", 
                                                 "attr_subsets", 
                                                 "cov_subsets"],
                            output_labels_prefix=["lbl", "cov"],
                            model_name="LR", model_params={},
                            conf_ctrl={},
                            metrics=["r2", "balanced-accuracy", "roc-auc"],
                            holdout_data=None,
                            # compute_shap=False,
                            outer_CV=5, n_jobs=-1, 
                            verbose=0,
                            random_seed=None):
        ''' run linear regression or logistic regression to estimate the expected prediction performance for a given dataset. 
Fits [input features] X [output labels] X [model x cross validation folds] models where,
    [input features] can be either:
            i) "attr_all": all input features are fed at once to predict the labels.
            ii) attr_subsets:  subsets of input features are created using the current causal graph (config file) and fed as input to individual models.
            iv) cov_subsets: All other covariates are used as input features and then each subsets of covariates are created using the current causal graph (config file) and fed as input.
    [output labels] each of the covariates in the dataset.
            v) cov_all:  all available covariates (starting with 'cov_') are fed as input features at once.
    [model] is determined automatically. It can be:
            i) logistic regression if label is binary
            ii) multiple logistic regression if multiclass label 
            iii) linear regression if continuous label

        PARAMETERS
        ----------
        outer_CV: int, default : 5
            number of trials (outer cross validations) to run the model fit
        test_dataset : str, default : None
            provide a separate test dataset / holdout dataset to evaluate all the model on.
            if set to None, then the test dataset is created as 20% of the training dataset using train_test_split.
        random_seed : int, default : 42
            random state for reproducibility
        '''
        # sanity check that the dataset table has been loaded
        assert len(self.df)>0 and hasattr(self, "DATASET_DIR"), "first generate the dataset table \
using self.generate_dataset_table() method or load an already generated dataset using \
self.load_generated_dataset()"  
        for metric in metrics:
            assert metric in (list(sklearn.metrics.get_scorer_names()) + self.my_custom_scorers) or ('logodds' in metric), f"metric_name '{metric}' is invalid.\
    It should be one of the sklearn.metrics.get_scorer_names()"
        start_time = datetime.now()

        # load the dataset csv tables
        df_csv_path = glob(f"{self.DATASET_DIR}/toybrains_*.csv")
        assert len(df_csv_path)==1, f"In {self.DATASET_DIR}, either no dataset tables were found or more than 1 tables were found = {df_csv_path}."
        df_data = pd.read_csv(df_csv_path[0]).set_index("subjectID")
        # load also the test dataset if provided
        holdout_dfs_dict = {}
        if holdout_data is not None:
            for holdout_name, holdout_data_i in holdout_data.items():
                df_hold_csv_path = glob(f"{holdout_data_i}/toybrains_*.csv")
                # print('[D]', holdout_name, df_hold_csv_path) 
                assert len(df_hold_csv_path)==1, f"In {holdout_data}, either no dataset tables were found or more than 1 tables were found = {df_hold_csv_path}."
                df_holdout_i = pd.read_csv(df_hold_csv_path[0]).set_index("subjectID")
                df_holdout_i['dataset_dir'] = holdout_data_i # required by self._load_images
                holdout_dfs_dict.update({holdout_name: df_holdout_i}) 

        # load the images of each dataset if 'images' are provided as input
        if "images" in input_feature_sets:
            df_data['dataset_dir'] = self.DATASET_DIR # required by self._load_images
            for splitname, df_data_i in [('traintest', df_data), *holdout_dfs_dict.items()]:
                self._load_images(df_data_i, name=splitname, verbose=verbose)

        labels = []
        for lbl_prefix in output_labels_prefix:
            labels += self.df.filter(regex = f'^{lbl_prefix}').columns.tolist()  
        assert len(labels)>0, "no labels are selected to be predicted. Please select at least one label from ['lbls', 'covs']"

        # create the directory to store the results
        model_params_str = self._convert_model_params_to_str(model_params)
        results_out_dir = f"{self.DATASET_DIR}/../baseline_results/{model_name}/{model_params_str}" 
        shutil.rmtree(results_out_dir, ignore_errors=True)
        os.makedirs(results_out_dir)
        
        # if no confound control methods are requested then create a 'default' one
        conf_ctrl.update({None: []})

        # generate the different settings of [input features] X [output labels] X [cross validation]
        all_settings = []

        for lbl in labels:
            # get the respective list of input feature columns for each feature_type 
            for fea_name, fea_cols in self._get_feature_cols(input_feature_sets, 
                                                             lbl=lbl).items():
                # if input is images append the loaded image pixel arrays to the dataframe
                if fea_name == "images":
                    df_data = df_data.join(fea_cols['traintest']) # fea_cols is a dict {split: dataframe, ..}
                    # do the same with the provided holdout data
                    for holdout_name, df_holdout_i in holdout_dfs_dict.items():
                        holdout_dfs_dict[holdout_name] = df_holdout_i.join(fea_cols[holdout_name])
            
                    # change fea_cols to the names of the pixel columns
                    fea_cols = fea_cols['traintest'].columns.tolist()

                # apply a confound control methods if requested
                for conf_ctrl_method, conf_cols in conf_ctrl.items():
                    # only select the input features and the label in the data table
                    data_columns = fea_cols + [lbl, f'probas_{lbl}'] + conf_cols
                    df_data_i = df_data[data_columns]

                    # create 'outer_CV' number of dataset categorization into training, and test sets
                    datasplits = self._split_dataset(
                                        df_data_i, stratify_by=lbl,
                                        CV=outer_CV,
                                        random_seed=random_seed)
                    
                    for trial_i, (df_train_i, df_test_i) in enumerate(datasplits):

                        # generate a confound controlled df_train_i datasplit
                        df_train_i, success = self._apply_conf_ctrl(conf_ctrl_method,
                                                           data=df_train_i, lbl=lbl,
                                                           confs=conf_cols,
                                                           random_state=random_seed, verbose=verbose>1)
                        # skip if the confound control was not successful
                        if not success: continue
                        
                        # drop the confound columns from the train & test data
                        df_train_i = df_train_i.drop(columns=conf_cols)
                        df_test_i = df_test_i.drop(columns=conf_cols)

                        other_kwargs = {
                            "dataset" : self.DATASET_DIR, 
                            "holdout_datasets" : list(holdout_data.items()) if holdout_data is not None else "None",
                            "type" : f"{conf_ctrl_method}{conf_cols}" if conf_ctrl_method is not None else "baseline",
                            "n_samples" : len(self.df),
                            "n_samples_test" : len(df_test_i),
                            }

                        all_settings.append(dict(
                                inp = fea_name,
                                out = lbl,
                                trial = trial_i,
                                model_name = model_name,
                                model_params = model_params,
                                train_data = df_train_i,
                                test_data = df_test_i,
                                inp_fea_list = fea_cols,
                                holdout_data = holdout_dfs_dict,
                                results_out_dir=results_out_dir,
                                # compute_shap = compute_shap,
                                metrics=metrics,
                                random_seed=random_seed, 
                                verbose=verbose,
                                results_kwargs=other_kwargs))
        
        if verbose>0: print(f"{'-'*50}\n[parallel jobs] Estimating baseline contrib scores on dataset: {os.path.basename(self.DATASET_DIR.rstrip('/'))}\
\n ... running a total of {len(all_settings)} different settings of [input] x [output] x [CV]")

        # run each model fit in parallel
        with Parallel(n_jobs=n_jobs) as parallel:
            parallel(
                delayed(
                    self._fit_contrib_estimator)(**settings) for settings in (all_settings))

        # merge run_*.csv into one run.csv
        df_out = pd.concat([pd.read_csv(csv) for csv in glob(f"{results_out_dir}/run-temp*.csv")], 
                           ignore_index=True)
        # Reorder columns and sort the final output table for readability
        col_order = ["dataset", "out", "inp", "trial", "model", "type"]
        df_out = df_out.sort_values(col_order) 
        col_order = col_order + [c for c in df_out.columns if c not in col_order]
        df_out = df_out[col_order] 
        if verbose>1: print("generated results table with {} rows and {} columns.".format(*df_out.shape))
        if verbose>2: print(df_out["dataset", "out", "inp", "trial", "model", "type", "score_test_balanced_accuracy"])
        df_out.to_csv(f"{results_out_dir}/run.csv", index=False)

        # delete the temp csv files
        os.system(f"rm {results_out_dir}/run-temp*.csv")
        
        runtime = str(datetime.now()-start_time).split(".")[0]
        if verbose>0: 
            print(f'TOTAL fit_contrib_estimators RUNTIME: {runtime}')
            print('--'*50)
        self.results["baseline_results"]= df_out
        return df_out



    def _apply_conf_ctrl(self, method, data, lbl, confs,
                         random_state=None, verbose=0):
        
        if method is None: return data, True

        for c in confs:
            assert c in data.columns, f"'{c}' not in the dataset columns. \
    Cannot control for confound variables that are not in the dataset."
        
        if "sample" in method:
            
            # first decide if we should do upsample or downsample or a mix of both
            if "upsample" in method: oversample = True
            elif "downsample" in method: oversample = False
            else: oversample = None
            sampler = CounterBalance(oversample=oversample, 
                                     random_state=random_state, debug=verbose>2)
            
            # create the categories on which to data should be balanced
            groups = np.zeros(len(data), dtype=int)
            for i, c in enumerate(confs):                
                c = data[c].astype('category')
                groups += c.cat.codes*10**(i)
                assert 1 < len(c.cat.categories) <= 10, f"confound variable '{c}' has \
n={len(c.cat.categories)} unique value. Currently we support a max of 10 and min of 2 categories."
            if verbose>0: 
                print(f"Created n={len(np.unique(groups))} unique groups '{list(np.unique(groups))}' for counterbalancing.")
            
            try:
                new_data, _ = sampler.fit_resample(data, y=data[lbl], groups=groups)
            except Exception as e:
                if 'has only 1 class or very few subjects in 1 of the classes' in str(e):
                    print(e, "[WARN] skipped confound control for this sample.")
                    return data, False
            
        # elif method == "residualize":
        #     return Residualize(cols).fit_transform(data)
        else:
            raise ValueError(f"conf_ctrl={method} is not a valid confound control method.\
Currently supported methods are ['residualize','upsample', 'downsample', 'resample']")
        
        if verbose>0: print(f"[apply_conf_ctrl] Applied confound control method '{method}' for confound variables {confs}. \
Data changed size from {data.shape} to {new_data.shape}.")
        
        return new_data, True
    

    def _fit_contrib_estimator(self,
                            train_data, test_data,
                            inp, out, trial, 
                            model_name, model_params,
                            metrics, 
                            inp_fea_list,
                            holdout_data,
                            # compute_shap,
                            results_out_dir,
                            random_seed, verbose, 
                            results_kwargs):
        ''' run one baseline linear model for the given 
        [label, features] with 'trial' number of cross-validation folds''' 
        
        start_time = datetime.now()

        if verbose>1: 
            inp_fea_list_print = inp_fea_list
            if len(inp_fea_list) > 10:
                inp_fea_list_print = inp_fea_list[:3] + ["..."] + inp_fea_list[-3:]
            print(f'Model           : {model_name}({model_params})')
            print(f'Input Features  :(name={inp}, n={len(inp_fea_list)}) {inp_fea_list_print}')
            print(f'Output label    : {out}')
            print(f'Confound control: {results_kwargs["type"]}')

        # run logistic regression and linear regression for tabular dataset # TODO support SVM too
        # compute_shap = (compute_shap) and (model_name.upper() in ['LR'] and inp in ["attr_all"]) 
        results_dict, model_config = fitmodel(
                                            train_data, test_data,
                                            X_cols=inp_fea_list, y_col=out,
                                            model_name=model_name, model_params=model_params,
                                            holdout_data=holdout_data,
                                            # compute_shap=compute_shap,
                                            metrics=metrics,
                                            random_seed=random_seed)

        # if compute_shap:
        #     # extract the SHAP scores and store as individual columns
        #     shap_scores = results_dict['shap_contrib_scores']
        #     results_dict.update({f"shap__{k}":v for k,v in shap_scores})
        #     results_dict.pop('shap_contrib_scores')

        result = {
            "inp": inp, "out": out, "trial": trial,
            **results_dict,
            "inp_fea_list":inp_fea_list, "random_seed": random_seed,
            **model_config,
            **results_kwargs,
            "runtime": str(int((datetime.now()-start_time).total_seconds()))
        }
        
        # save the results as a csv file
        pd.DataFrame([result]).to_csv(
            f"{results_out_dir}/run-temp_out-{out}_inp-{inp}_{model_config['model']}_{results_kwargs['type']}_{trial}.csv", 
                  index=False)

        
    def _split_dataset(self, 
                       df_data, stratify_by, 
                       CV=1, random_seed=42):
        """Split the dataset into training, validation, and test sets."""
        #TODO change label to categorical if required
        # df['label'] = pd.factorize(df[label])[0]
        
        df_splits = []
        stratify_by = df_data[stratify_by]
        # if n_trials=1 then just split training data into 80% train and 20% validation sets
        if CV == 1:
            df_train, df_test = train_test_split(df_data, test_size=0.2,
                                                 stratify=stratify_by, 
                                                 random_state=random_seed)
            df_splits.append((df_train, df_test))
        else: # split into 'CV' folds 
            skf = StratifiedKFold(n_splits=CV, 
                                  shuffle=True, random_state=random_seed)
            
            for train_idx, test_idx in skf.split(df_data, stratify_by):
                df_splits.append((df_data.iloc[train_idx], 
                                  df_data.iloc[test_idx])) # same test data for all folds
        
        return df_splits
    

    def _convert_model_params_to_str(self, model_params):
        if len(model_params)==0: 
            return "default"
        else:
            model_parems_str = "_".join([f"{k}-{v}" for k,v in model_params.items()])
            return slugify(model_parems_str)


    def _get_feature_cols(self, feature_types, lbl=''):
        
        all_attr_cols = [n for n,_ in self.GENVARS.items()]
        all_cov_cols  = [n for n,_ in self.COVARS.items() if lbl!=n]
        # create a list all cov to image attribute relations in the CGM
        attr_subsets = []
        cov_subsets = []
        for node in self.CGM_nodes:
            child_nodes = []
            child_nodes_covs = []
            for parent, child in self.CGM_edges:
                if (parent==node):
                    # when a cov is the child node then store this in a separate dict
                    if 'cov_' in child or 'lbl_' in child:
                        child_nodes_covs.append(child)
                    else:
                        child_nodes.append(child)
            if len(child_nodes): attr_subsets.append(child_nodes)
            if len(child_nodes_covs): cov_subsets.append(child_nodes_covs)
                
        features_dict = {}
        for f_type in feature_types:
            if f_type == "attr_subsets":
                # superset = []
                for subset in attr_subsets:
                    subset_name = ", ".join(subset)
                    features_dict.update({f"attr_{subset_name}": subset})

                    # superset.extend(subset)
                    # also add each attribute individually
                    # if len(subset)>1:
                    #     for attr in subset:
                    #         features_dict.update({attr_{attr}: [attr]})
                # finally add the superset of all attributes subsets as one feature
                # superset = list(set(superset))
                # if superset not in cov_subsets:
                #     features_dict.update({"attr_superset": superset})
                # features_dict.update({"attr_superset": list(set(superset))})

            elif f_type == "attr_superset":
                superset = sorted(list(set([s for subset in attr_subsets for s in subset])))
                features_dict.update({"attr_superset": superset})
        
            elif f_type == "cov_subsets":
                superset = []
                for subset in cov_subsets:
                    subset_name = ", ".join(subset)
                    features_dict.update({subset_name: subset})
                    superset.extend(subset)
                    # also add each attribute individually
                    # if len(subset)>1:
                    #     for cov in subset:
                    #         features_dict.update({cov: [cov]})
                # superset = list(set(superset))
                # if superset not in cov_subsets:
                #     features_dict.update({"cov_superset": superset})

            elif f_type == "attr_all":
                features_dict.update({"attr_all": all_attr_cols})
            elif f_type == "cov_all":
                features_dict.update({"cov_all": all_cov_cols})
            elif f_type == "images":
                features_dict.update({"images": self.IMAGES_ARR})
            else:
                raise ValueError(f"{f_type} is an invalid feature_type. \
Valid input features for the contribution estimation modelling are \
['images', 'attr_all','attr_subsets','cov_all','cov_subsets']. \
See doc string for more info on what each tag means.")
        
        # ensure that the provided lbl is not in the input feature set
        for _, val in features_dict.items():
            if lbl in val: val.remove(lbl)

        return features_dict
    

    def _load_images(self, df_data, name='traintest',verbose=0):
        assert 'dataset_dir' in df_data.columns, "df_data should have a column 'dataset_dir' that points to the location of the images"
        dataset_dir = df_data['dataset_dir'].iloc[0]
        # dont load images if it is already loaded
        if (name not in self.IMAGES_ARR) or (len(df_data)!=len(self.IMAGES_ARR[name])):
            if verbose>1: print(f"Loading {len(df_data)} images from disk '{dataset_dir}/images/*.jpg' ...")
            
            img_files = [f"{dataset_dir}/images/{subID:05}.jpg" for subID in df_data.index]
            with Parallel(n_jobs=-1) as parallel:
                img_arrs = parallel(delayed(_read_img)(f) for f in img_files)
            # verify that the parallel loading of images worked for all images
            subIDs, img_arrs = zip(*sorted(img_arrs, key=lambda x: x[0]))
            # zip it together in a dataframe
            col_names = ['p_'+'-'.join(map(str, i)) for i in np.ndindex(img_arrs[0].shape)]
            df_imgs = pd.DataFrame(np.stack(img_arrs).reshape(len(subIDs),-1), 
                                   index=subIDs, 
                                   columns=col_names)
            
            self.IMAGES_ARR.update({name: df_imgs}) 



def _read_img(img_path):
    subID = int(os.path.basename(img_path).split(".")[0])
    return (subID, io.imread(img_path))

##############################################  END  ###################################################
##############################################  END  ###################################################
##############################################  END  ###################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_samples', default=100, type=int)
    parser.add_argument('--dir', default='dataset/toybrains', type=str)
    parser.add_argument('-c', default = 'configs.lbl1cov1', type=str)
    parser.add_argument('-v', '--verbose', default=0, type=int)
    parser.add_argument('-j','--n_jobs', default=20, type=int)
    parser.add_argument('-s','--suffix', default = 'n', type=str)
    # parser.add_argument('--img_size', default=64, type=iny, help='the size (h x h) of the generated output images and labels')
    args = parser.parse_args()

    IMG_SIZE = 64 # 64 pixels x 64 pixels
    RANDOM_SEED = 42 if args.verbose>2 else None
    # create the output folder
    dataset = ToyBrainsData(out_dir=args.dir, 
                            config=args.c,
                            img_size=IMG_SIZE, verbose=args.verbose, 
                            seed=RANDOM_SEED)   
    # create the shapes dataset
    dataset.generate_dataset(n_samples=args.n_samples, 
                             n_jobs=args.n_jobs, 
                             outdir_suffix=args.suffix,
                            )