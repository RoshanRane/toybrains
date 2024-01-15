import os, shutil, sys
from glob import glob
import numpy as np
import random
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid") #'dark_background','seaborn-v0_8-muted', 'seaborn-v0_8-notebook',
import seaborn as sns
import math
from PIL import Image, ImageDraw
from colordict import ColorDict
from joblib import Parallel, delayed  
from copy import copy, deepcopy
from tqdm import tqdm
import argparse
import importlib
from datetime import datetime
from tabulate import tabulate

# add custom imports
from utils.dataset import split_dataset
from utils.baseline import run_lreg
importlib.reload(sys.modules[run_lreg.__module__])

import graphviz
# from causalgraphicalmodels import CausalGraphicalModel # Does not work with python 3.10

# from utils.vizutils import plot_col_dists

#################################  Helper functions  ###############################################

class PROB_VAR:
    '''Class to init a probabilistic variable that has states with a probability 
     distribution which can be modified and sampled from'''
    def __init__(self, name, states):
        self.name = name
        self.states = np.array(states)
        self.k = len(states)
        self.reset_weights()
        
    def bump_up_weight(self, idxs=None, amt=1):
        # if no idx given then it implies all idxs
        if idxs is None: 
            idxs = np.arange(self.k)
        elif isinstance(idxs, (int,float)): # and not a list
            idxs = [idxs]
            
        if isinstance(amt, (int,float)):
            amt = [amt]*len(idxs)
        for i,idx in enumerate(idxs):
            try:
                self.weights[idx] += amt[i]
            except IndexError as e:
                print(f"\n[IndexError] index={idx} is out-of-bound for variable '{self.name}' \
with n={self.k} states {self.states} and weights {self.weights}")
                raise e
        # self._smooth_weights()
        assert len(self.weights)==self.k, f"len(weights={self.weights}) are not equal to len(states={self.states}).\
 Something failed when performing self._smooth_weights()"
        return self
        
    def sample(self):
        probas = self.weights/self.weights.sum()
        return np.random.choice(self.states, p=probas).item()
    
    def reset_weights(self):
        self.weights = np.ones(self.k)
        
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

################################## Main Class ############################################

class ToyBrainsData:
    
    def __init__(self,  
                 config=None, 
                 out_dir="dataset/toybrains", 
                 img_size=64, seed=None, 
                 debug=False):
        
        self.I = img_size
        self.OUT_DIR = out_dir
        self.debug = False if (debug == 'False' or debug == False) else True
            
        # forcefully set a random seed
        if self.debug: seed = 42
        
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
            print("[WARN] No config file provided.")
            
        self._init_df()
        # store a dict of results tables (run.csv) on the dataset
        self.results = {"baseline_results":None,
                        "supervised_results":None,
                        "unsupervised_results":None,
                       }
        

    ### INIT METHODS
    def _init_df(self):
        # Initialize a table to store all dataset attributes
        columns = sorted(list(self.COVARS.keys())) + sorted(list(self.GENVARS.keys()))                
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
            assert cov in self.COVARS.keys(), f"[CONFIG file Sanity check] In the rules RULES_COV_TO_GEN, the covariate {cov} has not been previously defined in COVARS"
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
         user provides a tuple of values as cov_state (continuous) instead of a single cov_state'''
        rules = self.RULES_COV_TO_GEN[cov_name]
        if cov_state in rules.keys():
            return rules[cov_state]
        else:
            for cov_rule in rules.keys():
                if isinstance(cov_rule, tuple):
                    if cov_state in cov_rule:
                        return rules[cov_rule]
            else:
                return None
            
    def _adjust_covar_dists_and_sample(self, fix_covar={}):
        # separate the covars into parent and child nodes of the causal links
        child_covs = set()
        for cov_name, cov_state in self.RULES_COV_TO_GEN.items():
            for cov_state, rules in cov_state.items():
                for child_node, child_node_rule in rules.items():
                    if child_node in self.COVARS.keys():
                        child_covs.add(child_node)
        child_covs = list(child_covs)
                
        # first, sample the parent covars
        covars = {}
        for cov_name, covar in self.COVARS.items():
            if cov_name not in child_covs:
                if cov_name not in fix_covar:
                    cov_state = covar.sample()
                else:
                    cov_state = fix_covar[cov_name]
                covars.update({cov_name: cov_state})
                # adjust the probability of the child covars if there is a rule
                rules = self._get_rule(cov_name, cov_state)
                for var, rule in rules.items():
                    if var in child_covs:
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
            rules = self._get_rule(cov_name, cov_state)
            if (cov_name not in self.RULES_COV_TO_GEN.keys()) or (rules is None):
                # print unconfigured covs for debug
                if self.debug: print(f"[WARN] No rules have been defined for covariate = {cov_name} with state = {cov_state}")
                continue
            else:
                for var, rule in rules.items():
                    # ignore rules that are for cov->cov relation
                    if var in self.GENVARS:
                        self.GENVARS[var].bump_up_weight(**rule)
              
    def update_tweaks(self, RULES_COV_TO_GEN_TWEAKED):
        for key, states in RULES_COV_TO_GEN_TWEAKED.items():
            for state, new_values in states.items():
                 self.RULES_COV_TO_GEN[key][state] = new_values          
        
##########################################  methods for config and viz #############################################

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

    def show_current_config(self, 
                            show_dag_probas=True, 
                            return_causal_graph=False):
        """
        show_attr_probas = True : shows how the sampling probability distribution of the different
                                  image attributes change for each covariate/label state. This feature
                                   is used to verify that all the intended weight changes were applied.
        """

        
        # draw return a graphviz `dot` object, which jupyter can render
        if show_dag_probas:
            
            # convert to one-to-many dict of source node to destination nodes
            src_to_dst_map = {}
            for src, dst in self.CGM_edges:
                if src not in src_to_dst_map: # create a new entry
                    src_to_dst_map.update({src:[dst]})
                else: # append to list of dst nodes
                    src_to_dst_map[src] += [dst]
            
            # show one plot for each src node to all it's destination node
            for cov_name, dst_nodes in sorted(src_to_dst_map.items()):
                cov = self.COVARS[cov_name]
                # hack: reducing time by only taking few states when there are more than 5 states
                cov_states = cov.states if len(cov.states)<=10 else  cov.states[::len(cov.states)//10]
                # interatively collect the weights dist. of each dst node for every possible state of the covariate
                df_temps = []
                for cov_state in cov_states:
                    fix_covar = {cov_name : cov_state}
                    
                    df_temp = pd.DataFrame(index=range(1000)) 
                    for node_name in dst_nodes:
                        self._reset_vars() 
                        if node_name in self.GENVARS:
                            # only trigger the change in weights caused when covariate = cov_state
                            # do not set any other covariate state
                            self._adjust_genvar_dists(fix_covar)
                            node = self.GENVARS[node_name]
                        elif node_name in self.COVARS:
                            self._adjust_covar_dists_and_sample(fix_covar)
                            node = self.COVARS[node_name]
                            
                        k = list(range(node.k))
                        df_temp.loc[k, node_name] = node.states
                        # normalize weights to get p-distribution 
                        node_probas = (node.weights/node.weights.sum())*100 
                        df_temp.loc[k, f"{node_name}_probas"] = node_probas
                        
                    # add the cov state to the whole df                     
                    df_temp = df_temp.dropna(how='all')
                    df_temp[cov_name] = cov_state
                    if cov_states[0] in [True, False]:
                        df_temp[cov_name] = df_temp[cov_name].astype(bool)
                    df_temps.append(df_temp)
                df = pd.concat(df_temps, ignore_index=True)
                # display(df)

                # create figure
                sns.set(style="ticks")
                sns.set_style("whitegrid")
                fs = 12
                f,axes = plt.subplots(1, len(dst_nodes), 
                                      figsize=(2+2*len(dst_nodes), 3),
                                      sharex=False, sharey=True)
                f.suptitle(r"{}    $\longmapsto$   ".format(cov_name), fontsize=fs+2)
                axes = axes.ravel() if not isinstance(axes, matplotlib.axes.Axes) else [axes]
            
                for i, (ax, node) in enumerate(zip(axes, dst_nodes)):
                    legend = (i==(len(axes)-1))
                    g = sns.lineplot(df, hue=cov_name, 
                                     x=node, y=f"{node}_probas",  
                                     ax=ax, legend=legend)
                    # for line in g.get_lines():
                    #     ax.fill_between(line.get_xdata(), [0], color='blue', alpha=.25)

                    if legend: 
                        # make 2 cols if there are many lagend labels
                        sns.move_legend(g, loc="upper right", 
                                        bbox_to_anchor=(1.6,1), frameon=True) 
                        
                    # set xlabel and ylabel  and title
                    ax.set_title(node, fontsize=fs)
                    ax.set_ylabel("Proba. dist. (%)", fontsize=fs)
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
        
        return self.draw_dag()
    
    
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
        
##########################################  methods for generating dataset #############################################

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
                # number of sides in the regular polygon from 3 (triangle) to 12
                f'{shape_pos}_curv': PROB_VAR(f'{shape_pos}_curv',
                                              np.arange(3,12, dtype=int)), 
                # color of the regular polygon from shades of green to shades of red
                f'{shape_pos}_int' : PROB_VAR(f'{shape_pos}_int', 
                                            ['0-indianred','1-salmon','2-lightsalmon',
                                             '3-palegoldenrod','4-lightgreen','5-darkgreen']), 
                # the radius of the circle inside which the regular polygon is drawn
                f'{shape_pos}_vol-rad': PROB_VAR(f'{shape_pos}_volrad', 
                                            np.arange(2,5+1, dtype=int)),  
                # TODO # rot = np.random.randint(0,360)
            })
        
        
        
    def generate_dataset_table(self, n_samples, outdir_suffix='n'):
        self._init_df()
        
        if self.debug:
            print(f"Generative parameter {' '*7}|{' '*7} States \n{'-'*60}")
            for name, var in self.GENVARS.items():
                print(f"{name} {' '*(25-len(name))} {var.states}")
            
        print("Sampling image gen. attributes for n={} toybrain samples".format(n_samples))
        
        for subID in tqdm(range(n_samples)):
            # first reset all generative image attributes to have uniform distribution
            self._reset_vars()
            # (1) sample the covariates and labels for this data point
            covars = self._adjust_covar_dists_and_sample()
            #  and adjust the image attribute probability distributions
            self._adjust_genvar_dists(covars)
            # (2) sample the image attributes conditional on the sampled labels and covariates 
            genvars = {var_name: gen_var.sample() for var_name, gen_var in self.GENVARS.items()}
            
            # combine the gen params 'brain-vol_radmajor' and 'brain-vol_radminor' into one 'brain_vol'
            genvars.update({'brain-vol': math.pi*genvars['brain-vol_radmajor']*genvars['brain-vol_radminor'] })
            # calculate the volume of all regular_polygon shapes
            for shape_pos in self.SHAPE_POS.keys():
                genvars.update({
                    f'{shape_pos}_vol': 
                    self.area_of_regular_polygon(
                        n=genvars[f'{shape_pos}_curv'], r=genvars[f'{shape_pos}_vol-rad'])})# TODO
                
            # (3) store the covars and then the generative attributes
            for k,v in covars.items():
                self.df.at[f'{subID:05}', k] = v
            for k,v in genvars.items():
                self.df.at[f'{subID:05}', k] = v
        
        # create the output folder and save the table
        # add sample size to output dir name and create the folders
        if outdir_suffix[0]=='n': 
            outdir_suffix = outdir_suffix.replace('n',f"n{n_samples}", 1)
            
        self.OUT_DIR_SUF = f"{self.OUT_DIR}_{outdir_suffix}"
        # delete previous folder if they already exist
        shutil.rmtree(self.OUT_DIR_SUF, ignore_errors=True)
        os.makedirs(self.OUT_DIR_SUF)
        # save the dataframe in the dataset folder
        self.df.to_csv(f"{self.OUT_DIR_SUF}/toybrains_{outdir_suffix}.csv")
        
        return self.df
    
        
    def generate_dataset_images(self, n_jobs=10):
        """Use the self.df and create the images and store them"""
        n_samples = len(self.df)
        print("Generating n={} toybrain images".format(n_samples))
        shutil.rmtree(f"{self.OUT_DIR_SUF}/images", ignore_errors=True)
        os.makedirs(f"{self.OUT_DIR_SUF}/images")
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

            draw.regular_polygon((x,y,genvars[f'{shape_pos}_vol-rad']), 
                                 n_sides=genvars[f'{shape_pos}_curv'], 
                                 rotation=np.random.randint(0,360),
                                 fill=self.get_color_val(genvars[f'{shape_pos}_int']), 
                                 outline=self.get_color_val(genvars['brain-int_border']))
        # (d) save the image
        image.save(f"{self.OUT_DIR_SUF}/images/{subject.name}.jpg")
        
    
    def generate_dataset(self, n_samples, n_jobs=10, outdir_suffix='n'):
        """Creates toy dataset and save to disk."""
        # first generate dataset table and update self.df
        self.generate_dataset_table(n_samples,
                                     outdir_suffix=outdir_suffix)
        self.generate_dataset_images(n_jobs=n_jobs)
            
            
    def load_generated_dataset(self, dataset_path):
        self.OUT_DIR_SUF = dataset_path
        self.df = pd.read_csv(glob(f'{dataset_path}/toybrains_*.csv')[0])
         
            
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

##########################################  methods for baseline models fit  #############################################
    
    # run baseline on both attributes and covariates
    def fit_baseline_models(self,
                            input_feature_types=["attr_all", "attr_subsets", "cov_subsets"],
                            CV=5, n_jobs=10, 
                            random_seed=None, debug=False):
        ''' run linear regression or logistic regression to estimate the expected prediction performance for a given dataset. 
Fits [input features] X [output labels] X [model x cross validation folds] models where,
    [input features] can be either:
            i) attr_subsets: all image attributes are fed as input and then several subsets of attributes are created using the current causal graph (config file) and fed as input.
            ii) cov_subsets: All other covariates are used as input features and then each subsets of covariates are created using the current causal graph (config file) and fed as input.
    [output labels] each of the covariates in the dataset.
    [model] can be:
            i) logistic regression if binary label
            ii) multiple logistic regression if multiclass label 
            iii) linear regression if continuous label

        PARAMETERS
        ----------
        CV : int, default : 10
            number of cross validation
        random_seed : int, default : 42
            random state for reproducibility
        debug : boolean, default : False
            debug mode
        '''
        # sanity check that the dataset table has been loaded
        assert len(self.df)>0 and hasattr(self, "OUT_DIR_SUF"), "first generate the dataset table \
using self.generate_dataset_table() method or load an already generated dataset using \
self.load_generated_dataset()" 
        
        start_time = datetime.now()
        df_csv_path = glob(f"{self.OUT_DIR_SUF}/toybrains_*.csv")[0]
            
        labels = self.df.filter(regex = '^lbl').columns.tolist() + self.df.filter(regex = '^cov').columns.tolist()
        # create the directory to store the results
        results_out_dir = f"{self.OUT_DIR_SUF}/baseline_results" 
        shutil.rmtree(results_out_dir, ignore_errors=True)
        os.makedirs(results_out_dir)
        if debug: #simplify
            n_jobs=1
            CV=2
            random_seed=42
        
        for_result_out = {
            "dataset" : self.OUT_DIR_SUF,
            "type" : "baseline",
            "n_samples" : len(self.df)}
           
        # generate the different settings of [input features] X [output labels] X [cross validation]
        all_settings = []
        for lbl in labels:
            # get the respective list of input feature columns for each feature_type requested
            # Also, exclude the label from feature columns list
            input_features_dict = self._get_feature_cols(input_feature_types, lbl=lbl)
            for fea_name, fea_cols in input_features_dict.items():
                # if the label is in features list then remove it
                if lbl in fea_cols: fea_cols.remove(lbl)
                # and ensure the new list of features is not empty
                if len(fea_cols)==0: continue
                for cv_i in range(CV):
                    all_settings.append({"trial":cv_i, 
                                         "label":lbl, 
                                         "features":fea_cols, 
                                         "feature_set_name":fea_name})
        
        print(f'running a total of {len(all_settings)} different settings of \
[input] x [output] x [CV] and saving result in {self.OUT_DIR_SUF}')
        
        # run each model fit in parallel
        with Parallel(n_jobs=n_jobs) as parallel:
            parallel(
                delayed(
                    self._fit_baseline_model)(
                        **settings,
                        df_csv_path=df_csv_path,
                        CV=CV,
                        results_out_dir=results_out_dir,
                        random_seed=random_seed, debug=debug, 
                        results_kwargs=for_result_out
                    ) for settings in tqdm(all_settings))

        # merge run_*.csv into one run.csv
        df_out = pd.concat([pd.read_csv(csv) for csv in glob(f"{results_out_dir}/run-bsl_*.csv")], 
                           ignore_index=True)
        # Reorder columns and sort the final output table for readability
        col_order = ["dataset", "out", "inp", "trial", "model", "type"]
        df_out = df_out.sort_values(col_order) 
        col_order = col_order + [c for c in df_out.columns if c not in col_order]
        df_out = df_out[col_order] 
        df_out.to_csv(f"{results_out_dir}/run.csv", index=False)

        # delete the temp csv files
        os.system(f"rm {results_out_dir}/run-bsl_*.csv")

        runtime = str(datetime.now()-start_time).split(".")[0]
        print(f'TOTAL RUNTIME: {runtime}')
        self.results["baseline_results"]= df_out
        self._show_baseline_results()
        return df_out
    

    # run one using settings
    def _fit_baseline_model(
        self,
        trial, label, features, 
        feature_set_name,
        df_csv_path,
        CV, results_out_dir,
        random_seed, debug, results_kwargs):
        ''' run one baseline linear model for the given 
        [label, features] with 'trial' number of cross-validation folds''' 
        data = []
        # split the dataset for training, validation, and test from raw dataset
        for data_split in split_dataset(df_csv_path, label, 
                                        CV, trial, 
                                        random_seed, 
                                        debug):
            # get the input and output
            X = data_split[features]
            y = data_split[label].values
            data.append([X, y])
        # display(data)
                
        if debug: 
            print(f'Features: {feature_set_name}')
            print(f'Input features: {data[0][0].columns.tolist()}')
            print(f'Output label  : {label}')

        # run logistic regression and linear regression for tabular dataset
        compute_shap = feature_set_name in ["attr_all"] #"attr_superset", 
        results_dict, model_config = run_lreg(data, 
                                              compute_shap=compute_shap)
        if debug:
            print(f"Train metric: {results_dict['train_metric']:>8.4f} "
                  f"Validation metric: {results_dict['val_metric']:>8.4f} "
                  f"Test metric: {results_dict['test_metric']:>8.4f}")

        result = {
            "inp" : feature_set_name, 
            "out" : label,
            "trial" : trial,
            **results_dict,
            **model_config,
            **results_kwargs
        }
        
        # save the results as a csv file
        pd.DataFrame([result]).to_csv(
            f"{results_out_dir}/run-bsl_lbl-{label}_inp-{feature_set_name}_{trial}-of-{CV}_{model_config['model']}.csv", 
                  index=False)
        

    def _get_feature_cols(self,
                          feature_types, lbl=''):
        
        if "attr_subsets" in feature_types: feature_types.append("attr_all")
        if "cov_subsets" in feature_types: feature_types.append("cov_all")
        all_attr_cols = [n for n,_ in self.GENVARS.items()]
        all_cov_cols  = [n for n,_ in self.COVARS.items()]
                
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
                superset = []
                for subset in attr_subsets:
                    subset_name = ", ".join(subset)
                    features_dict.update({f"attr_{subset_name}": subset})

                    superset.extend(subset)
                    # also add each attribute individually
                    # if len(subset)>1:
                    #     for attr in subset:
                    #         features_dict.update({attr_{attr}: [attr]})
                # finally add the superset of all attributes subsets as one feature
                superset = list(set(superset))
                if superset not in cov_subsets:
                    features_dict.update({"attr_superset": superset})
                features_dict.update({"attr_superset": list(set(superset))})
            
                        
            elif f_type == "cov_subsets":
                cov_cols = [cov for cov in all_cov_cols if lbl!=cov]
                superset = []
                for subset in cov_subsets:
                    subset_name = ", ".join(subset)
                    features_dict.update({subset_name: subset})
                    superset.extend(subset)
                    # also add each attribute individually
                    # if len(subset)>1:
                    #     for cov in subset:
                    #         features_dict.update({cov: [cov]})
                superset = list(set(superset))
                if superset not in cov_subsets:
                    features_dict.update({"cov_superset": superset})

            elif f_type == "attr_all":
                features_dict.update({"attr_all": all_attr_cols})
            elif f_type == "cov_all":
                if len(cov_cols)>1: 
                    features_dict.update({"cov_all": cov_cols})
            else:
                raise ValueError(f"{f_type} is an invalid feature_type. Valid input features for the baseline modelling are ['attr_subsets', 'cov_subsets']. See doc string for more info on what each tag means.")
        return features_dict
    
    # summary
    def _show_baseline_results(self, 
        split=None,
        cmap='YlOrBr',
    ):
        ''' summary 

        PARAMETER
        ---------
        df_data : pandas.dataframe
            run.csv

        col : None or string
            target columns if None then display all the metric
        '''
        df = self.results["baseline_results"]
        desc = pd.DataFrame(df.groupby(['dataset','out', 'inp'])[
            ['train_metric', 'val_metric', 'test_metric']].describe())
        desc = desc[
                [('train_metric', 'mean'), ('train_metric', 'std'), ('train_metric', 'min'), ('train_metric', 'max'),
                 ('val_metric', 'mean'), ('val_metric', 'std'), ('val_metric', 'min'), ('val_metric', 'max'),
                 ('test_metric', 'mean'), ('test_metric', 'std'), ('test_metric', 'min'), ('test_metric', 'max')]]
        
        # format to percentages
        func = lambda s: f"{s*100:.2f}%" 
        return desc.style.bar(align='mid').format(func) #desc.style.background_gradient(cmap=cmap)



##############################################  END  ###################################################
##############################################  END  ###################################################
##############################################  END  ###################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_samples', default=100, type=int)
    parser.add_argument('--dir', default='dataset/toybrains', type=str)
    parser.add_argument('-c', default = 'configs.lbl1cov1', type=str)
    parser.add_argument('-d', '--debug',  action='store_true')
    parser.add_argument('-j','--n_jobs', default=20, type=int)
    parser.add_argument('-s','--suffix', default = 'n', type=str)
    # parser.add_argument('--img_size', default=64, type=iny, help='the size (h x h) of the generated output images and labels')
    args = parser.parse_args()

    IMG_SIZE = 64 # 64 pixels x 64 pixels
    RANDOM_SEED = 42 if args.debug else None
    # create the output folder
    dataset = ToyBrainsData(out_dir=args.dir, 
                            config=args.c,
                            img_size=IMG_SIZE, debug=args.debug, 
                            seed=RANDOM_SEED)   
    # create the shapes dataset
    dataset.generate_dataset(n_samples=args.n_samples, 
                             n_jobs=args.n_jobs, 
                             outdir_suffix=args.suffix,
                            )