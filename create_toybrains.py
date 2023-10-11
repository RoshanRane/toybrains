import os, shutil
from glob import glob
import numpy as np
import random
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
plt.style.use("seaborn-v0_8-white") #'dark_background','seaborn-v0_8-muted', 'seaborn-v0_8-notebook',
import seaborn as sns
import math
from PIL import Image, ImageDraw
from colordict import ColorDict
from joblib import Parallel, delayed  
from copy import copy, deepcopy
import itertools
from tqdm.notebook import tqdm
import argparse
import importlib
from datetime import datetime
from tabulate import tabulate

# add custom imports
from utils.dataset import split_dataset
from utils.tabular import run_lreg

# sys.path.insert(0, "../")
# sns.set_theme(style="ticks", palette="pastel")

import graphviz
from causalgraphicalmodels import CausalGraphicalModel

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
                 config="configs.lbl1cov1", 
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
        
        # Import the covariates and the base configured in covariates-to-image-attribute relationships
        # if user provided '.py' in the config filename argument then remove it
        if config[-3:]=='.py': config = config[:-3]
        config = importlib.import_module(config)
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
            
        self._init_df()
        # store a dict of results tables (run.csv) on the dataset
        self.results = {"baseline_results":None,
                        "supervised_results":None,
                        "unsupervised_results":None,
                       }

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
                

    def show_current_config(self, 
                            show_dag_probas=True, 
                            return_causal_graph=False):
        """
        show_attr_probas = True : shows how the sampling probability distribution of the different
                                  image attributes change for each covariate/label state. This feature
                                   is used to verify that all the intended weight changes were applied.
        """
        # compute the nodes and edges of the Causal Graphical Model
        nodes, edges = set(), set()
        for c, c_states in self.RULES_COV_TO_GEN.items():
            nodes.add(c)
            for _, ats in c_states.items():
                for at in ats.keys():
                    nodes.add(at)
                    edges.add((c,at))
                    
        nodes = sorted(list(nodes))
        edges = sorted(list(edges))
        
        # draw return a graphviz `dot` object, which jupyter can render
        dot = self.draw_dag(edges)
        display(dot)
        
        if show_dag_probas:
            
            # convert to one-to-many dict of source node to destination nodes
            src_to_dst_map = {}
            for src, dst in edges:
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
        
        
        if return_causal_graph:
            return CausalGraphicalModel(nodes=nodes, edges=edges)
        else:
            return nodes, edges
    
    
    def draw_dag(self, edges):
            """
            dot file representation of the CGM.
            """
            dot = graphviz.Digraph(graph_attr={'rankdir': ''})
            # separate out the source nodes from the destination nodes
            src_nodes, dst_nodes = zip(*edges)
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
                            "tooltip":grp, "image":"test.png",
                            "style":"filled", "fillcolor": get_color_hex(grp), 
                            } 
                
                with dot.subgraph(name=f'cluster_dst') as dot_dst:
                    dot_dst.attr(rank="dst", style="invis")
                    with dot_dst.subgraph(name=f'cluster_dst_{grp}') as dot_c:
                        dot_c.attr(label=grp, labelloc='b', style="dashed")
                        for node in dst_nodes:
                            if node.split("_")[0] == grp:
                                dot_c.node(node, "_".join(node.split("_")[1:]), **settings)
                
            for a, b in edges:
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
        
        
    ### INIT METHODS
    def _init_df(self):
        # Initialize a table to store all dataset attributes
        columns = sorted(list(self.COVARS.keys()))
        for name,_ in self.GENVARS.items(): 
            if '-rad' in name:
                columns.append('_gen_' + name)
            else:
                columns.append('gen_' + name)
                
        self.df =  pd.DataFrame(columns=columns)
        self.df.index.name = "subjectID"   
            
            
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
        
        
    def generate_dataset_table(self, n_samples, outdir_suffix='n'):
        self._init_df()
        
        if self.debug:
            print(f"Generative parameter {' '*7}|{' '*7} States \n{'-'*60}")
            for name, var in self.GENVARS.items():
                print(f"{name} {' '*(25-len(name))} {var.states}")
            
        print("Sampling n={} toybrain image settings".format(n_samples))
        
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
                
            # (3) store the covars and then the generative attributes with a prefix 'gen_'
            for k,v in covars.items():
                self.df.at[f'{subID:05}', k] = v
            # store the generative properties with a prefix 'gen_'
            for k,v in genvars.items():
                if '-rad' in k: # radius is stored as a secondary gen variable 
                    self.df.at[f'{subID:05}', '_gen_'+k] = v
                else:
                    self.df.at[f'{subID:05}', 'gen_'+k] = v
        
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
            if 'gen_' in key:
                # remove the prefixes
                key = key.split('gen_')[1] 
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
        self.generate_dataset_table(n_samples)
        self.generate_dataset_images(n_jobs=n_jobs, 
                                     outdir_suffix=outdir_suffix)
            
            
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
                            CV=10, n_jobs=10, 
                            random_seed=42, debug=False):
        ''' run linear regression or logistic regression to estimate the expected prediction performance
        for a given dataset. 
        Fits [input features] X [output labels] X [cross validation] models where,
            input features can be either:
                i) image attributes (attr), 
                ii) other covariates in the dataset (cov)
                iii) image attributes and other covariates together (attr+cov)
            output labels are all covariates in the dataset
            model can be:
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
        assert len(self.df)>0, "first generate the dataset table using self.generate_dataset_table() method." 
        start_time = datetime.now()
        df_csv_path = glob(f"{self.OUT_DIR_SUF}/toybrains_*.csv")[0]
        features = ['attr', 'attr+cov', 'cov']
        labels = self.df.filter(regex = '^lbl').columns.tolist() + self.df.filter(regex = '^cov').columns.tolist()
        n_covs = len(self.df.filter(regex = '^cov').columns)
        # create the directory to store the results
        results_out_dir = f"{self.OUT_DIR_SUF}/baseline_results" 
        shutil.rmtree(results_out_dir, ignore_errors=True)
        os.makedirs(results_out_dir)
        if debug: n_jobs=1
        
        for_result_out = {
            "dataset" : self.OUT_DIR_SUF,
            "type" : "baseline",
            "n_samples" : len(self.df),
            "CV" : CV,
        }

        # generate the different settings of [input features] X [output labels] X [cross validation]
        all_settings = []
        for lbl in labels:
            for fea in features:
                # if there are no input features then skip
                # if there are no covariates in the features when setting is "attr+cov" then skip
                if not ((lbl.split('_')[0]=='cov') and (n_covs==1) and (fea in ['attr+cov','cov'])):
                    for cv_i in range(CV):
                        all_settings.append((lbl,fea,cv_i))

        print(f'running a total of {len(all_settings)} different settings of \
[input] x [output] x [CV] and saving result in {self.OUT_DIR_SUF}')
        
        # run each model fit in parallel
        with Parallel(n_jobs=n_jobs) as parallel:
            parallel(
                delayed(
                    self._fit_baseline_model)(
                        label=label, feature=feature, trial=trial,
                        df_csv_path=df_csv_path,
                        CV=CV,
                        results_out_dir=results_out_dir,
                        random_seed=random_seed, debug=debug, 
                        results_kwargs=for_result_out
                    ) for label, feature, trial in tqdm(all_settings))

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
        # self.viz_baseline_results(df_out)
        return df_out
    

    # run one using settings
    def _fit_baseline_model(
        self,
        label, feature, trial,
        df_csv_path,
        CV, results_out_dir,
        random_seed, debug, results_kwargs):
        '''
        run one baseline: label X feature X trial
        ''' 
        data = []
        # split the dataset for training, validation, and test from raw dataset
        for data_split in split_dataset(df_csv_path, label, 
                                        CV, trial, 
                                        random_seed, 
                                        debug):
            # get the input and output
            X,y = self._get_X_y(data_split, label=label, data_type=feature)
            data.append([X,y])
        # display(data)
        X = data[0][0].columns.tolist()
        if debug: 
            print(f'Input features: {X}')
            print(f'Output label  : {label}')

        # run logistic regression and linear regression for tabular dataset
        results_dict, model_config = run_lreg(data)

        if debug:
            print(f"Train metric: {results_dict['train_metric']:>8.4f} "
                  f"Validation metric: {results_dict['val_metric']:>8.4f} "
                  f"Test metric: {results_dict['test_metric']:>8.4f}")

        result = {
            "inp" : feature,
            "out" : label,
            "trial" : trial,
            **results_dict,
            **model_config,
            **results_kwargs
        }
        
        # save the results as a csv file
        pd.DataFrame([result]).to_csv(
            f"{results_out_dir}/run-bsl_lbl-{label}_inp-{feature}_{trial}-of-{CV}_{model_config['model']}.csv", 
                  index=False)

        
    def _get_X_y(self, df, label, data_type='attr'):
        '''get tabular data using data_type criteria'''

        assert data_type in ['attr',  'cov', 'attr+cov', 'cov+attr'], \
    "data type should be one of ['attr',  'cov', 'attr+cov', 'cov+attr']"
        assert label in df.columns, f"label {label} should be in dataframe"

        # set the target label
        target = list(df['label'])

        # set the data using data_type
        columns = []
        if 'attr' in data_type:
            new_columns = df.filter(regex='^gen').columns.tolist()
            columns += new_columns
        if 'cov' in data_type:
            new_columns = df.filter(regex='^cov').columns.tolist()
            columns += new_columns
            # if label is in the input features then remove 
            if label in columns: columns.remove(label)

        data = df[columns]

        return data, target

    # visualization
    def viz_baseline_results(self, run_results):
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
            
        x = 'test_metric'
        y = 'out'
        hue = 'inp'
        hue_order = ['attr', 'attr+cov', 'cov']
        num_rows = viz_df[y].nunique()
        datasets = list(viz_df['dataset'].unique())
        num_subplots = len(datasets)

        # setup the figure properties
        sns.set(style='whitegrid', context='paper')
        fig, axes = plt.subplots(num_subplots, 1,
                                 sharex=True, sharey=True,
                                 constrained_layout=True,
                                 figsize=(5,(num_rows)*num_subplots))
        fs=12
        axes = axes.ravel() if num_subplots>1 else [axes] 

        # set custom x-axis tick positions and labels
        x_tick_positions = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        x_tick_labels = ['0%', '20%', '40%', '60%', '80%', '100%']
        
        for i, (ax, dataset) in enumerate(zip(axes, datasets)):
            dfn = viz_df.query(f"dataset == '{dataset}'")

            # plotting details
            palette = sns.color_palette()

            ax = sns.barplot(y=y, x=x, data=dfn, 
                            ax=ax,
                            hue=hue, hue_order=hue_order, 
                            errorbar=('ci', 95))

            ax.set_title(f"dataset = {dataset}", fontsize=fs)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels(x_tick_labels)         
            
            # adjust the legend
            if i==0:
                handles,labels = ax.get_legend_handles_labels()
                legend_label_map = {'attr':"Attributes", 'cov':"Covariates", 'attr+cov':"Attributes + Covariates"}
                labels = [legend_label_map[l] for l in labels]
                ax.legend(handles, labels, 
                          loc='upper right', bbox_to_anchor=(1.6,1),
                          fontsize=fs-4,
                          frameon=True, title='Input features')
            else: ax.get_legend().remove()
            

        # add labels for the last subplot
        axes[-1].set_xlabel(
            r"{}  $R2$ or Pseudo-$R2$ (%)".format(x.replace('_',' ')), 
            fontsize=fs)
        

        plt.suptitle("Baseline Analysis Plot", fontsize=fs+2)
        plt.show()
    # plt.savefig("figures/results_bl.pdf", bbox_inches='tight')

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
    parser.add_argument('--n_jobs', default=10, type=int)
    parser.add_argument('--suffix', default = 'n', type=str)
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