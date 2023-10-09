import os, shutil
from glob import glob
import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use("seaborn-v0_8-white") #'dark_background','seaborn-v0_8-muted', 'seaborn-v0_8-notebook',
from matplotlib.ticker import FormatStrFormatter
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

# add custom imports
from utils.dataset import generate_dataset
from utils.tabular import get_table_loader, run_lreg

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
        
    def bump_up_weight(self, idxs, amt=1):
        if isinstance(idxs, (int,float)): # and not a list
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
        # min_window = self.k-2 if self.k-2>2 else 2
        self._smooth_weights()
        # self._smooth_weights()
        assert len(self.weights)==self.k, f"len(weights={self.weights}) are not equal to len(states={self.states}).\
 Something failed when performing self._smooth_weights()"
        return self
        
    def _smooth_weights(self): 
        """Smooths the self.weights numpy array by taking the 
        average of its neighbouring values within a specified window.
        Args:
        - window (int): the window size for smoothing
        """
        # Pad the array with ones for the sliding window
        # for odd len arrays pad differently as opposed to even lenght arrays
        #  
        window=2 #TODO currently only works with 2 
        pad = (window//2-1, window//2)
        arr = np.pad(self.weights, pad, mode='edge')
        # Create a 2D array of sliding windows
        shape = arr.shape[:-1] + (arr.shape[-1]-window+1, window)
        strides = arr.strides + (arr.strides[-1],)
        windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        
        # Take the average of each sliding window to smooth the array
        self.weights = np.mean(windows, axis=1)
        # remove the paddings
        # windows  = windows[pad[0]-1:-pad[1]]
        return self
        
    def sample(self):
        probas = self.weights/self.weights.sum()
        return np.random.choice(self.states, p=probas).item()
    
    def reset_weights(self):
        self.weights = np.ones(self.k)
        
        
################################## Main Class ############################################

class ToyBrainsData:
    
    def __init__(self, out_dir="dataset/toybrains", 
                 img_size=64, seed=None,  
                 base_config="configs.lbl5cov3_base", 
                 tweak_config=None,
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
        if base_config[-3:]=='.py': base_config = base_config[:-3]
        base_config = importlib.import_module(base_config)
        assert hasattr(base_config, 'COVARS')
        self.COVARS = {cov: PROB_VAR(name=cov, **states) for cov, states in base_config.COVARS.items()}
        assert hasattr(base_config, 'RULES_COV_TO_GEN')
        # sanity checks
        for cov, rules in base_config.RULES_COV_TO_GEN.items():
            assert cov in self.COVARS.keys(), f"In the rules RULES_COV_TO_GEN, the covariate {cov} has not been previously defined in COVARS"
            for cov_state, updates in rules.items():
                if not isinstance(cov_state, tuple):
                    assert cov_state in self.COVARS[cov].states, f"In the rules RULES_COV_TO_GEN, the {cov}={cov_state} has not been previously defined in COVARS"
                else: # continuous states
                    for cov_state_i in cov_state:
                        assert cov_state_i in self.COVARS[cov].states, f"In the rules RULES_COV_TO_GEN, the {cov}={cov_state_i} has not been previously defined in COVARS"
                        
                
        self.RULES_COV_TO_GEN = base_config.RULES_COV_TO_GEN
        # Import update the tweaks too, if provided
        if (tweak_config is not None) and (tweak_config != 'None'):
            # if user provided '.py' in the config filename argument then remove it
            if tweak_config[-3:]=='.py': tweak_config = tweak_config[:-3]
            tweak_config = importlib.import_module(tweak_config)
            assert hasattr(tweak_config, 'RULES_COV_TO_GEN_TWEAKS')
            self.RULES_COV_TO_GEN_TWEAKS = tweak_config.RULES_COV_TO_GEN_TWEAKS
            self._update_config_tweaks()
            
        self._init_df()
        # store a dict of results tables (run.csv) on the dataset
        self.results = {"baseline_results":None,
                        "supervised_results":None,
                        "unsupervised_results":None,
                       }

##########################################  methods for config viz #############################################
    
    def _update_config_tweaks(self):
        for key, subkey, new_values in self.RULES_COV_TO_GEN_TWEAKS:
             self.RULES_COV_TO_GEN[key][subkey] = new_values
                

    def show_current_config(self, show_dag=True, 
                            show_attr_probas=True, 
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
        
        if show_dag:
            # draw return a graphviz `dot` object, which jupyter can render
            dot = self.draw_dag(edges)
            display(dot)
        
        if show_attr_probas:
            # cov_cols, attr_cols = zip(*edges)
            attr_cols = sorted(list(self.GENVARS.keys()))
            cov_cols = sorted(list(self.COVARS.keys()))
            subplot_nrows = len(cov_cols)
            subplot_ncols = len(attr_cols)
            fs=12
             # create figure
            f,axes = plt.subplots(subplot_nrows, subplot_ncols, 
                                  figsize=(3+1.5*subplot_ncols, 1.5*subplot_nrows),
                                  sharex="col", sharey="row")
            f.suptitle("Image attribute generation probabilities (cols) set for different covariates or labels (rows)", 
                       fontsize=fs+6)
            
            max_attr_len = int(np.array([attr.k for _,attr in self.GENVARS.items()]).max())
            
            # each row shows one covariate vs all attributes
            for i, axis_row in enumerate(axes):
                
                cov_name = cov_cols[i]
                cov = self.COVARS[cov_name]
                
                # interatively collect the weights of all attributes for each covariate state in a dataframe
                df = pd.DataFrame(columns=[cov_name]+attr_cols)
                if cov.states[0] in [True, False]:
                    df[cov_name] = df[cov_name].astype(bool)
                # hack: reducing time by only taking few states when there are more than 5 states
                cov_states = cov.states if len(cov.states)<=4 else  cov.states[::len(cov.states)//4]
                
                for cov_state in cov_states:
                    # only trigger the change in weights caused when covariate = cov_state
                    # do not set any other covariate state
                    self._reset_genvars()
                    self._adjust_genvar_dists({cov_name:cov_state}, self.RULES_COV_TO_GEN)
                    
                    df_temp = pd.DataFrame(index=range(max_attr_len)) 
                    for attr_name, attr in self.GENVARS.items():
                        k = list(range(attr.k))
                        df_temp.loc[k, attr_name] = attr.states
                        # normalize weights to get p-distribution 
                        attr_probas = attr.weights/attr.weights.sum() 
                        df_temp.loc[k, f"{attr_name}_probas"] = attr_probas
                    
                    # add the cov state to the whole df 
                    df_temp[cov_name] = cov_state
                    df = pd.concat([df, df_temp], ignore_index=True)
                
                # display(df)
                
                for j, ax in enumerate(axis_row):
                    
                    attr_name = attr_cols[j]
                    g = sns.lineplot(df, 
                                     hue=cov_name, 
                                     x=attr_name, y=f"{attr_name}_probas",  
                                     ax=ax, legend=(j==0))
                    
                    if j==0: 
                        # make 2 cols if there are many lagend labels
                        ncol=2 if len(ax.legend_.legendHandles)>3 else 1
                        sns.move_legend(g, loc="upper left", 
                                        bbox_to_anchor=(-1.5,1.), ncol=ncol,
                                        frameon=True, 
                                        title_fontproperties={'size':fs, 
                                                              'weight':'heavy'}) #
                    # # set xlabel and ylabel at the top and leftside of the plots resp.
                    ax.set_ylabel(None)
                    # # shorten the xtick labels if it is too long
                    ticklabels = ax.get_xticklabels()
                    if i==0: ax.set_title(attr_name, fontsize=fs+2)
                    if len(ticklabels)>0 and len(ticklabels[0].get_text())>6:
                        for k, lbl in enumerate(ticklabels):
                            if len(lbl.get_text())>4: 
                                ticklabels[k].set_text(lbl.get_text()[:4])
                        ax.set_xticks(ax.get_xticks(), labels=ticklabels) #, fontsize=fs-4)
            plt.ylim([0,1]) # probability
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
            dot = graphviz.Digraph(format='png', graph_attr={'rankdir': ''})
            # separate out the source nodes from the destination nodes
            src_nodes, dst_nodes = zip(*edges)
            src_nodes = sorted(list(set(src_nodes)))
            dst_nodes = sorted(list(set(dst_nodes)))
            # categorize all nodes (attrib vars) into groups for easy reading
            src_grps = sorted(list(set([node.split("_")[0] for node in src_nodes])))
            dst_grps = sorted(list(set([node.split("_")[0] for node in dst_nodes])))
                              
            src_grp_to_green_map = {grp:(60+i*100)%255 for i, grp in enumerate(src_grps)}
            dst_grp_to_green_map = {grp:(60+i*75)%255 for i, grp in enumerate(dst_grps)}
            
            # add all source nodes
            for node in src_nodes:
                grp = node.split("_")[0]
                red, green, blue, alpha = 30, src_grp_to_green_map[grp], 200, 100
                color_hex = f'#{red:x}{green:x}{blue:x}{alpha:x}'
                settings = {"shape": "ellipse", "group":grp, "tooltip":grp,
                            "style":"filled", "fillcolor":color_hex,
                            # "color":color_hex,"penwidth":"2"
                            }
                dot.node(node, node, settings)
                
            # add destination nodes
            for grp in dst_grps:
                # add each destination grp as a parent node and each sub category as child node
                red, green, blue, alpha = 200, dst_grp_to_green_map[grp], 30, 100
                color_hex = f'#{red:x}{green:x}{blue:x}{alpha:x}'
                
                settings = {"shape": "ellipse", "group":grp, "tooltip":grp,
                            "style":"filled", "fillcolor":color_hex
                            } 
                
                with dot.subgraph(name=f'cluster_{grp}') as dot_c:
                    dot_c.attr(label=grp, labelloc='b',
                               style="dashed")
                    for node in dst_nodes:
                        if node.split("_")[0] == grp:
                            dot_c.node(node, "_".join(node.split("_")[1:]), settings)
                
            for a, b in edges:
                # set the arrow color same as the color of the attrib variable
                grp = b.split("_")[0]
                red, green, blue, alpha = 200, dst_grp_to_green_map[grp], 30, 200
                color_hex = f'#{red:x}{green:x}{blue:x}{alpha:x}'
                dot.edge(a, b, _attributes={"color":color_hex, 
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
            
            
    def _reset_genvars(self):
        [gen_var.reset_weights() for _, gen_var in self.GENVARS.items()]


    def _get_rule_for_covstate(self, cov_state, cov_rules):
        '''Function handles user-convinience settings in config of RULES_COV_TO_GEN
        such as if user provides a tuple of values in the keys instead of a single value'''
        if cov_state in cov_rules.keys():
            return cov_rules[cov_state]
        else:
            for cov_rule in cov_rules.keys():
                if isinstance(cov_rule, tuple):
                    if cov_state in cov_rule:
                        return cov_rules[cov_rule]
            else:
                return None
            
    
    def _adjust_genvar_dists(self, covars, rules):
        """Configure the relationship between covariates and the generative attributes"""
        ### model `Covariates -> image attributes` distribution
        for cov_name, cov_state in covars.items():
            if (cov_name not in rules.keys()) or (
                self._get_rule_for_covstate(cov_state, rules[cov_name]) is None):
                if self.debug: print(f"[WARN] No rules have been defined for covariate = {cov_name} and state = {cov_state}")
                continue
            attr_rules = self._get_rule_for_covstate(cov_state, rules[cov_name])
            for attr, attr_rule in attr_rules.items():
                self.GENVARS[attr].bump_up_weight(**attr_rule)
        
        
    def generate_dataset_table(self, n_samples, outdir_suffix='n'):
        self._init_df()
        
        if self.debug:
            print(f"Generative parameter {' '*7}|{' '*7} States \n{'-'*60}")
            for name, var in self.GENVARS.items():
                print(f"{name} {' '*(25-len(name))} {var.states}")
            
        print("Sampling n={} toybrain image settings".format(n_samples))
        
        for subID in tqdm(range(n_samples)):
            # first reset all generative image attributes to have uniform distribution
            self._reset_genvars()
            # (1) sample the covariates and labels for this data point and influence the image generation probabilities s
            covars = {name: covar.sample() for name, covar in self.COVARS.items()}
            self._adjust_genvar_dists(covars, self.RULES_COV_TO_GEN)
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
                            CV=10, N_JOBS=10, 
                            random_seed = 42, debug = False):
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
        # create the directory to store the results
        results_out_dir = f"{self.OUT_DIR_SUF}/baseline_results" 
        shutil.rmtree(results_out_dir, ignore_errors=True)
        os.makedirs(results_out_dir)
        
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
                for cv_i in range(CV):
                    all_settings.append((lbl,fea,cv_i))
                    
        print(f'running a total of {len(all_settings)} different settings of\
[input features] x [output labels] x [cross validation] and saving result in {self.OUT_DIR_SUF}')
        
        # run each model fit in parallel
        with Parallel(n_jobs=N_JOBS) as parallel:
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
        self.viz_baseline_results(df_out)
        return df_out
    

    # run one using settings
    def _fit_baseline_model(
        self,
        label, feature, trial,
        df_csv_path,
        CV, results_out_dir,
        random_seed, debug, results_kwargs):
        '''
        run one baesline: label X feature X trial
        ''' 
        # split the dataset for training, validation, and test from raw dataset
        dataset = generate_dataset(df_csv_path, label, CV, trial, random_seed, debug)

        # load the dataset
        data = get_table_loader(dataset=dataset, label=label, data_type=feature, random_seed=random_seed)
        if debug: print(f'Inputs: {data[0].columns}')

        # run logistic regression and linear regression for tabular dataset
        results_dict, model_name, metric, modelpipe = run_lreg(data)

        if debug:
            print(f"Train metric: {results_dict['train_metric']:>8.4f} "
                  f"Validation metric: {results_dict['val_metric']:>8.4f} "
                  f"Test metric: {results_dict['test_metric']:>8.4f}")

        result = {
            "inp" : feature,
            "out" : label,
            "trial" : trial,
            "model" : model_name,
            "metric": metric,
            **results_dict,
            "model_config" : modelpipe,
        }

        result.update(results_kwargs)
        df = pd.DataFrame([result])
        
        df.to_csv(f"{results_out_dir}/run-bsl_lbl-{label}_inp-{feature}_{trial}-of-{CV}_{model_name}.csv", 
                  index=False)
        

    # visualization
    def viz_baseline_results(self, run_results):
        ''' vizualization output of baseline models
        '''
        if isinstance(run_results, str) and os.path.exists(run_results):
            RUN = pd.read_csv(run_results)
        elif isinstance(run_results, pd.DataFrame):
            RUN = run_results
        else:
            raise ValueError(f"{run_results} is neither a path to the results csv nor a pandas dataframe")

        viz_df = RUN.copy(deep=True)

        x = 'test_metric'
        y = 'out'
        hue = 'inp'
        hue_order = ['attr', 'cov', 'attr+cov']
        datasets = list(viz_df['dataset'].unique())
        num_rows = len(datasets)
        join = False

        # setup the figure properties
        sns.set(style='whitegrid', context='paper')
        fig, axes = plt.subplots(num_rows, 1,
                                 sharex=True, sharey=True,
                                 dpi=120, figsize=(7,5))

        if num_rows == 1: axes = [axes]
        plt.xlim([-0.1, 1.1])

        for ax, dataset in zip(axes, datasets):
            dfn = viz_df.query(f"dataset == '{dataset}'")

            # plotting details
            palette = sns.color_palette()
            dodge, scale, errwidth, capsize = 0.4, 0.4, 0.9, 0.08

            ax = sns.pointplot(y=y, x=x, 
                               hue=hue, hue_order=hue_order,
                               join=join, data=dfn, ax=ax,
                               errorbar='sd', errwidth=errwidth, capsize=capsize,
                               dodge=dodge, scale=scale, palette=palette)

            ax.legend_.remove()
            ax.set_title(f"{dataset}")
            ax.set_xlabel("")
            ax.set_ylabel("")

            # draw the chance line in the legend
            ax.axvline(x=0.5, label="chance", c='gray', ls='--', lw=1.5)

        # set custom x-axis tick positions and labels
        x_tick_positions = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        x_tick_labels = ['0', '20', '40', '60', '80', '100']

        for ax in axes:
            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels(x_tick_labels)

        for ax in axes[:-1]:
            ax.set_xlabel("")

        # add labels for the last subplot
        axes[-1].set_xlabel(f"{x} - Accuracy (%)")

        legend_handles = []

        legend_labels = ["Attributes (A)", "Covariates (C)", "A+C"]
        for i, label in enumerate(legend_labels):
            legend_handles.append(plt.Line2D([0], [0], marker='', linestyle='-', color=palette[i], label=label))

        # add the legend outside the subplots
        plt.legend(handles=legend_handles, loc='upper right', title='Inputs', fontsize=7, bbox_to_anchor=(1.0, 0.4))

        plt.suptitle("Baseline Analysis Plot")
        plt.tight_layout()
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
    parser.add_argument('-c', default = 'configs.lbl5cov3_base', type=str)
    parser.add_argument('--config_tweak', default = None, type=str)
    parser.add_argument('-d', '--debug',  action='store_true')
    parser.add_argument('--n_jobs', default=10, type=int)
    parser.add_argument('--suffix', default = 'n', type=str)
    # parser.add_argument('--img_size', default=64, type=iny, help='the size (h x h) of the generated output images and labels')
    args = parser.parse_args()

    IMG_SIZE = 64 # 64 pixels x 64 pixels
    RANDOM_SEED = 42 if args.debug else None
    # create the output folder
    dataset = ToyBrainsData(out_dir=args.dir, 
                            base_config=args.c,
                            tweak_config=args.config_tweak,
                            img_size=IMG_SIZE, debug=args.debug, 
                            seed=RANDOM_SEED)   
    # create the shapes dataset
    dataset.generate_dataset(n_samples=args.n_samples, 
                             n_jobs=args.n_jobs, 
                             outdir_suffix=args.suffix,
                            )