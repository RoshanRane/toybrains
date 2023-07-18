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
import os, shutil
from joblib import Parallel, delayed  
from copy import copy, deepcopy
import itertools
from tqdm import tqdm
import argparse
from config import PROB_VAR, COVARS, RULES_COV_TO_GEN, loop_update_rules
import importlib

from helper.viz_helpers import plot_col_dists

class ToyBrainsData:
    
    def __init__(self, out_dir="./shapes/", img_size=64, seed=None, debug=False, njobs=1, config=None):
        
        self.I = img_size
        self.OUT_DIR = out_dir
        self.IMGS_DIR = f'{self.OUT_DIR}/images/'
        self.LBLS_DIR = f'{self.OUT_DIR}/masks/'
        self.njobs = njobs
        shutil.rmtree(self.OUT_DIR, ignore_errors=True)
        os.makedirs(self.OUT_DIR, exist_ok=True)
        os.makedirs(self.IMGS_DIR, exist_ok=True)
        os.makedirs(self.LBLS_DIR, exist_ok=True)
        self.debug = debug
        self.config = config

        # forcefully set a random seed
        if self.debug: seed = 42
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # the center of the image for drawing reference
        self.ctr = (self.I/2, self.I/2) #np.random.randint(self.I/2-2,self.I/2+2, size=2)
        
        # initialize all the generative properties for the images and the labels and covariates
        self._setup_genvars_covars()
        
        
    def _setup_genvars_covars(self):
    
        # (1) define all the generative properties for the images
        self.GENVARS = {
            # 1. brain_vol created as a ellipse with a minor and major radius
            # ranging between 1633 to 2261 [(S/2-12)*(S/2-6) to (S/2-8)*(S/2-2)]
            'brain_vol-radminor': PROB_VAR('brain_vol-rad-minor', 
                                       np.arange(self.I/2 - 12, self.I/2 - 8 + 1, dtype=int)),
            'brain_vol-radmajor': PROB_VAR('brain_vol-rad-major', 
                                       np.arange(self.I/2 - 6, self.I/2 - 2 + 1, dtype=int)),                        
            # 2. brain_thick: the thickness of the blue border around the brain ranging between 1 to 4
            'brain_thick':        PROB_VAR('brain_thick', np.arange(1,4+1, dtype=int)), 
            # 3. the intensity or brightness of the brain region ranging between 'greyness1' (210) to 'greyness5' (170)
            'brain_int':          PROB_VAR('brain_int', [210,200,190,180,170]), 
            # 4. the intensity or brightness of the ventricles and brain borders ranging between 'blueness1' to 'blueness3' 
            'border_int':         PROB_VAR('border_int', ['0-mediumslateblue','1-slateblue','2-darkslateblue','3-darkblue']),
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
                f'{shape_pos}_int': PROB_VAR(f'{shape_pos}_int', 
                                            ['0-indianred','1-salmon','2-lightsalmon',
                                             '3-palegoldenrod','4-lightgreen','5-darkgreen']), 
                # the radius of the circle inside which the regular polygon is drawn
                f'{shape_pos}_vol-rad': PROB_VAR(f'{shape_pos}_volrad', 
                                            np.arange(2,5+1, dtype=int)),  
                # TODO # rot = np.random.randint(0,360)
            })
        
        # (2) Import the configured Covariates and its rules
        self.COVARS = COVARS
        
        if self.config is not None:
            pkg = importlib.import_module(f'setting.{self.config}')
            cfg = pkg.c
            loop_update_rules(RULES_COV_TO_GEN, cfg)
        
        self.RULES_COV_TO_GEN = RULES_COV_TO_GEN

        
    ### INIT METHODS
    def init_df(self):
        # Initialize a dataframe to store all data
        self.df =  pd.DataFrame()
        self.df.index.name = "subjectID"   
        for name,_ in self.GENVARS.items(): # TODO remove
            if '-rad' in name:
                self.df['_gen_' + name] = np.nan
            else:
                self.df['gen_' + name] = np.nan
                
        for name, var in self.COVARS.items():
            self.df[name] = np.nan
            
    def reset_genvars(self):
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
            
    
    def adjust_genvar_dists(self, covars, rules):
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
        
        
    def generate_dataset(self, n_samples):
        """Creates toy dataset and save to disk."""
        self.init_df()
            
        if self.debug:
            print(f"Generative parameter {' '*7}|{' '*7} States \n{'-'*60}")
            for name, var in self.GENVARS.items():
                print(f"{name} {' '*(25-len(name))} {var.states}")
            
        print("Generating {} synthetic toy brain images:".format(n_samples))

        ## TODO parallize data gen
        # result = Parallel(n_jobs=self.njobs)(delayed(self._gen_image)(subID) for subID in tqdm(range(n_samples)))
        for subID in tqdm(range(n_samples)):
            
            # first reset all generative image attributes to have uniform distribution
            self.reset_genvars()
            # (1) sample the covariates and labels for this data point and influence the image generation probabilities s
            covars = {name: covar.sample()  for name, covar in self.COVARS.items()}
            self.adjust_genvar_dists(covars, self.RULES_COV_TO_GEN)
            
            # Create a new image of size 64x64 with a black background and a draw object on it
            image = Image.new('RGB',(self.I,self.I),(0,0,0))
            draw = ImageDraw.Draw(image)

            # (3) sample the image attributes conditional on the sampled labels and covariates 
            genvars = {}
            for gen_var_name, gen_var in self.GENVARS.items():
                if ('brain_' in gen_var_name) or (gen_var_name=='border_int'):
                    genvars.update({gen_var_name: gen_var.sample()})
            
            # (4) Draw the brain 
            # (4a) Draw an outer ellipse of the image
            x0,y0 = (self.ctr[0]-genvars['brain_vol-radminor'],
                     self.ctr[1]-genvars['brain_vol-radmajor'])
            x1,y1 = (self.ctr[0]+genvars['brain_vol-radminor']-1,
                     self.ctr[1]+genvars['brain_vol-radmajor']-1)

            draw.ellipse((x0,y0,x1,y1), 
                         fill=self.get_color_val(genvars['brain_int']), 
                         width=genvars['brain_thick'],
                         outline=self.get_color_val(genvars['border_int'])
                        )
            # save the brain mask
            brain_mask = (np.array(image).sum(axis=-1) > 0) #TODO save it

            # (4b) Draw ventricles as 2 touching arcs 
            for gen_var_name, gen_var in self.GENVARS.items():
                if ('vent_' in gen_var_name):
                    genvars.update({gen_var_name: gen_var.sample()})

            xy_l = (self.I*.3, self.I*.3, self.I*.5, self.I*.5)
            xy_r = (self.I*.5, self.I*.3, self.I*.7, self.I*.5)
            draw.arc(xy_l, start=+310, end=+90, 
                     fill=self.get_color_val(genvars['border_int']), 
                     width=genvars['vent_thick'])
            draw.arc(xy_r, start=-290, end=-110, 
                     fill=self.get_color_val(genvars['border_int']), 
                     width=genvars['vent_thick'])

            # (4c) draw 5 shapes (triangle, square, pentagon, hexagon, ..)
            # with different size, color and rotations
            for shape_pos, (x,y) in self.SHAPE_POS.items():
                
                for gen_var_name, gen_var in self.GENVARS.items():
                    if (shape_pos in gen_var_name):
                        genvars.update({gen_var_name: gen_var.sample()})
                
                draw.regular_polygon((x,y,genvars[f'{shape_pos}_vol-rad']), 
                                     n_sides=genvars[f'{shape_pos}_curv'], 
                                     rotation=np.random.randint(0,360),
                                     fill=self.get_color_val(genvars[f'{shape_pos}_int']), 
                                     outline=self.get_color_val(genvars['border_int']))
            # (4d) save the image
            image.save(f"{self.IMGS_DIR}{subID:05}.jpg")
            
            # (5) store the sampled covariates and labels  
            for k,v in covars.items():
                self.df.at[f'{subID:05}', k] = v
            # combine the gen params 'brain_vol-radmajor' and 'brain_vol-radminor' into one 'brain_vol'
            genvars.update({'brain_vol': math.pi*genvars['brain_vol-radmajor']*genvars['brain_vol-radminor'] })
            # calculate the the volume of all regular_polygon shapes
            for shape_pos in self.SHAPE_POS.keys():
                genvars.update({
                    f'{shape_pos}_vol': 
                    self.area_of_regular_polygon(
                        n=genvars[f'{shape_pos}_curv'], r=genvars[f'{shape_pos}_vol-rad'])})# TODO
                
            # store the generative properties with a prefix 'gen_'
            for k,v in genvars.items():
                if '-rad' in k: # radius is stored as the secondary gen variable 
                    self.df.at[f'{subID:05}', '_gen_'+k] = v
                else:
                    self.df.at[f'{subID:05}', 'gen_'+k] = v
            
            ##TODO store the weights or generative probabilities of all paramaters too
            # if debug:  print(f"variable {self.name} with states {self.states} and weights={self.weights}")
            
            # save the data csv every 1000 samples
            if subID//1000 == 0:
                self.df.to_csv(f"{self.OUT_DIR}/toybrains_n{n_samples}.csv")      
            
        # format the subject IDs same as the filename
        self.df.to_csv(f"{self.OUT_DIR}/toybrains_n{n_samples}.csv")
        
        
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


    ### PLOT METHODS
    def show_current_config(self, show_attr_probas=True):
        attr_cols = sorted(list(self.GENVARS.keys()))
        cov_cols = sorted(list(self.COVARS.keys()))
        subplot_nrows = len(attr_cols)
        subplot_ncols = len(cov_cols)
        fs=12
         # create 
        f,axes = plt.subplots(subplot_nrows, subplot_ncols, 
                              figsize=(2+2*subplot_ncols, 2*subplot_nrows),
                              sharex='row', sharey=True, constrained_layout=True)
        f.suptitle("Probability distribution of image attributes (rows) vs different conditions of the covariates/labels (cols):", 
                   fontsize=fs+6)
        plt.ylim([0,1]) # probability
        
        
        if show_attr_probas:
            
            for i, axis_row in enumerate(axes):
                attr_name = attr_cols[i]
                attr = self.GENVARS[attr_name]
                
                for j, ax in enumerate(axis_row):
                    
                    cov_name = cov_cols[j]
                    cov = self.COVARS[cov_name]
                        
                    df_temp = pd.DataFrame()
                    # collect the weights for all states of the covariate in df_temp
                    for cov_state in cov.states: ## TODO make this efficient for age - only sample 3 if 3 groups are defined in the rules
                        
                        dfi_temp = pd.DataFrame({attr_name: attr.states, cov_name: cov_state})
                        self.reset_genvars()
                        self.adjust_genvar_dists({cov_name:cov_state}, self.RULES_COV_TO_GEN)
                        # normalize weights to get p-distribution
                        attr_probas = attr.weights/attr.weights.sum() #+ 0.1*np.random.rand(len(attr.weights)) # add some jitter noise for the plot visibility
                        dfi_temp["attr_probas"] = attr_probas
                        df_temp = pd.concat([df_temp, dfi_temp], ignore_index=True)
                    
                    g = sns.lineplot(df_temp, x=attr_name, y="attr_probas", hue=cov_name, 
                                     ax=ax, legend=(i==0), alpha=0.9)
                    if i==0: 
                        sns.move_legend(g, "upper center", bbox_to_anchor=(0.5,1.5), 
                                        alignment='center', ncols=2,
                                        title_fontproperties={'size':fs}) #, 'weight':'heavy'
                    # set xlabel and ylabel at the top and leftside of the plots resp.
                    ax.set_ylabel(attr_name, fontsize=fs) if j==0 else ax.set_ylabel(None)
                    ax.set_xlabel(None)
                    # shorten the xtick labels if it is too long
                    ticklabels = ax.get_xticklabels()
                    if len(ticklabels[0].get_text())>6:
                        for k, lbl in enumerate(ticklabels):
                            if len(lbl.get_text())>4: 
                                ticklabels[k].set_text(lbl.get_text()[:4])
                        ax.set_xticks(ax.get_xticks(), labels=ticklabels) #, fontsize=fs-4)
            plt.show()
    
    
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_samples', default=100, type=int)
    parser.add_argument('--dir', default='toybrains', type=str)
    parser.add_argument('-d', '--debug',  action='store_true')
    parser.add_argument('-c', default = None, type=str)
    # parser.add_argument('--img_size', default=64, type=iny, help='the size (h x h) of the generated output images and labels')
    args = parser.parse_args()
    
    IMG_SIZE = 64 # 64 pixels x 64 pixels
    RANDOM_SEED = 42 if args.debug else None
    # create the output folder
    dataset = ToyBrainsData(out_dir=args.dir, config=args.c,
                            img_size=IMG_SIZE, debug=args.debug, 
                            seed=RANDOM_SEED)   
    # create the shapes dataset
    dataset.generate_dataset(n_samples=args.n_samples)