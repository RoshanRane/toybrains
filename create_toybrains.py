import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['axes.facecolor']='black'
plt.rcParams['savefig.facecolor']='black'

from PIL import Image, ImageDraw
from colordict import ColorDict
import math

import os
import shutil
from joblib import Parallel, delayed  
from tqdm import tqdm
import argparse


class _GEN_VAR:
    def __init__(self, name, states):
        self.name = name
        self.states = np.array(states)
        self.k = len(states)
        self.reset_weights()
        
    def bump_up_weight(self, idxs, amt=1):
        if not isinstance(idxs, (list,tuple)):
            idxs = [idxs]
        for i in idxs:
            try:
                self.weights[i] += amt
            except IndexError as e:
                print(f"\n[IndexError] index={i} is out-of-bound for variable '{self.name}' \
    with n={self.k} states {self.states} and weights {self.weights}")
                raise e
        # min_window = self.k-2 if self.k-2>2 else 2
        self._smooth_weights()
        assert len(self.weights)==self.k, f"len(weights={self.weights}) are not equal to len(states={self.states}).\
 Something failed when performing self._smooth_weights()"
        return self
        
    def _smooth_weights(self, window=2):
        """Smooths the self.weights numpy array by taking the 
        average of its neighbouring values within a specified window.
        Args:
        - window (int): the window size for smoothing
        """
        # Pad the array with ones for the sliding window
        # for odd len arrays pad differently as opposed to even lenght arrays
        #  
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
    
    
class ToyBrainsData:
    
    def __init__(self, out_dir="./shapes/", img_size=64, seed=None, debug=False, njobs=1):
        
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
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # the center of the image for drawing reference
        self.ctr = (self.I/2, self.I/2) #np.random.randint(self.I/2-2,self.I/2+2, size=2)
        
        # define all the generative properties for the images
        self.GENVARS = {
            # 1. brain_vol created as a ellipse with a minor and major radius
            # ranging between 1633 to 2261 [(S/2-12)*(S/2-6) to (S/2-8)*(S/2-2)]
            'brain_vol-radminor':_GEN_VAR('brain_vol-rad-minor', 
                                       np.arange(self.I/2 - 12, self.I/2 - 8 + 1, dtype=int)),
            'brain_vol-radmajor':_GEN_VAR('brain_vol-rad-major', 
                                       np.arange(self.I/2 - 6, self.I/2 - 2 + 1, dtype=int)),                        
            # 2. brain_thick: the thickness of the blue border around the brain ranging between 1 to 4
            'brain_thick':_GEN_VAR('brain_thick', np.arange(1,4+1, dtype=int)), 
            # 3. the intensity or brightness of the brain region ranging between 'greyness1' (210) to 'greyness5' (170)
            'brain_int':_GEN_VAR('brain_int', [210,200,190,180,170]), 
            # 4. the intensity or brightness of the ventricles and brain borders ranging between 'blueness1' to 'blueness3' 
            'border_int':_GEN_VAR('border_int', ['0-mediumslateblue','1-slateblue','2-darkslateblue','3-darkblue']),
            # ventricle (the 2 touching arcs in the center) thickness ranging between 1 to 4
            'vent_thick':_GEN_VAR('vent_thick', np.arange(1,4+1, dtype=int)),
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
                f'{shape_pos}_curv': _GEN_VAR(f'{shape_pos}_curv',
                                              np.arange(3,12, dtype=int)), 
                # color of the regular polygon from shades of green to shades of red
                f'{shape_pos}_int': _GEN_VAR(f'{shape_pos}_int', 
                                            ['0-indianred','1-salmon','2-lightsalmon',
                                             '3-palegoldenrod','4-lightgreen','5-darkgreen']), 
                # the radius of the circle inside which the regular polygon is drawn
                f'{shape_pos}_vol-rad': _GEN_VAR(f'{shape_pos}_volrad', 
                                            np.arange(2,5+1, dtype=int)),  
                # TODO # rot = np.random.randint(0,360)
            })
        
        # define all the covariates
        self.COVARS = {
            'cov_sex' : ['Male', 'Female'],
            'cov_site': ['siteA', 'siteB', 'siteC', 'siteD'],
            'cov_age' : np.arange(20,50+1),
        }
        
        # define all labels
        self.LABELS = {
            'lblbin_stop-smidl'        : [True, False],
            'lblbin_stop-smidl-bvol'   : [True, False],
            'lblbin_stop-smidl-vthick' : [True, False],
            'lblbin_bvol-vthick'       : [True, False],
        }
        
        self.LBL_TO_GEN_MAP = {
            # color tends towards red, curv to lower, volume to lower
            '-stop':  [('shape-top_curv', (1,2,3)), ('shape-top_int', (1,2)), ('shape-top_vol-rad' ,(1,2))],
            '-smidl': [('shape-midl_curv',(1,2,3)), ('shape-midl_int',(1,2)), ('shape-midl_vol-rad',(1,2))],
            '-smidr': [('shape-midr_curv',(1,2,3)), ('shape-midr_int',(1,2)), ('shape-midr_vol-rad',(1,2))],
            '-sbotr': [('shape-botr_curv',(1,2,3)), ('shape-botr_int',(1,2)), ('shape-botr_vol-rad',(1,2))],
            '-sbotl': [('shape-botl_curv',(1,2,3)), ('shape-botl_int',(1,2)), ('shape-botl_vol-rad',(1,2))],
            # brain volume reduces
            '-bvol' : [('brain_vol-radminor',1), ('brain_vol-radminor',1)],
            # ventricle thickness increases
            '-vthick' : [('vent_thick',1)],
        }
#         # create label by booling functions
#         y = np.zeros((n_sample, 15))
#         y[:, 0] = ((1 - concept[:, 0] * concept[:, 2]) + concept[:, 3]) > 0
#         y[:, 1] = concept[:, 1] + (concept[:, 2] * concept[:, 3])
#         y[:, 2] = (concept[:, 3] * concept[:, 4]) + (concept[:, 1] * concept[:, 2])
#         y[:, 3] = np.bitwise_xor(concept[:, 0], concept[:, 1])
#         y[:, 4] = concept[:, 1] + concept[:, 4]
#         y[:, 5] = (1 - (concept[:, 0] + concept[:, 3] + concept[:, 4])) > 0
#         y[:, 6] = np.bitwise_xor(concept[:, 1] * concept[:, 2], concept[:, 4])
#         y[:, 7] = concept[:, 0] * concept[:, 4] + concept[:, 1]
#         y[:, 8] = concept[:, 2]
#         y[:, 9] = np.bitwise_xor(concept[:, 0] + concept[:, 1], concept[:, 3])
#         y[:, 10] = (1 - (concept[:, 2] + concept[:, 4])) > 0
#         y[:, 11] = concept[:, 0] + concept[:, 3] + concept[:, 4]
#         y[:, 12] = np.bitwise_xor(concept[:, 1], concept[:, 2])
#         y[:, 13] = (1 - (concept[:, 0] * concept[:, 4] + concept[:, 3])) > 0
#         y[:, 14] = np.bitwise_xor(concept[:, 4], concept[:, 3])

        # x = np.zeros((n_sample, width, height, 3))
    
    
    def generate_dataset(self, n_samples):
        
        """Creates toy dataset and save to disk."""
        # Initialize a dataframe to store all data
        self.df =  pd.DataFrame()
        self.df.index.name = "subjectID"   
        for name,_ in self.GENVARS.items(): # TODO remove
            if '-rad' in name:
                self.df['_gen_' + name] = np.nan
            else:
                self.df['gen_' + name] = np.nan
        for name,_ in self.COVARS.items():
            self.df[name] = np.nan
            
        if self.debug:
            print(f"Generative parameter {' '*7}|{' '*7} States \n{'-'*60}")
            for name, var in self.GENVARS.items():
                print(f"{name} {' '*(25-len(name))} {var.states}")
            
            
        print("Generating {} synthetic 'toy brain' images:".format(n_samples))

        for subID in tqdm(range(n_samples)):
            
            ## TODO parallize data gen
            # result = Parallel(n_jobs=self.njobs)(delayed(self._gen_image)(subID) for subID in tqdm(range(n_samples)))
            
            genvars = {}
            # reset the distribution of the generative properties of the image generations
            [gen_var.reset_weights() for _, gen_var in self.GENVARS.items()]
                
            # (1) sample the convariates / confounders for this data point
            covars = {covar: np.random.choice(vals).item() for covar, vals in self.COVARS.items()}
            
            # if male increase the changes of sampling a higher brain volume by 2
            if covars['cov_sex']=='Male':
                for var in ['brain_vol-radmajor', 'brain_vol-radminor']:
                    self.GENVARS[var].bump_up_weight(-1, amt=3)
                    self.GENVARS[var].bump_up_weight(-2, amt=2) 
            # if female increase the changes of sampling a lower brain volume by 2
            else:
                for var in ['brain_vol-radmajor', 'brain_vol-radminor']:
                    self.GENVARS[var].bump_up_weight(0, amt=3)
                    self.GENVARS[var].bump_up_weight(1, amt=2) 
            
            if covars['cov_site']=='siteA':
                self.GENVARS['brain_int'].bump_up_weight(0, amt=4) 
                self.GENVARS['border_int'].bump_up_weight(0, amt=2)
            elif covars['cov_site']=='siteB':
                self.GENVARS['brain_int'].bump_up_weight((1,2), amt=3)
                self.GENVARS['border_int'].bump_up_weight(1, amt=2)
            elif covars['cov_site']=='siteC':
                self.GENVARS['brain_int'].bump_up_weight((2,3), amt=3)
                self.GENVARS['border_int'].bump_up_weight(2, amt=2)
            else:# covars['site']=='siteD':
                self.GENVARS['brain_int'].bump_up_weight(4, amt=4) 
                self.GENVARS['border_int'].bump_up_weight(1, amt=2)
            
            if 20<=covars['cov_age']<=30:
                for var in ['brain_vol-radmajor', 'brain_vol-radminor']:
                    self.GENVARS[var].bump_up_weight((-1,-2), amt=3)
                self.GENVARS['vent_thick'].bump_up_weight(3, amt=3)
            elif 30<covars['cov_age']<=40:
                for var in ['brain_vol-radmajor', 'brain_vol-radminor']:
                    self.GENVARS[var].bump_up_weight((-3,-4), amt=3)
                self.GENVARS['vent_thick'].bump_up_weight(2, amt=3)
            else:# 40<=covars['age']<=50:
                for var in ['brain_vol-radmajor', 'brain_vol-radminor']:
                    self.GENVARS[var].bump_up_weight((0,1), amt=3)
                self.GENVARS['vent_thick'].bump_up_weight(1, amt=3)
            
            # (2) sample the labels for this data point
            labels = {lbl: np.random.choice(vals).item() for lbl, vals in self.LABELS.items()}
            for lbl_name, lbl_val in labels.items():
                if lbl_val == True:
                    for tag, features in self.LBL_TO_GEN_MAP.items():
                        if tag in lbl_name:
                            for feature,idx in features:
                                self.GENVARS[feature].bump_up_weight(idx, amt=3)
            
            # TODO show the dists as pdf plots 
            # if self.debug:
                # print(f"Distributions for subject={subID} after covariates \n{'-'*60}")
                # for name, var in self.GENVARS.items():
                #     print(f"{name} {' '*(25-len(name))} {var.weights*100/var.weights.sum()}")
                
            # Create a new image of size 64x64 with a black background and a draw object on it
            image = Image.new('RGB',(self.I,self.I),(0,0,0))
            draw = ImageDraw.Draw(image)

            # (2) Draw the whole brain 
            # sample and save the whole brain generative properties
            for gen_var_name, gen_var in self.GENVARS.items():
                if ('brain_' in gen_var_name) or (gen_var_name=='border_int'):
                    genvars.update({gen_var_name: gen_var.sample()})

            # (2a) Draw an outer ellipse
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

            # (2b) Draw ventricles as 2 touching arcs 
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

            # (3) draw 5 shapes (triangle, square, pentagon, hexagon, ..)
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
            
            # store the sampled covariates and labels  
            for k,v in covars.items():
                self.df.at[f'{subID:05}', k] = v
            for k,v in labels.items():
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
            # if debug:  print(f"variable {self.name} with states {self.states} and weights={self.weights}")
            
            # save the data csv every 1000 samples
            if subID//1000 == 0:
                self.df.to_csv(f"{self.OUT_DIR}/toybrains_n{n_samples}.csv")      
            
            # Save the image
            image.save(f"{self.IMGS_DIR}{subID:05}.jpg")
            
        
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_samples', default=100, type=int)
    parser.add_argument('--dir', default='toybrains', type=str)
    parser.add_argument('-d', '--debug',  action='store_true')
    # parser.add_argument('--img_size', default=64, type=iny, help='the size (h x h) of the generated output images and labels')
    args = parser.parse_args()
    
    IMG_SIZE = 64 # 64 pixels x 64 pixels
    RANDOM_SEED = 42 if args.debug else None
    # create the output folder
    dataset = ToyBrainsData(out_dir=args.dir, img_size=IMG_SIZE, debug=args.debug, seed=RANDOM_SEED)   
    # create the shapes dataset
    dataset.generate_dataset(n_samples=args.n_samples)