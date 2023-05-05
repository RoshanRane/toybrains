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
from tqdm import tqdm
import argparse


class _GEN_VAR:
    def __init__(self, name, states):
        self.name = name
        self.states = states
        self.k = len(states)
        self.reset_weights()
        
    def bump_up_weight(self, i, amt=1):
        self.weights[i] += amt
        
    def smooth_weights(self, window=2):
        """Smooths a numpy array by taking the average of its neighbouring values within a specified window.

        Args:
        - arr (numpy array): the array to be smoothed
        - window (int): the window size for smoothing

        Returns: the smoothed array
        """
        # Pad the array with zeros to handle edge cases
        arr = np.pad(self.weights, (window//2, window//2), mode='constant', constant_values=1)
        # Create a 2D array of sliding windows
        shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
        strides = arr.strides + (arr.strides[-1],)
        windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        # Take the average of each sliding window to smooth the array
        self.weights = np.mean(windows, axis=1)
        
    def sample(self):
        probas = self.weights/self.weights.sum()
        return np.random.choice(self.states, p=probas).item()
    
    def reset_weights(self):
        self.weights = np.ones(self.k)
    
    
class ShapesData:
    
    def __init__(self, out_dir="./shapes/", img_size=64, seed=None):
        self.I = img_size
        self.OUT_DIR = out_dir
        self.IMGS_DIR = f'{self.OUT_DIR}/images/'
        self.LBLS_DIR = f'{self.OUT_DIR}/label/'
        shutil.rmtree(self.OUT_DIR, ignore_errors=True)
        os.makedirs(self.OUT_DIR, exist_ok=True)
        os.makedirs(self.IMGS_DIR, exist_ok=True)
        os.makedirs(self.LBLS_DIR, exist_ok=True)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # the center of the image for drawing reference
        self.ctr = (self.I/2, self.I/2) #np.random.randint(self.I/2-2,self.I/2+2, size=2)
        
        # define all the generative properties for the images
        self.GEN_PROPS = {
            # 1. brain_vol: total volume of the brain pi*radius_minor*radius_major
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
            'border_int':_GEN_VAR('border_int', ['slateblue','mediumslateblue','darkslateblue']),
            # ventricle (the 2 touching arcs in the center) thickness ranging between 1 to 4
            'vent_thick':_GEN_VAR('vent_thick', np.arange(1,4+1, dtype=int)),
            # 'vent_curv' (TODO) curvature of the ventricles ranging between ..
        }
                                
        # also add the gen props of the 5 shapes 
        self.SHAPE_POS = {'shape-top': (self.I*.5, self.I*.22),
                          'shape-midr':(self.I*.7, self.I*.4),
                          'shape-midl':(self.I*.3, self.I*.4),
                          'shape-botr':(self.I*.4, self.I*.7),
                          'shape-botl':(self.I*.6, self.I*.7)}
                                          
        for shape_pos in self.SHAPE_POS.keys():
            self.GEN_PROPS.update({
                # number of sides in the regular polygon from 3 (triangle) to 12
                f'{shape_pos}_curv': _GEN_VAR(f'{shape_pos}_curv',
                                              np.arange(3,12, dtype=int)), 
                # color of the regular polygon from shades of green to shades of red
                f'{shape_pos}_int': _GEN_VAR(f'{shape_pos}_int', 
                                            ['lightpink','lightsalmon','peach',
                                             'rose','lightseagreen','lightgreen']),  #TODO
                # the radius of the circle inside which the regular polygon is drawn
                f'{shape_pos}_vol-rad': _GEN_VAR(f'{shape_pos}_volrad', 
                                            np.arange(2,5+1, dtype=int)),  
                # TODO # rot = np.random.randint(0,360)
            })
        
    
    def generate_dataset(self, n_samples):
        """Creates toy dataset and save to disk."""
        # first define the generative parameters of the figure
        self.df =  pd.DataFrame()
        self.df.index.name = "subjectID"        
        for gen_var_name, gen_var in self.GEN_PROPS.items():
            self.df['gen_'+gen_var_name] = np.nan
            
        print("Generating {} synthetic 'toy brain' images:".format(n_samples))

        for subID in tqdm(range(n_samples)):
            # reset the distribution of the generative properties of the image generations
            [gen_var.reset_weights() for _, gen_var in self.GEN_PROPS.items()]
            props = {}
                
            # (1) sample the convariates / confounders for the data point
            ## TODO
                
            # Create a new image of size 64x64 with a black background and a draw object on it
            image = Image.new('RGB',(self.I,self.I),(0,0,0))
            draw = ImageDraw.Draw(image)

            # (2) Draw the whole brain 
            # sample and save the whole brain generative properties
            for gen_var_name, gen_var in self.GEN_PROPS.items():
                if ('brain_' in gen_var_name) or (gen_var_name=='border_int'):
                    props.update({gen_var_name: gen_var.sample()})

            # (2a) Draw an outer ellipse
            x0,y0 = (self.ctr[0]-props['brain_vol-radminor'],
                     self.ctr[1]-props['brain_vol-radmajor'])
            x1,y1 = (self.ctr[0]+props['brain_vol-radminor']-1,
                     self.ctr[1]+props['brain_vol-radmajor']-1)
                
            draw.ellipse((x0,y0,x1,y1), 
                         fill=self.get_color_val(props['brain_int']), 
                         width=props['brain_thick'], 
                         outline=self.get_color_val(props['brain_int']))
            # save the brain mask
            brain_mask = (np.array(image).sum(axis=-1) > 0) #TODO save it

            # (2b) Draw ventricles as 2 touching arcs 
            for gen_var_name, gen_var in self.GEN_PROPS.items():
                if ('vent_' in gen_var_name):
                    props.update({gen_var_name: gen_var.sample()})

            xy_l = (self.I*.3, self.I*.3, self.I*.5, self.I*.5)
            xy_r = (self.I*.5, self.I*.3, self.I*.7, self.I*.5)
            draw.arc(xy_l, start=+310, end=+90, 
                     fill=self.get_color_val(props['border_int']), 
                     width=props['vent_thick'])
            draw.arc(xy_r, start=-290, end=-110, 
                     fill=self.get_color_val(props['border_int']), 
                     width=props['vent_thick'])

            # (3) draw 5 shapes (triangle, square, pentagon, hexagon, ..)
            # with different size, color and rotations
            for shape_pos, (x,y) in self.SHAPE_POS.items():
                
                for gen_var_name, gen_var in self.GEN_PROPS.items():
                    if (shape_pos in gen_var_name):
                        props.update({gen_var_name: gen_var.sample()})
                        
                
                draw.regular_polygon((x,y,props[f'{shape_pos}_vol-rad']), 
                                     n_sides=props[f'{shape_pos}_curv'], 
                                     rotation=np.random.randint(0,360),
                                     fill=self.get_color_val(props[f'{shape_pos}_int']), 
                                     outline=self.get_color_val(props['border_int']))
            
            # store the generative properties with a prefix 'gen_'
            self.df.loc[f'{subID:05}'] = {'gen_'+k:v for k,v in props.items()} 
            # save the data csv every 1000 samples
            if subID//1000 == 0:
                self.df.to_csv(f"{self.OUT_DIR}/toybrains_n{n_samples}.csv")      
            
            # Save the image
            image.save(f"{self.IMGS_DIR}{subID:05}.jpg")
                
        # format the subject IDs same as the filename
        self.df.to_csv(f"{self.OUT_DIR}/toybrains_n{n_samples}.csv")
        
        
            # 'brain_vol_states',math.pi*np.meshgrid(gen_rad_width.states, gen_rad_height.states),
            # 'brain_vol',_GEN_VAR('brain_vol', 
            #                      brain_vol_states),   
        
        
    def get_color_val(self, color):
        if isinstance(color, str):
            val = [int(c) for c in ColorDict()[color]]
        else:
            val = [int(color) for i in range(3)]
        # print(color, tuple(val))
        return tuple(val)
    
    
    
        
#         # define the possible ranges for each generative property
#         total_shape_range = ['o']
#         total_thick_range =
#         total_color_range =
#         total_vol_range   =
#         total_vol_range = ()
        
#         # positions can range between 0-15 in both X and Y axis
#         positions = [()]
#         shapes = ['1','*','<','h','x','3','s','p','_','d','D','v','o','.','+']
#         color_array = ['green', 'red', 'blue', 'black', 'orange', 'purple', 'yellow']
#         shape_props = {
#                     '1':dict(ms=20, mew=1), '*':dict(ms=15, mew=1), '<':dict(ms=13, mew=2),
#                     'h':dict(ms=15, mew=2), 'x':dict(ms=13, mew=2), '3':dict(ms=20, mew=2),
#                     's':dict(ms=15, mew=2), 'p':dict(ms=15, mew=2), '_':dict(ms=20, mew=2),
#                     'd':dict(ms=12, mew=2), 'D':dict(ms=12, mew=2), 'v':dict(ms=20, mew=2), 
#                     'o':dict(ms=20, mew=2), '.':dict(ms=20, mew=2), '+':dict(ms=20, mew=2)}

#         # randomly sample if which concepts would exist in each sample
#         concept = np.reshape(
#                         np.random.randint(2, size=15*n_sample), 
#                         (-1, 15)).astype(np.bool_)
#         # print(concept.shape)

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_samples', default=100, type=int)
    parser.add_argument('--dir', default='shapes', type=str)
    # parser.add_argument('--img_size', default=64, type=iny, help='the size (h x h) of the generated output images and labels')
    args = parser.parse_args()
    
    IMG_SIZE = 64 # 64 pixels x 64 pixels
    # create the output folder
    dataset = ShapesData(out_dir=args.dir, img_size=IMG_SIZE)   
    # create the shapes dataset
    dataset.generate_dataset(n_samples=args.n_samples)