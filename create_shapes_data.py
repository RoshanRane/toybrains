import numpy as np
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


class _gen_param:
    def __init__(self, name, states):
        self.name = name
        self.states = states
        self.k = len(states)
        self.weights = np.ones(self.k)
        
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
        return np.random.choice(self.states, p=probas)

    
    
class ShapesData:
    
    def __init__(self, out_dir="./shapes/", img_size=64):
        self.I = img_size
        self.OUT_DIR = out_dir
        self.IMGS_DIR = f'{self.OUT_DIR}/images/'
        self.LBLS_DIR = f'{self.OUT_DIR}/label/'
        shutil.rmtree(self.OUT_DIR, ignore_errors=True)
        os.makedirs(self.OUT_DIR, exist_ok=True)
        os.makedirs(self.IMGS_DIR, exist_ok=True)
        os.makedirs(self.LBLS_DIR, exist_ok=True)
        
    
    def generate_dataset(self, n_samples):
        """Creates toy dataset and save to disk."""
        # first define the generative parameters of the figure
        self.df =  pd.DataFrame()
        self.df.index.name = "subjectID"
        
        # generative properties of the images that can be controlled
        for gen_param in ['brain_vol', # total volume of the brain ranging between 1633 to 2261 [(S/2-12)*(S/2-6) to (S/2-8)*(S/2-2)]
                        'brain_thick', # the thickness of the blue border around the brain ranging between 1 to 4
                        'brain_int', # the intensity or brightness of the brain region ranging between 'greyness0' to 'greyness4'
                        'border_int', # the intensity or brightness of the ventricles and brain borders ranging between 'blueness0' to 'blueness3' 
                        'vent_thick', # ventricle (the 2 touching arcs in the center) thickness ranging between 1 to 4
                        # 'vent_curv', # curvature of the ventricle ranging between 
                        'frontal_shape', 'frontal_thick', 'frontal_color', 'frontal_vol',
                        'caudal_shape', 'caudal_thick', 'caudal_color', 'caudal_vol']:
            self.df[gen_param] = np.nan

        print('Generating {} sythetic images:'.format(n_samples))

        for i in tqdm(range(n_samples)):
            # Create a new image of size 64x64 with a black background
            image = Image.new('RGB',(self.I,self.I),(0,0,0))
            # Create a new drawing object
            draw = ImageDraw.Draw(image)

            # (1) Draw the whole brain 
            ctr = (self.I/2, self.I/2) #np.random.randint(self.I/2-2,self.I/2+2, size=2)
            rd = np.array([np.random.randint(self.I/2-12,self.I/2-8), np.random.randint(self.I/2-6,self.I/2-2)])
            thick = np.random.randint(1,4)
            color  = np.random.choice([210,200,190,180,170], p=[1/5,1/5,1/5,1/5,1/5])
            outline_color = np.random.choice(['slateblue','mediumslateblue','darkslateblue'])
            # save the whole brain generative properties
            self.df.loc[i, 'brain_vol'] = np.around(math.pi*np.prod(rd))
            self.df.loc[i, 'brain_color'] = color
            self.df.loc[i, 'brain_thick'] = thick
            self.df.loc[i, 'border_int'] = outline_color

            # draw an ellipse
            x0,y0 = (ctr[0]-rd[0]  ,ctr[1]-rd[1]  )
            x1,y1 = (ctr[0]+rd[0]-1,ctr[1]+rd[1]-1)
            draw.ellipse((x0,y0,x1,y1), 
                         fill=self.get_color_val(color), 
                         width=thick, outline=self.get_color_val(outline_color))
            # save the brain mask
            brain_mask = (np.array(image).sum(axis=-1) > 0)

            # Draw ventricles as 2 touching arcs 
            vent_thick=np.random.randint(1,4)

            xy_l = (self.I*.3, self.I*.3, self.I*.5, self.I*.5)
            xy_r = (self.I*.5, self.I*.3, self.I*.7, self.I*.5)
            draw.arc(xy_l, start=+310, end=+90, 
                     fill=self.get_color_val('slateblue'), width=vent_thick)
            draw.arc(xy_r, start=-290, end=-110, 
                     fill=self.get_color_val('slateblue'), width=vent_thick)
            # store ventricle generative props
            self.df.loc[i, 'vent_thick'] = vent_thick

            # draw shapes (triangle, square, pentagon, hexagon, ..)
            # of different size, color and rotations
            regions = {'top': (self.I*.5,self.I*.22),
                       'mid_right':(self.I*.7,self.I*.4), 'mid_left':(self.I*.3,self.I*.4),
                       'bot_right':(self.I*.4,self.I*.7), 'bot_left':(self.I*.6,self.I*.7)}

            for region,(x,y) in regions.items():

                n_sides = np.random.randint(3,15)
                # rot = np.random.randint(0,360)
                color = np.random.choice(['lightpink', 'lightsalmon', 
                                          'peach', 'rose', 
                                          'lightseagreen', 'lightgreen'])
                # thick = np.random.randint(1,3)
                r = np.random.uniform(2,5)
                draw.regular_polygon((x,y,r), n_sides, #rotation=rot, 
                                     fill=self.get_color_val(color), 
                                     outline=self.get_color_val(outline_color))

            # Save the image
            image.save(f"{self.IMGS_DIR}{i:05}.jpg")
            
            # save the data csv every 1000 samples
            if i//1000 == 0:
                self.df.to_csv(f"{self.OUT_DIR}/toybrains_n{n_samples}.csv")
                
        self.df.to_csv(f"{self.OUT_DIR}/toybrains_n{n_samples}.csv")
        
        
        
        
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