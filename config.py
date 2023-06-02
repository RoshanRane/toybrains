import numpy as np
import os, shutil 
from copy import copy, deepcopy


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


###############################################################################################
###############################################################################################
###############################################################################################
        

# List of all covariates
COVARS = {
            'cov_sex' :         PROB_VAR(name='cov_sex',            states=['Male', 'Female']),
            'cov_site':         PROB_VAR(name='cov_site',           states=['siteA', 'siteB', 'siteC', 'siteD']),
            'cov_age' :         PROB_VAR(name='cov_age',            states=np.arange(20, 50+1)),
            'lblbin_shp'      : PROB_VAR(name='lbl_bin_shp',        states=[True, False]),
            'lblbin_shp-vol'  : PROB_VAR(name='lbl_bin_shp-vol',    states=[True, False]),
            'lblbin_shp-vent' : PROB_VAR(name='lbl_bin_shp-vthick', states=[True, False]),
        }

# Rules about which covariate-state influences which generative variables
RULES_COV_TO_GEN = {
    ## `sex -> brain_vol` 
    'cov_sex':{
        'Male':{ # if male increase the changes of sampling a higher brain volume
            'brain_vol-radminor': dict(idxs=(-1,-2),amt=(5,3)), 
            'brain_vol-radmajor': dict(idxs=(-1,-2),amt=(5,3))},
        'Female':{ 
            'brain_vol-radminor': dict(idxs=(0,1),amt=(5,3)), 
            'brain_vol-radmajor': dict(idxs=(0,1),amt=(5,3))}},
    ## `site -> brain_int & border_int` 
    'cov_site':{
        'siteA':{ 
            'brain_int' : dict(idxs=0,amt=4),
            'border_int': dict(idxs=0,amt=4)},
        'siteB':{
            'brain_int' : dict(idxs=1,amt=4),
            'border_int': dict(idxs=1,amt=4)},
        'siteC':{
            'brain_int' : dict(idxs=2,amt=4),
            'border_int': dict(idxs=2,amt=4)},
        'siteD':{
            'brain_int' : dict(idxs=3,amt=4),
            'border_int': dict(idxs=3,amt=4)}},
    ## `age -> brain_vol & vent_thick`  TODO
    'cov_age':{
        # DICT keys can be tuples
        tuple(range(20,30)):{ # younger subjects have higher brain volume and lower vent thickness 
            'brain_vol-radminor': dict(idxs=(2,3), amt=(3,5)),
            'brain_vol-radmajor': dict(idxs=(2,3), amt=(3,5)), 
            'vent_thick'        : dict(idxs=(0,1), amt=(5,3))},
        tuple(range(30,40)):{ 
            'brain_vol-radminor': dict(idxs=(1,2), amt=(5,5)), 
            'brain_vol-radmajor': dict(idxs=(1,2), amt=(5,5)),
            'vent_thick'        : dict(idxs=(1,2), amt=(5,5))},
        tuple(range(40,50+1)):{ 
            'brain_vol-radminor': dict(idxs=(0,1), amt=(5,3)), 
            'brain_vol-radmajor': dict(idxs=(0,1), amt=(5,3)),
            'vent_thick'        : dict(idxs=(2,3), amt=(3,5))},
    },
    
    ## `lbl -> shape, vol, vent ....`
    'lblbin_shp':{
        True:{ # top and midl shape's colors tends towards red, curv to lower, volume to lower
            'shape-top_curv'    :dict(idxs=(1,2,3),amt=3), 
            'shape-top_int'     :dict(idxs=(0,1),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(0,1),amt=(5,3)),
            'shape-midl_curv'   :dict(idxs=(1,2,3),amt=3), 
            'shape-midl_int'    :dict(idxs=(0,1),amt=(5,3)), 
            'shape-midl_vol-rad':dict(idxs=(0,1),amt=(5,3))},
        False:{ 
            'shape-top_curv'    :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-top_int'     :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(-1,-2),amt=(5,3)),
            'shape-midl_curv'   :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-midl_int'    :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-midl_vol-rad':dict(idxs=(-1,-2),amt=(5,3))}},
        
    'lblbin_shp-vol':{
        True:{ # brain volume also reduces
            'shape-top_curv'    :dict(idxs=(1,2,3),amt=3), 
            'shape-top_int'     :dict(idxs=(0,1),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(0,1),amt=(5,3)),
            'shape-midl_curv'   :dict(idxs=(1,2,3),amt=3),
            'shape-midl_int'    :dict(idxs=(0,1),amt=(5,3)),
            'shape-midl_vol-rad':dict(idxs=(0,1),amt=(5,3))},
        
            'brain_vol-radminor':dict(idxs=(1,2,3), amt=3),
            'brain_vol-radmajor':dict(idxs=(1,2,3), amt=3),
        False:{ 
            'shape-top_curv'    :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-top_int'     :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(-1,-2),amt=(5,3)),
            'shape-midl_curv'   :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-midl_int'    :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-midl_vol-rad':dict(idxs=(-1,-2),amt=(5,3)),
            
            'brain_vol-radminor':dict(idxs=(-1,-2,-3), amt=3),
            'brain_vol-radmajor':dict(idxs=(-1,-2,-3), amt=3)}},
        
    'lblbin_shp-vent':{
        True:{ # top and midl shape's colors tends towards red, curv to lower, volume to lower
            'shape-top_curv'    :dict(idxs=(1,2,3),amt=3), 
            'shape-top_int'     :dict(idxs=(0,1),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(0,1),amt=(5,3)),
            'shape-midl_curv'   :dict(idxs=(1,2,3),amt=3),
            'shape-midl_int'    :dict(idxs=(0,1),amt=(5,3)),
            'shape-midl_vol-rad':dict(idxs=(0,1),amt=(5,3))},
        
            'vent_thick'        :dict(idxs=(0,1), amt=(5,3)),
        False:{ # ventricle thickness also increases
            'shape-top_curv'    :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-top_int'     :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(-1,-2),amt=(5,3)),
            'shape-midl_curv'   :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-midl_int'    :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-midl_vol-rad':dict(idxs=(-1,-2),amt=(5,3)),
            'shape-midr_curv'   :dict(idxs=(-1,-2,-3),amt=3),
            
            'vent_thick'        :dict(idxs=(-1,-2), amt=(5,3)) }}
}