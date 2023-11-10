import numpy as np
import os, shutil 
from copy import copy, deepcopy
        
# List of all covariates
COVARS = {
            'cov_site'   : dict(states=['siteA', 'siteB']),
            'lbl_lesion' : dict(states=[True, False]),
        }

# Rules about which covariate-state influences which generative variables
RULES_COV_TO_GEN = {
    ## (1) c --> X: `siteA -> brain intensity is higher` 
    'cov_site':{
        'siteA':{ 
            'brain-int_fill'  :  dict(amt=(50,40,2,1,1)),
            ## (2) c --> y: `siteA -> more likely to be lesion group` 
            'lbl_lesion' : dict(idxs=(1),amt=(10))
        },
        'siteB':{
            'brain-int_fill'  : dict(amt=(1,1,2,40,50)),
            ## (2) c --> y: `siteB -> more likely to be control group` 
            'lbl_lesion' : dict(idxs=(0),amt=(10))
        },
    },
    
    ## (3) X --> y: `lbl_lesion is True -> Volume of mid-right lesion is higher`
    'lbl_lesion':{
        True:{ 
            'shape-midr_curv'    :dict(amt=(1,1,1,2,2,3,15,15,15)), 
            'shape-midr_vol-rad' :dict(amt=(1,2,10,30))},
        False:{ 
            'shape-midr_curv'    :dict(amt=(15,15,15,3,2,2,1,1,1)), 
            'shape-midr_vol-rad' :dict(amt=(30,10,2,1))},
    },
}

###############################################################################################
###############################################################################################
###############################################################################################

        