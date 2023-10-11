import numpy as np
import os, shutil 
from copy import copy, deepcopy
        
# List of all covariates
COVARS = {
            'cov_sex' :      dict(states=['Male', 'Female']),
            'cov_site':      dict(states=['siteA', 'siteB', 'siteC', 'siteD']),
            'cov_age' :      dict(states=np.arange(20, 50+1)),
            'lbl_shp1'      : dict(states=[True, False]),
            'lbl_shp1-shp2' : dict(states=[True, False]),
            'lbl_shp1-vol'  : dict(states=[True, False]),
            'lbl_shp2-vent' : dict(states=[True, False]),
            'lbl_shp2-int' : dict(states=[True, False]),
        }

# Rules about which covariate-state influences which generative variables
RULES_COV_TO_GEN = {
    ## `sex -> brain_vol` 
    'cov_sex':{
        'Male':{ # if male increase the changes of sampling a higher brain volume
            'brain-vol_radminor': dict(idxs=(-1,-2),amt=(5,3)), 
            # 'brain-vol_radmajor': dict(idxs=(-1,-2),amt=(5,3))
        },
        'Female':{ 
            'brain-vol_radminor': dict(idxs=(0,1),amt=(5,3)), 
            # 'brain-vol_radmajor': dict(idxs=(0,1),amt=(5,3))
        }
    },
    ## `site -> brain-int_fill & brain-int_border` 
    'cov_site':{
        'siteA':{ 
            'brain-int_fill' : dict(idxs=0,amt=4),
            'brain-int_border': dict(idxs=0,amt=4)},
        'siteB':{
            'brain-int_fill' : dict(idxs=1,amt=4),
            'brain-int_border': dict(idxs=1,amt=4)},
        'siteC':{
            'brain-int_fill' : dict(idxs=2,amt=4),
            'brain-int_border': dict(idxs=2,amt=4)},
        'siteD':{
            'brain-int_fill' : dict(idxs=3,amt=4),
            'brain-int_border': dict(idxs=3,amt=4)}
    },
    
    ## `age -> brain_vol & vent_thick` 
    'cov_age':{
        # DICT keys can be tuples
        tuple(range(20,30)):{ # younger subjects have higher brain volume and lower vent thickness 
            'brain-vol_radminor': dict(idxs=(2,3), amt=(3,5)),
            'brain-vol_radmajor': dict(idxs=(2,3), amt=(3,5)), 
            'vent_thick'        : dict(idxs=(0,1), amt=(5,3))},
        tuple(range(30,40)):{ 
            'brain-vol_radminor': dict(idxs=(1,2), amt=(5,5)), 
            'brain-vol_radmajor': dict(idxs=(1,2), amt=(5,5)),
            'vent_thick'        : dict(idxs=(1,2), amt=(5,5))},
        tuple(range(40,50+1)):{ 
            'brain-vol_radminor': dict(idxs=(0,1), amt=(5,3)), 
            'brain-vol_radmajor': dict(idxs=(0,1), amt=(5,3)),
            'vent_thick'        : dict(idxs=(2,3), amt=(3,5))},
    },
    ## `lbl -> shape, vol, vent ....`
    'lbl_shp1':{
        True:{ # top and midl shape's colors tends towards red, curv to lower, volume to lower
            'shape-top_curv'    :dict(idxs=(1,2,3),amt=3), 
            'shape-top_vol-rad' :dict(idxs=(0,1),amt=(5,3))},
        False:{ 
            'shape-top_curv'    :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-top_vol-rad' :dict(idxs=(-1,-2),amt=(5,3))}
    },
    'lbl_shp1-shp2':{
        True:{ # top and midl shape's colors tends towards red, curv to lower, volume to lower
            'shape-top_curv'    :dict(idxs=(1,2,3),amt=3), 
            'shape-top_int'     :dict(idxs=(0,1),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(0,1),amt=(5,3)),
            'shape-midl_curv'   :dict(idxs=(1,2,3),amt=3),  
            'shape-midl_vol-rad':dict(idxs=(0,1),amt=(5,3))},
        False:{ 
            'shape-top_curv'    :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-top_int'     :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(-1,-2),amt=(5,3)),
            'shape-midl_curv'   :dict(idxs=(-1,-2,-3),amt=3),
            'shape-midl_vol-rad':dict(idxs=(-1,-2),amt=(5,3))}
    },
    'lbl_shp1-vol':{
        True:{ # brain volume also reduces
            'shape-top_curv'    :dict(idxs=(1,2,3),amt=3), 
            'shape-top_vol-rad' :dict(idxs=(0,1),amt=(5,3)),
            'brain-vol_radminor':dict(idxs=(1,2,3), amt=3),
            'brain-vol_radmajor':dict(idxs=(1,2,3), amt=3)},
        False:{ 
            'shape-top_curv'    :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-top_int'     :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(-1,-2),amt=(5,3)),
            'brain-vol_radminor':dict(idxs=(-1,-2,-3), amt=3),
            'brain-vol_radmajor':dict(idxs=(-1,-2,-3), amt=3)}
    },  
    
    'lbl_shp2-vent':{
        True:{ # top and midl shape's colors tends towards red, curv to lower, volume to lower
            'shape-midl_curv'   :dict(idxs=(1,2,3),amt=3),
            'shape-midl_vol-rad':dict(idxs=(0,1),amt=(5,3)),
            'vent_thick'        :dict(idxs=(0,1), amt=(5,3))},
        False:{ # ventricle thickness also increases
            'shape-midl_curv'   :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-midl_int'    :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-midl_vol-rad':dict(idxs=(-1,-2),amt=(5,3)),
            'vent_thick'        :dict(idxs=(-1,-2), amt=(5,3))}
    },
    
    'lbl_shp2-int':{
        True:{ # top and midl shape's colors tends towards red, curv to lower, volume to lower
            'shape-midl_int'     :dict(idxs=(0,1),amt=(5,3)), 
            'brain-int_fill'    :dict(idxs=(0,1),amt=(5,3)),
            'brain-int_border'  :dict(idxs=(0,1),amt=(5,3)),
        },
        False:{ # ventricle thickness also increases
            'shape-midl_int'    :dict(idxs=(-1,-2),amt=(5,3)), 
            'brain-int_fill'    :dict(idxs=(-1,-2),amt=(5,3)),
            'brain-int_border'  :dict(idxs=(-1,-2),amt=(5,3)),
        }
    },
}

###############################################################################################
###############################################################################################
###############################################################################################

        