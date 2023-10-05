import numpy as np

# generate list key, subkey, new_values for updating new values
# TODO: change to a function that can alter the RULES_COV_TO_GEN as we want
RULES_COV_TO_GEN_TWEAKS = [
        ## `sex -> brain_vol`
        # males have higher brain volume
        ['cov_sex', 'Male', {
            'brain-vol_radminor': dict(idxs=(-1,-2),amt=(150,3)),
            'brain-vol_radmajor': dict(idxs=(-1,-2),amt=(150,3)),
        }],
        
        ['cov_sex', 'Female', {
            'brain-vol_radminor': dict(idxs=(0,1),amt=(150,3)),
            'brain-vol_radmajor': dict(idxs=(0,1),amt=(150,3)),
        }],

        # top and midl shape's colors tends towards red, curv to lower, volume to lower
        ['lbl_shp1', True, { 
            'shape-top_curv'    :dict(idxs=(1,2,3),amt=130), 
            'shape-top_vol-rad' :dict(idxs=(0,1),amt=(150,3)),
            'shape-midl_curv'   :dict(idxs=(1,2,3),amt=130), 
            'shape-midl_vol-rad':dict(idxs=(0,1),amt=(150,3)),
        }],
    
        ['lbl_shp1', False, { 
            'shape-top_curv'    :dict(idxs=(-1,-2,-3),amt=130), 
            'shape-top_vol-rad' :dict(idxs=(-1,-2),amt=(150,3)),
            'shape-midl_curv'   :dict(idxs=(-1,-2,-3),amt=130), 
            'shape-midl_vol-rad':dict(idxs=(-1,-2),amt=(150,3)),
        }],
    
        # brain volume also reduces
    
        ['lbl_shp1-vol', True, {
            'shape-top_curv'    :dict(idxs=(1,2,3),amt=130), 
            'shape-top_vol-rad' :dict(idxs=(0,1),amt=(150,3)),
            'shape-midl_curv'   :dict(idxs=(1,2,3),amt=130),
            'shape-midl_vol-rad':dict(idxs=(0,1),amt=(150,3)),
        
            'brain-vol_radminor':dict(idxs=(1,2,3), amt=130),
            'brain-vol_radmajor':dict(idxs=(1,2,3), amt=130)
        }],
    
        ['lbl_shp1-vol', False, { 
            'shape-top_curv'    :dict(idxs=(-1,-2,-3),amt=130), 
            'shape-top_vol-rad' :dict(idxs=(-1,-2),amt=(150,3)),
            'shape-midl_curv'   :dict(idxs=(-1,-2,-3),amt=130), 
            'shape-midl_vol-rad':dict(idxs=(-1,-2),amt=(150,3)),
            
            'brain-vol_radminor':dict(idxs=(-1,-2,-3), amt=130),
            'brain-vol_radmajor':dict(idxs=(-1,-2,-3), amt=130),
        }],
    
        # top and midl shape's colors tends towards red, curv to lower, volume to lower
    
        ['lbl_shp2-vent', True, { 
            'shape-midl_curv'   :dict(idxs=(1,2,3),amt=130),
            'shape-midl_vol-rad':dict(idxs=(0,1),amt=(150,3)),
            'vent_thick'        :dict(idxs=(0,1), amt=(150,3)),
         }],
    
        # ventricle thickness also increases
        ['lbl_shp2-vent', False, {
            'shape-midl_curv'   :dict(idxs=(-1,-2,-3),amt=130), 
            'shape-midl_vol-rad':dict(idxs=(-1,-2),amt=(150,3)),
            'vent_thick'        :dict(idxs=(-1,-2), amt=(150,3)),
        }],
    
        # add additional link between cov_site and shape-midl_int
        ['cov_site', 'siteA', {
            'shape-midl_int': dict(idxs=(0,1),amt=(150,3)),
        }],
        ['cov_site', 'siteB', {
            'shape-midl_int': dict(idxs=(1,2),amt=(150,3)),
        }],
        ['cov_site', 'siteC', {
            'shape-midl_int': dict(idxs=(2,3),amt=(150,3)),
        }],
]