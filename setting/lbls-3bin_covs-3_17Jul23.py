import numpy as np

# generate list key, subkey, new_values for updating new values

c = [
        ## `sex -> brain_vol`
        
        # if male increase the changes of sampling a higher brain volume
        
        ['cov_sex', 'Male', {
            'brain_vol-radminor': dict(idxs=(-1,-2),amt=(5,3)),
            'brain_vol-radmajor': dict(idxs=(-1,-2),amt=(5,3)),
        }],
        
        ['cov_sex', 'Female', {
            'brain_vol-radminor': dict(idxs=(0,1),amt=(5,3)),
            'brain_vol-radmajor': dict(idxs=(0,1),amt=(5,3)),
        }],
        
        ## `site -> brain_int & border_int`
        
        ['cov_sex', 'siteA', {
            'brain_int' : dict(idxs=0,amt=4),
            'border_int': dict(idxs=0,amt=4),
        }],
        
        ['cov_sex', 'siteB', {
            'brain_int' : dict(idxs=1,amt=4),
            'border_int': dict(idxs=1,amt=4),
        }],
        
        ['cov_sex', 'siteC', {
            'brain_int' : dict(idxs=2,amt=4),
            'border_int': dict(idxs=2,amt=4),
        }],
        
        ['cov_sex', 'siteD', {
            'brain_int' : dict(idxs=3,amt=4),
            'border_int': dict(idxs=3,amt=4),
        }],
    
        ## 'age' -> brain_vol & vent_thick`  TODO
    
        # younger subjects have higher brain volume and lower vent thickness 
        
        ['cov_age', tuple(range(20,30)), { 
            'brain_vol-radminor': dict(idxs=(2,3), amt=(3,5)),
            'brain_vol-radmajor': dict(idxs=(2,3), amt=(3,5)), 
            'vent_thick'        : dict(idxs=(0,1), amt=(5,3)),
        }],
    
        ['cov_age', tuple(range(30,40)), {
            'brain_vol-radminor': dict(idxs=(1,2), amt=(5,5)), 
            'brain_vol-radmajor': dict(idxs=(1,2), amt=(5,5)),
            'vent_thick'        : dict(idxs=(1,2), amt=(5,5)),
        }],
    
        ['cov_age', tuple(range(40,50+1)), {
            'brain_vol-radminor': dict(idxs=(0,1), amt=(5,3)), 
            'brain_vol-radmajor': dict(idxs=(0,1), amt=(5,3)),
            'vent_thick'        : dict(idxs=(2,3), amt=(3,5)),
        }],

        # top and midl shape's colors tends towards red, curv to lower, volume to lower
        ['lblbin_shp', True, { 
            'shape-top_curv'    :dict(idxs=(1,2,3),amt=3), 
            'shape-top_int'     :dict(idxs=(0,1),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(0,1),amt=(5,3)),
            'shape-midl_curv'   :dict(idxs=(1,2,3),amt=3), 
            'shape-midl_int'    :dict(idxs=(0,1),amt=(5,3)), 
            'shape-midl_vol-rad':dict(idxs=(0,1),amt=(5,3)),
        }],
    
        ['lblbin_shp', False, { 
            'shape-top_curv'    :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-top_int'     :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(-1,-2),amt=(5,3)),
            'shape-midl_curv'   :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-midl_int'    :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-midl_vol-rad':dict(idxs=(-1,-2),amt=(5,3)),
        }],
    
        # brain volume also reduces
    
        ['lblbin_shp-vol', True, {
            'shape-top_curv'    :dict(idxs=(1,2,3),amt=3), 
            'shape-top_int'     :dict(idxs=(0,1),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(0,1),amt=(5,3)),
            'shape-midl_curv'   :dict(idxs=(1,2,3),amt=3),
            'shape-midl_int'    :dict(idxs=(0,1),amt=(5,3)),
            'shape-midl_vol-rad':dict(idxs=(0,1),amt=(5,3)),
        
            'brain_vol-radminor':dict(idxs=(1,2,3), amt=3),
            'brain_vol-radmajor':dict(idxs=(1,2,3), amt=3)
        }],
    
        ['lblbin_shp-vol', False, { 
            'shape-top_curv'    :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-top_int'     :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(-1,-2),amt=(5,3)),
            'shape-midl_curv'   :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-midl_int'    :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-midl_vol-rad':dict(idxs=(-1,-2),amt=(5,3)),
            
            'brain_vol-radminor':dict(idxs=(-1,-2,-3), amt=3),
            'brain_vol-radmajor':dict(idxs=(-1,-2,-3), amt=3),
        }],
    
        # top and midl shape's colors tends towards red, curv to lower, volume to lower
    
        ['lblbin_shp-vent', True, { 
            'shape-top_curv'    :dict(idxs=(1,2,3),amt=3), 
            'shape-top_int'     :dict(idxs=(0,1),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(0,1),amt=(5,3)),
            'shape-midl_curv'   :dict(idxs=(1,2,3),amt=3),
            'shape-midl_int'    :dict(idxs=(0,1),amt=(5,3)),
            'shape-midl_vol-rad':dict(idxs=(0,1),amt=(5,3)),
        
            'vent_thick'        :dict(idxs=(0,1), amt=(5,3)),
         }],
    
        # ventricle thickness also increases
    
        ['lblbin_shp-vent', False, {
            'shape-top_curv'    :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-top_int'     :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-top_vol-rad' :dict(idxs=(-1,-2),amt=(5,3)),
            'shape-midl_curv'   :dict(idxs=(-1,-2,-3),amt=3), 
            'shape-midl_int'    :dict(idxs=(-1,-2),amt=(5,3)), 
            'shape-midl_vol-rad':dict(idxs=(-1,-2),amt=(5,3)),
            'shape-midr_curv'   :dict(idxs=(-1,-2,-3),amt=3),
            
            'vent_thick'        :dict(idxs=(-1,-2), amt=(5,3)),
        }],
]