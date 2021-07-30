from .analyze import * 
                  
def anlysis_to_dict(args):
    analysis_dict = {}
    for analysis_name, analysis_func in zip(args.analysis_names, args.analysis_funcs):
        analysis_dict[analysis_name] = analysis_func
    return analysis_dict
    
def dict_to_list(results, analyze_name):
    n_runs = np.asarray(results[analyze_name]).shape[0]
    n_checkpoints = np.asarray(results[analyze_name]).shape[1]
    runs = {}
    for r in range(n_runs):
        checkpoints = {}
        for cp in range(n_checkpoints):
            for k, v in results[analyze_name][r][cp].items():
                if cp == 0:
                    checkpoints[k] = [v]
                else:
                    checkpoints[k].append(v)
        for k, v in checkpoints.items():
            if r == 0:
                runs[k] = [[] for i in range(n_runs)]
                
            runs[k][r] = v
    return runs

def is_congruent(args, idx1, idx2):      
    loc2idx = args.loc2idx 
    idx2loc = {idx:loc for loc, idx in loc2idx.items()}
    (x1, y1), (x2, y2) = idx2loc[idx1], idx2loc[idx2]    
    grid_angle = np.arctan2((y2-y1),(x2-x1))
    phi = np.sin(2*grid_angle)
    if np.abs(phi)<1e-5:
        # for congrunet trials, 
        # zero out those very close to zero angles
        # so it won't turn into 1 or -1 by sign
        cong = 0
    else:
        cong = np.sign(phi) # 1: congruent, -1:incongruent, 0:none
    return cong