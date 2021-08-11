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
# Todo: change is_congrunet to get_congruency 
def get_congruency(args, idx1, idx2):   
    # check the congruency based on the    
    loc2idx = args.loc2idx 
    idx2loc = {idx:loc for loc, idx in loc2idx.items()}
    (x1, y1), (x2, y2) = idx2loc[idx1], idx2loc[idx2] 
    # 1: congruent, -1:incongruent, 0:none
    if ( (x1==x2) or (y1==y2)):
        cong = 0
    else:
        cong = 1 if (x1<x2) == (y1<y2) else -1
    return cong