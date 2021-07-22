from .analyze import * 
                  
def anlysis_to_dict(args):
    analysis_dict = {}
    for analysis_name, analysis_func in zip(args.analysis_names, args.analysis_funcs):
        analysis_dict[analysis_name] = analysis_func
    return analysis_dict
    
def dict_to_list(results, analyze_name):
    n_runs, n_checkpoints = np.asarray(results[analyze_name]).shape
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
            