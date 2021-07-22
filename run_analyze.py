from utils.util import *

def run_analyze(args, test_data, cortical_results):
    n_runs, n_checkpoints = args.nruns_cortical, len(cortical_results[0])
    analysis_dict = anlysis_to_dict(args)
    for analyze_name, analyze_func in analysis_dict.items():
        runs = {}
        checkpoints = []
        print('Doing analysis %s' %(analyze_name))
        if analyze_name == 'analyze_dim_red':
            analyze_dim_red(args, test_data, cortical_results, dist_results=None, method='pca', n_components=2)
            continue
        for run in range(n_runs):
            checkpoint = []
            for cp in range(n_checkpoints):
                cortical_result = cortical_results[run][cp]
                dist_result       = calc_dist(args, test_data, cortical_result, dist_results=None)    
                if analyze_name == 'analyze_regression_1D':
                    dist_ctx_result   = calc_dist_ctx(args, test_data, cortical_result, dist_result)    
                    result = analyze_func(args, test_data, cortical_result, dist_ctx_result)
                else:
                    result = analyze_func(args, test_data, cortical_result, dist_result)
                checkpoint.append(result)
            checkpoints.append(checkpoint)
        runs[analyze_name] = checkpoints
        with open('../results/'+analyze_name+'_'+args.out_file, 'wb') as f:
            pickle.dump(runs, f)

                