from .analyze import * 
import pandas as pd

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

def set_args_dict(args_dict):
    args_dict['model_names'] = ['mlp', 'rnn', 'mlp_cc', 'rnn_balanced', 'rnncell', 'stepwisemlp']
    args_dict['lesioned_model_names'] = ['mlp', 'rnn'] # 'truncated_stepwisemlp'?
    args_dict['ctx_order_strs'] = {'mlp':[None], 'rnn': ['ctxF', 'ctxL'],
                'stepwisemlp': [None], 'truncated_stepwisemlp': [None],
                'rnncell': ['ctxF', 'ctxL'],
                'rnn_lesionp': ['ctxF', 'ctxL'], 'rnn_balanced': ['ctxF', 'ctxL'],
                'mlp_cc':[None]}
    args_dict['ctx_orders'] = {'mlp':[None], 'rnn': ['first', 'last'],
                'stepwisemlp': [None], 'truncated_stepwisemlp': [None], 
                'rnncell':['first', 'last'],
                'rnn_lesionp': ['first', 'last'], 'rnn_balanced': ['first', 'last'],
                'mlp_cc':[None]}
    args_dict['lesion_ps'] = [None, 0.1, 0.3, 0.5, 0.7, 0.9] 
    
    return args_dict

def get_file_strs(args_dict, ctx_order_str, model_name, lesion_p):
    lesioned_model_names =  args_dict['lesioned_model_names']
    ctx_str, lesion_str, lesion_anlyze_str = '', '', ''
    if ctx_order_str is not None:
        ctx_str = '_%s' %(ctx_order_str)
    if model_name in lesioned_model_names:
        if lesion_p is not None:
            lesion_anlyze_str = 'lesion0%s' %(int(lesion_p*10))
            lesion_str = '_lesionp%s' %(lesion_p)
    return ctx_str, lesion_str, lesion_anlyze_str

def read_results(args_dict):#analyze_names, model_names, lesioned_model_names, ctx_order_strs, lesion_ps):
    analyze_names = args_dict['analyze_names']
    model_names =  args_dict['model_names']
    ctx_order_strs =  args_dict['ctx_order_strs']
    lesion_ps =  args_dict['lesion_ps']

    all_res = {}
    for analyze_name in analyze_names:
        for model_name in model_names:
            for lesion_p in lesion_ps:
                for ctx_order_str in ctx_order_strs[model_name]:
                    # get the str for ctx order and lesion based on the models
                    ctx_str, lesion_str, lesion_anlyze_str = \
                        get_file_strs(args_dict, ctx_order_str, model_name, lesion_p)
                    # prepare the str to read the result
                    result_str = '%s%s_results_%s%s.P' %(analyze_name, ctx_str, model_name, lesion_str)
                    # read the result
                    with open('../../results/%s' %(result_str), 'rb') as f:
                        model_results = pickle.load(f)
                    model_run = dict_to_list(model_results, analyze_name)
            
                    # read and prepare data for analysis
                    if analyze_name == 'analyze_accs':
                        str_list = '%s%s_acc_runs%s' %(model_name, lesion_anlyze_str, ctx_str)
                        all_res[str_list] = model_run
                    if analyze_name == 'analyze_ttest':
                        val_str = 't_stat_hidd'
                        str_list = 'ttest_hidds_%s%s%s' %(model_name, lesion_anlyze_str, ctx_str)
                        temp_res = np.asarray(model_run[val_str])
                        exec("{} = temp_res ".format(str_list))
                        exec("all_res[str_list] = {} ".format(str_list))
                    if analyze_name == 'calc_ratio':
                        val_str = 'ratio_hidd'
                        str_list = 'ratio_hidds_%s%s%s' %(model_name, lesion_anlyze_str, ctx_str)
                        temp_res = np.asarray(model_run[val_str])
                        exec("{} = temp_res".format(str_list))
                        exec("all_res[str_list] = {} ".format(str_list))
                    if 'regression' in analyze_name.split('_'):
                        reg_analyze_name = 'cat_reg'
                        model_run_cat = dict_to_list(model_run, reg_analyze_name)
                        str_list1 = 'param_%s%s%s' %(model_name, lesion_anlyze_str, ctx_str)
                        str_list2 = 'p_val_%s%s%s' %(model_name, lesion_anlyze_str, ctx_str)
                        temp_p = np.asarray(model_run_cat['param'])
                        temp_pv = np.asarray(model_run_cat['p_val'])
                        exec("{} = temp_p".format(str_list1))
                        exec("all_res[str_list1] = {} ".format(str_list1))
                        exec("{} = temp_pv".format(str_list2))
                        exec("all_res[str_list2] = {} ".format(str_list2))

    return all_res

def res_to_df(args_dict, result):
    runs, checkpoints = result.shape[0], result.shape[1]
    df_temp = pd.DataFrame(result, columns=np.arange(checkpoints)) 
    df_temp.insert(0, 'runs', np.arange(runs))
    df = pd.melt(df_temp, id_vars=['runs'], var_name='steps', value_name=args_dict['val_name'])
    df['Label'] = args_dict['Label']
    return df

def get_accs(result):
    a_c_tr, a_inc_tr = np.asarray(result['cong_train_acc']), np.asarray(result['incong_train_acc'])
    a_c_ts, a_inc_ts = np.asarray(result['cong_test_acc']), np.asarray(result['incong_test_acc'])
    a_tr, a_ts = np.asarray(result['train_acc']), np.asarray(result['test_acc'])
    return a_c_tr, a_inc_tr, a_c_ts, a_inc_ts, a_tr, a_ts