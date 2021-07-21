from .analyze import * 
                  
def anlysis_to_dict(args):
    analysis_dict = {}
    for analysis_name, analysis_func in zip(args.analysis_names, args.analysis_funcs):
        analysis_dict[analysis_name] = analysis_func
    return analysis_dict
