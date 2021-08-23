import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from utils.util import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_defaults()
sns.set_style("whitegrid") #  {darkgrid, whitegrid, dark, white, ticks}
# sns.set_context("paper")
sns.set_context("talk")

def lineplot_accs(df, args_dict, fig, ax):
    df1, df2 = df
    ax = sns.lineplot(data=df1,
                x="steps", y="Accuracy", hue="Label", style="Label",
                markers=True, dashes=False, err_style="bars", ci=68, palette=['green'], ax=ax)
    ax = sns.lineplot(data=df2,
                x="steps", y="Accuracy", hue="Label", style="Label",
                markers=True, dashes=False, err_style="bars", ci=68, palette=['purple'], ax=ax)
    # set the title and save the fig
    save_fig(args_dict, fig)

    return ax

def boxplot_ratio_accs(df, args_dict, fig, ax):
        analyze_name =  args_dict['analyze_name']
        sub_title = args_dict['sub_title']
        val_name, threshold = args_dict['val_name'], args_dict['threshold']
        mi, mx = args_dict['mi'], args_dict['mx']
        is_box_plot, is_accs_plot = args_dict['is_box_plot'], args_dict['is_accs_plot']
        df_r, df_r_a_tr = df

        if analyze_name == 'analyze_ttest':
            val_name = 'tvalues_ttest'
        if is_box_plot:
            ax = sns.boxplot(data=df_r, 
                            x='steps', y=val_name, ax=ax)
            ax = sns.stripplot(data=df_r, 
                            x='steps', y=val_name, ax=ax)
        else:
            ax = sns.lineplot(data=df_r,
                            x="steps", y=val_name, hue="Label", style="Label",
                            markers=True, dashes=False, err_style="bars", ci=68, palette=['green'], ax=ax)
        if is_accs_plot:
            if analyze_name == 'analyze_ttest':
                val_name = 'Ratio(cong/incong)'
                mi2, mx2 = 0.5, 2.8
                ax2 = ax.twinx()
                ax2 = sns.lineplot(data=df_r_a_tr,
                                x="steps", y=val_name, hue="Label", style="Label",
                                markers=True, dashes=False, err_style="bars", ci=68, palette=['darkblue'], ax=ax2)
                ax2.legend(loc='upper left')                
                ax2.set_ylim([mi2, mx2])
            else:
                ax = sns.lineplot(data=df_r_a_tr,
                                x="steps", y=val_name, hue="Label", style="Label",
                                markers=True, dashes=False, err_style="bars", ci=68, palette=['darkblue'], ax=ax)
        ax.set_ylim([mi, mx])
        ax.set_title(sub_title)
        if threshold is not None:
            ax.axhline(y=threshold, color='r', linewidth=2)
            if analyze_name == 'analyze_ttest':
               ax.axhline(y=-1*threshold, color='r', linewidth=2)
        # set the title and save the fig
        save_fig(args_dict, fig)

def plot_reg_1D(args_dict, params, pvals, fig, axs):
    # analyze_name =  args_dict['analyze_name']
    sub_title = args_dict['sub_title']
    # val_name = args_dict['val_name']
    mi, mx = args_dict['mi'], args_dict['mx']
    # params, pvals = args_dict['params'], args_dict['pvals']

    runs, checkpoints, ncoef = params.shape[0], params.shape[1], params.shape[-1]
    print('runs, checkpoints: ', runs, checkpoints)
    val_names=['Intercept', 'Ground Truth E', '1D Rank', 'Warped E']
    for coef in range(ncoef):
        if coef==0:
            # fig.delaxes(axs[coef])
            continue
        val_name = val_names[coef]
        df = pd.DataFrame(params[:,:,coef], columns= np.arange(checkpoints))
        df.insert(0, 'runs', np.arange(runs))
        df2 = pd.melt(df, id_vars=['runs'],var_name='steps', value_name=val_name)
        # plot
        ax = axs[coef-1]
        ax = sns.boxplot(x='steps', y=val_name, data=df2, ax=ax)
        ax = sns.stripplot(x='steps', y=val_name, data=df2, ax=ax)
        for run in range(runs):
            pval = pvals[run,:,coef]
            for i, p in enumerate(pval):
                s = '*' if p<0.05 else ' '
                ax.annotate(s, (i, params[run, i, coef]), color='r')
        for ax in axs.flatten():
            ax.set_ylim([mi, mx])
            ax.axhline(y=0, color='r', linewidth=2)
            ax.set_title(sub_title)

    save_fig(args_dict, fig)

        # if ctx_order is not None:
        #         fig.suptitle('Reg. 1D Results - %s - Ax %s' %(model_str, ctx_order), fontweight='bold', fontsize='25')
        # else:
        #         fig.suptitle('Reg. 1D Results - %s' %(model_str), fontweight='bold', fontsize='25')
        # plt.tight_layout()
        # fig_str = '%s_reg_1D_results_%s_hidds' %(ctx_order_str, mfig_str)
        # fig.savefig(('../../figures/' + fig_str + '.pdf'), 
        #         bbox_inches = 'tight', pad_inches = 0)
        # fig.savefig(('../../figures/' + fig_str + '.png'), 
        #         bbox_inches = 'tight', pad_inches = 0)


def save_fig(args_dict, fig):
    ctx_order, ctx_order_str = args_dict['ctx_order'], args_dict['ctx_order_str']
    model_str, mfig_str  = args_dict['model_str'], args_dict['mfig_str']
    savefig_str = args_dict['savefig_str']

    if ctx_order is not None:
            fig.suptitle('%s - Ax %s' %(model_str, ctx_order), fontweight='bold', fontsize='25')
    else:
            fig.suptitle('%s' %(model_str), fontweight='bold', fontsize='25')
    plt.tight_layout()
    plt.show()
    fig_str = '%s_%s_%s_hidds' %(ctx_order_str, savefig_str, mfig_str)
    fig.savefig(('../../figures/' + fig_str + '.pdf'), 
            bbox_inches = 'tight', pad_inches = 0)
    fig.savefig(('../../figures/' + fig_str + '.png'), 
            bbox_inches = 'tight', pad_inches = 0)










# def boxplot(df, args_dict, fig, ax):#, ctx_order, ctx_order_str, model_str, mfig_str, sub_title, fig, ax, savefig_str): 
#         analyze_name =  args_dict['analyze_name']
#         ctx_order, ctx_order_str = args_dict['ctx_order'], args_dict['ctx_order_str']
#         model_str, mfig_str  = args_dict['model_str'], args_dict['mfig_str']
#         savefig_str, sub_title = args_dict['savefig_str'], args_dict['sub_title']
#         val_name, threshold = args_dict['val_name'], args_dict['threshold']
#         mi, mx = args_dict['mi'], args_dict['mx']
#         # mi, mx = -8.5, 11
#         # runs, checkpoints = val.shape[0], val.shape[1]
#         # print('runs, checkpoints: ', runs, checkpoints)
#         # val_name = 'tvals_hidds'
#         # threshold = 1.96

#         # df = pd.DataFrame(val, columns= np.arange(checkpoints))
#         # df.insert(0, 'runs', np.arange(runs))
#         # df2 = pd.melt(df, id_vars=['runs'],var_name='steps', value_name=val_name)
#         # plot 
#         # ax = axs
#         ax = sns.boxplot(x='steps', y=val_name, data=df, ax=ax)
#         ax = sns.stripplot(x='steps', y=val_name, data=df, ax=ax)
#         ax.axhline(y=threshold, color='r', linewidth=2)
#         if analyze_name == 'analyze_ttest':
#             ax.axhline(y=-1*threshold, color='r', linewidth=2)
#         ax.set_ylim([mi, mx])
#         ax.set_title(sub_title)
#         if ctx_order is not None:
#                 fig.suptitle('%s - Ax %s' %(model_str, ctx_order), fontweight='bold', fontsize='25')
#         else:
#                 fig.suptitle('%s' %(model_str), fontweight='bold', fontsize='25')

#         plt.tight_layout()
#         plt.show()
#         # savefig_str = 'ttest_results'
#         fig_str = '%s_%s_%s_hidds' %(ctx_order_str, savefig_str, mfig_str)
#         fig.savefig(('../../figures/' + fig_str + '.pdf'), 
#                 bbox_inches = 'tight', pad_inches = 0)
#         fig.savefig(('../../figures/' + fig_str + '.png'), 
#                 bbox_inches = 'tight', pad_inches = 0)

# def plot_ratio_dists(df, args_dict, fig, ax):#, ctx_order, ctx_order_str, model_str, mfig_str, sub_title, fig, ax, savefig_str): 
#         analyze_name =  args_dict['analyze_name']
#         ctx_order, ctx_order_str = args_dict['ctx_order'], args_dict['ctx_order_str']
#         model_str, mfig_str  = args_dict['model_str'], args_dict['mfig_str']
#         savefig_str, sub_title = args_dict['savefig_str'], args_dict['sub_title']
#         val_name, threshold = args_dict['val_name'], args_dict['threshold']
#         mi, mx = args_dict['mi'], args_dict['mx']

#         # mi, mx = 0, 5
#         # val_name = "Ratio Dist. (cong/incong)"
        
#         ax = sns.lineplot(data=df,
#                                 x="steps", y=val_name, hue="Lesion", palette="flare",
#                                 marker="o", dashes=False, err_style="bars", ci=68, ax=ax)

#         ax.set_ylim([mi, mx])
#         ax.set_title(sub_title)
#         ax.axhline(y=threshold, color='r', linewidth=2)        
#         if ctx_order is not None:
#                 fig.suptitle('%s - Ax %s' %(model_str, ctx_order), fontweight='bold', fontsize='25')
#         else:
#                 fig.suptitle('%s' %(model_str), fontweight='bold', fontsize='25')

#         plt.tight_layout()
#         # savefig_str = 'ablation_ratio_dists_results'
#         fig_str = '%s_%s_%s' %(ctx_order_str, savefig_str, mfig_str)
#         fig.savefig(('../../figures/' + fig_str + '.pdf'), 
#                         bbox_inches = 'tight', pad_inches = 0)
#         fig.savefig(('../../figures/' + fig_str + '.png'), 
#                         bbox_inches = 'tight', pad_inches = 0)  
#         return ax
