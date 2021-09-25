import argparse
from numpy.lib.function_base import gradient
import torch 
import torch.nn as nn
import pickle
import random
import numpy as np 

from dataset import get_loaders
from models import *
from train import train
from test import test
from run_analyze import run_analyze
from utils.analyze import *
from utils.util import *

# To ignore printing all the RuntimeWarnings
# Since, we are aware that we have "divide by NaN". 
np.seterr(divide='ignore', invalid='ignore')

parser = argparse.ArgumentParser()
# Setup
parser.add_argument('--use_cuda', action='store_true',
                    help='Use GPU, if available')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--print_every', type=int, default=200,
                    help='Number of steps before printing average loss')
parser.add_argument('--out_file', default='results.P')
# Episodic memory system
parser.add_argument('--N_episodic', type=int, default=1000,
                    help='Number of steps for pre-training episodic system')
parser.add_argument('--use_em', action='store_true',
                    help='whether using the Episodic Memory System')                    
parser.add_argument('--bs_episodic', type=int, default=16,
                    help='Minibatch size for episodic system')
parser.add_argument('--lr_episodic', type=float, default=0.001,
                    help='Learning rate for episodic system')
# Cortical system
parser.add_argument('--use_images', action='store_true',
                    help='Use full face images and CNN for cortical system')
parser.add_argument('--cortical_model', type=str, default='rnn',
                    help='Use a recurrent neural network (LSTM) or MLP for cortical system')
parser.add_argument('--cortical_task', type=str, default='face_task',
                    help='The task for the cortical model - either face_task or wine_task')
parser.add_argument('--image_dir', default='images/',
                    help='Path to directory containing face images')
parser.add_argument('--N_cortical', type=int, default=6000,
                    help='Number of steps for training cortical system')
parser.add_argument('--balanced', action='store_true', # ToDo:change this to store_true
                    help='Balance wins and losses of each face other than (0,0), (3,3). Only works with wine dataset')                 
parser.add_argument('--bs_cortical', type=int, default=1,
                    help='Minibatch size for cortical system')
parser.add_argument('--lr_cortical', type=float, default=0.001,
                    help='Learning rate for cortical system')
parser.add_argument('--nruns_cortical', type=int, default=2,
                    help='Number of runs for cortical system')
parser.add_argument('--checkpoints', type=int, default=50, #50 # the name is confusing, change to something like checkpoint_every or cp_every 
                    help='Number of steps during training before analyzing the results')
parser.add_argument('--truncated_mlp', type=str, default='false',
                    help='Whether to detach the first hidden layer in the Stepwise MLP model')
parser.add_argument('--order_ctx', type=str, default='first',
                    help='Use context/axis as first or last input in the recurrent cortical system')
parser.add_argument('--N_responses', type=str, default='one',
                    help='How many responses to perform - multitasking')
parser.add_argument('--N_contexts', type=int, default=2,
                    help='Number of contexts')
parser.add_argument('--dimred_method', type=str, default='pca',
                    help='Dimentionality reduction method')
parser.add_argument('--measure_grad_norm', action='store_true',
                    help='Whether measuring the gradient w.r.t the inputs')
parser.add_argument('--is_lesion', action='store_true',
                    help='Ablate context')
parser.add_argument('--lesion_p', type=float, default=0.1,
                    help='Lesion probability')
parser.add_argument('--sbs_analysis', action='store_true',
                    help='Whether analyzing step by step')
parser.add_argument('--sbs_every', type=int, default=1, 
                    help='Number of steps during training before analyzing step-by-step')



def main(args):
    # CUDA
    if args.use_cuda:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
    else:
        use_cuda = False
        device = "cpu"
    args.device = device
    print("Using CUDA: ", use_cuda)

    # Random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.use_em:
        # Episodic memory system: Pre-train, test, analyze (hub retrieval)
        meta = True # meta-learning for episodic memory system
        episodic_system = EpisodicSystem().to(device)
        data = get_loaders(batch_size=args.bs_episodic, meta=meta, 
                        use_images=False, image_dir=args.image_dir, 
                        n_episodes=args.N_episodic, 
                        N_responses=None, N_contexts=None, cortical_task=None)
        train_data, train_loader, test_data, test_loader = data
        episodic_train_losses = train(meta, episodic_system, train_loader, args)
        episodic_train_acc = test(meta, episodic_system, train_loader, args)
        episodic_test_acc = test(meta, episodic_system, test_loader, args)
        episodic_analysis = analyze_episodic(episodic_system, test_data, args)
        print("Episodic system training accuracy:", episodic_train_acc)
        print("Episodic system testing accuracy:", episodic_test_acc)
        episodic_results = {'loss' : episodic_train_losses,
                            'train_acc': episodic_train_acc,
                            'test_acc' : episodic_test_acc,
                            'analysis' : episodic_analysis}

    # Cortical system: Train, test, analyze (PCA, correlation)
    meta = False # cortical learning is vanilla
    cortical_runs = []
    # cortical_runs_bef_c, cortical_runs_aft_c = [] , []
    # cortical_runs_bef_inc, cortical_runs_aft_inc = [] , [] 
    congruencies = []
    cortical_runs_ratio_diffs = [] 
    for run in range(args.nruns_cortical):
        n_gradient_ctx, n_gradient_f1, n_gradient_f2 = [], [], []
        n_gradient_ctx_cong, n_gradient_f1_cong, n_gradient_f2_cong = [], [], []
        n_gradient_ctx_incong, n_gradient_f1_incong, n_gradient_f2_incong = [], [], []
        cortical_run = []
        # one list for cong/incong
        cortical_run_ratio_diffs = [] # tuples of (diff, cong/incong)
        # cortical_run_bef_c, cortical_run_aft_c = [] , [] 
        # cortical_run_bef_inc, cortical_run_aft_inc = [] , [] 
        if args.cortical_model=='rnn':
            print('Cortical system is running with an LSTM')
            cortical_system = RecurrentCorticalSystem(args)
            # Use context/axis as the last one in the sequence
            cortical_system.order_ctx = args.order_ctx
        elif args.cortical_model=='mlp':
            print('Cortical system is running with an MLP')
            cortical_system = CorticalSystem(args)
        elif args.cortical_model=='stepwisemlp':
            print('Cortical system is running with a StepwiseMLP')
            cortical_system = StepwiseCorticalSystem(args) 
        elif args.cortical_model=='rnncell':
            print('Cortical system is running with a RNNCell')
            cortical_system = RNNCell(args)
        elif args.cortical_model == 'mlp_cc': # mlp as a cognitive controller
            print('Cortical system is running with a CognitiveController')
            cortical_system = CognitiveController(args)

                       
        cortical_system.to(device)

        data = get_loaders(batch_size=args.bs_cortical, meta=False,
                          use_images=args.use_images, image_dir=args.image_dir,
                          n_episodes=None, N_responses=args.N_responses, 
                          N_contexts=args.N_contexts, 
                          cortical_task=args.cortical_task,
                          balanced = args.balanced)
        train_data, train_loader, test_data, test_loader, analyze_data, analyze_loader = data
        args.loc2idx = test_data.loc2idx
        # model
        model = cortical_system
        model.train()
        # loss function
        loss_fn = nn.CrossEntropyLoss()
        loss_fn.to(args.device)

        # optimizer
        lr = args.lr_episodic if meta else args.lr_cortical
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # training loop
        train_losses = [] # for recording all train losses
        ave_loss = [] # running average loss for printing
        if args.N_responses=='two':
            train_losses1 = []
            train_losses2 = []
            ave_loss1 = []
            ave_loss2 = []

        N = args.N_episodic if meta else args.N_cortical 
        i = 0
        done = False
        while not done:
            # Todo: check this for batch in analyze_loader:
            for batch in train_loader:
                optimizer.zero_grad()
                if meta:
                    m, x_ = batch
                    m = m.to(args.device) # [batch, n_train, sample (with y)]
                    x = x_[:,:,:-1].to(args.device) # [batch, n_test, input (no y)]
                    y = x_[:,:,-1].type(torch.long).to(args.device) 
                    # y: [batch, n_test, 1]
                    y_hat, attention = model(x, m) # yhat: [batch, n_test, 2]
                    y_hat = y_hat.view(-1, y_hat.shape[2]) # [batch*n_test, 2]
                    y = y.view(-1) # [batch*n_test]
                else:
                    if args.cortical_task == 'face_task':
                        f1, f2, ctx, y, idx1, idx2 = batch # face1, face2, context, y, index1, index2
                        y = y.to(args.device).squeeze(1)
                    elif args.cortical_task == 'wine_task':
                        f1, f2, ctx, y1, y2, idx1, idx2 = batch # face1, face2, context, y1, y2, index1, index2
                        y1 = y1.to(args.device).squeeze(1) # [batch]
                        y2 = y2.to(args.device).squeeze(1) # [batch]
                    f1 = f1.to(args.device)
                    f2 = f2.to(args.device)
                    ctx = ctx.to(args.device)
                    y_hat, out = model(f1, f2, ctx)
                    if (args.N_responses == 'one'):
                        if args.cortical_task == 'wine_task':
                            y = y1
                        loss = loss_fn(y_hat, y) # if reduction = 'none' : [batch], if reduction='mean': scaler
                    if args.N_responses == 'two':
                        y_hat1 = y_hat[0] # [batch, 2]
                        y_hat2 = y_hat[1] # [batch, 2]
                        loss1 = loss_fn(y_hat1, y1)
                        loss2 = loss_fn(y_hat2, y2)
                        loss = loss1 + loss2
                        train_losses1.append(loss1.data.item())
                        train_losses2.append(loss2.data.item())
                        ave_loss1.append(loss1.data.item())
                        ave_loss2.append(loss2.data.item())

                if args.measure_grad_norm:
                    model.f1_embed.retain_grad()
                    model.f2_embed.retain_grad()
                    model.ctx_embed.retain_grad()

                loss.backward()
                # Record loss
                train_losses.append(loss.data.item())
                ave_loss.append(loss.data.item())
                
                if args.measure_grad_norm:
                    n_grd_ctx = torch.linalg.norm(model.ctx_embed.grad, dim=1)
                    n_grd_f1 = torch.linalg.norm(model.f1_embed.grad, dim=1)
                    n_grd_f2 = torch.linalg.norm(model.f2_embed.grad, dim=1)

                    n_gradient_ctx.append(n_grd_ctx.numpy())
                    n_gradient_f1.append(n_grd_f1.numpy())
                    n_gradient_f2.append(n_grd_f2.numpy())

                    for ii, (i1, i2) in enumerate(zip(idx1, idx2)):
                        # 1: congruent, -1:incongruent, 0:none
                        cong = get_congruency(args, i1, i2)
                        congruencies.append(cong)
                        if cong==1:
                            n_gradient_ctx_cong.append(n_grd_ctx[ii].numpy())
                            n_gradient_f1_cong.append(n_grd_f1[ii].numpy())
                            n_gradient_f2_cong.append(n_grd_f2[ii].numpy())
                        if cong==-1:
                            n_gradient_ctx_incong.append(n_grd_ctx[ii].numpy())
                            n_gradient_f1_incong.append(n_grd_f1[ii].numpy())
                            n_gradient_f2_incong.append(n_grd_f2[ii].numpy())
                # -------------------------------------------------------------------------------
                # analyze before taking a step
                # -------------------------------------------------------------------------------
                if args.sbs_analysis:
                    if i % args.sbs_every == 0:
                        cortical_result_b = analyze_cortical(cortical_system, test_data, analyze_loader, args)
                        dist_result_b      = calc_dist(args, test_data, cortical_result_b, dist_results=None)    
                        ratio_b           = calc_ratio(args, test_data, cortical_result_b, dist_result_b)
                        
                # -------------------------------------------------------------------------------
                # take a step
                # -------------------------------------------------------------------------------
                optimizer.step()
                
                if i % args.print_every == 0:
                    if args.N_responses == 'two':
                        print("Run: {}, Step: {}, Loss: {}, Loss1: {}, Loss2: {}".format(run, i, np.mean(ave_loss),
                                                                                        np.mean(ave_loss1), 
                                                                                        np.mean(ave_loss2)))
                    else:
                        print("Run: {}, Step: {}, Loss: {}".format(run, i, np.mean(ave_loss)))
                    ave_loss = []
                    ave_loss1 = []
                    ave_loss2 = []
                # -------------------------------------------------------------------------------
                # analyze after taking a step
                # -------------------------------------------------------------------------------
                if args.sbs_analysis:
                    if i % args.sbs_every == 0:
                        # 1: congruent, -1:incongruent, 0:none
                        cong = get_congruency(args, idx1[0], idx2[0])
                        # get rid of the accs stuff here
                        # get these: cortical_run_b, cortical_run_a, cong/incong, 
                        # calc_dist(cortical_run_b), calc_dist(cortical_run_a)
                        # ratio_b = calc_ratio(cortical_run_b), ratio_a = calc_ratio(cortical_run_a)
                        # diff = raio_a - ratio_b
                        # scalar cong, incong

                        cortical_result_a = analyze_cortical(cortical_system, test_data, analyze_loader, args)
                        dist_result_a      = calc_dist(args, test_data, cortical_result_a, dist_results=None)    
                        ratio_a           = calc_ratio(args, test_data, cortical_result_a, dist_result_a)
                        diff_ratio = ratio_a['ratio_embed'] - ratio_b['ratio_embed']
                        cortical_run_ratio_diffs.append((diff_ratio, cong))

                # -------------------------------------------------------------------------------
                # analyze after taking a step - each checkpoint
                # -------------------------------------------------------------------------------
                if i % args.checkpoints == 0: 
                    cortical_train_acc, _ , cong_train_acc, incong_train_acc = test(meta, cortical_system, train_loader, args)
                    cortical_test_acc, _, cong_test_acc, incong_test_acc  = test(meta, cortical_system, test_loader, args)
                    cortical_system.analyze=True
                    cortical_analyze_acc, cortical_analyze_correct, _, _ = test(meta, cortical_system, analyze_loader, args)
                    cortical_system.analyze=False
                    cortical_result = analyze_cortical(cortical_system, test_data, analyze_loader, args)
                    cortical_result['train_acc'] = cortical_train_acc
                    cortical_result['test_acc'] = cortical_test_acc
                    cortical_result['cong_train_acc'] = cong_train_acc
                    cortical_result['incong_train_acc'] = incong_train_acc
                    cortical_result['cong_test_acc'] = cong_test_acc
                    cortical_result['incong_test_acc'] = incong_test_acc
                    cortical_result['analyze_acc'] = cortical_analyze_acc
                    cortical_result['analyze_correct'] = cortical_analyze_correct
                    
                    if args.measure_grad_norm:
                        cortical_result['grad_ctx'] = np.mean(n_gradient_ctx)
                        cortical_result['grad_f1'] = np.mean(n_gradient_f1)
                        cortical_result['grad_f2'] = np.mean(n_gradient_f2)

                        cortical_result['grad_ctx_cong'] = np.mean(n_gradient_ctx_cong)
                        cortical_result['grad_f1_cong'] = np.mean(n_gradient_f1_cong)
                        cortical_result['grad_f2_cong'] = np.mean(n_gradient_f2_cong)

                        cortical_result['grad_ctx_incong'] = np.mean(n_gradient_ctx_incong)
                        cortical_result['grad_f1_incong'] = np.mean(n_gradient_f1_incong)
                        cortical_result['grad_f2_incong'] = np.mean(n_gradient_f2_incong)
                        
                        n_gradient_ctx, n_gradient_f1, n_gradient_f2 = [], [], []
                        n_gradient_ctx_cong, n_gradient_f1_cong, n_gradient_f2_cong = [], [], []
                        n_gradient_ctx_incong, n_gradient_f1_incong, n_gradient_f2_incong = [], [], []

                    cortical_run.append(cortical_result)

                if i >= N:
                    done = True 
                    break
                i += 1
        cortical_runs.append(cortical_run)
        cortical_runs_ratio_diffs.append(cortical_run_ratio_diffs)
        
        print("Cortical system training accuracy:", cortical_train_acc)
        print("Cortical system testing accuracy:", cortical_test_acc)
        print("Cortical system analyzing accuracy:", cortical_analyze_acc)

    # End run
    print('num of checkpoints: ', len(cortical_run))
    args.ncheckpoints_cortical = len(cortical_run)
    print('num of runs: ', len(cortical_runs))
    # cortical_train_losses, cortical_system = train(meta, cortical_system, train_loader, args)
    cortical_train_losses = train_losses
    cortical_train_acc, _, _, _ = test(meta, cortical_system, train_loader, args)
    cortical_test_acc, _, _, _ = test(meta, cortical_system, test_loader, args)
    
    cortical_mrun_results = run_analyze(args, test_data, cortical_runs)

    cortical_results = {'loss': cortical_train_losses,
                        'train_acc': cortical_train_acc,
                        'test_acc': cortical_test_acc,
                        'analyze_acc': cortical_analyze_acc,
                        'analyze_correct': cortical_analyze_correct,
                        'analysis': cortical_mrun_results}
    if args.sbs_analysis:
        # 20 runs, each list of tuples (diff, congruency), save these
        # 20 sets of two lines(for cong and incong), diff x
        cortical_results['cortical_runs_ratio_diffs'] = cortical_runs_ratio_diffs
    
    if not args.use_em:
        episodic_results = {}
    
    results = {'Episodic' : episodic_results,
               'Cortical' : cortical_results}
    
    with open('../results/'+args.out_file, 'wb') as f:
        pickle.dump(results, f)
# only ttest, ratio for sbs
if __name__ == '__main__':
    args = parser.parse_args()

    # for step-by-step analysis, since it is computationally expensive
    # we only do one analysis (ratio) for it 
    if args.sbs_analysis:
        analysis_names = ['calc_ratio']
        analysis_funcs = [calc_ratio]
    else:
        analysis_names = ['analyze_accs','hist_data', 'calc_ratio', \
                      'analyze_dim_red', 'analyze_ttest', 'analyze_corr', \
                      'analyze_regression', 'analyze_regression_1D', \
                      'analyze_regression_exc', 'analyze_test_seq', 'proportions']
        analysis_funcs = [analyze_accs, hist_data, calc_ratio, \
                      analyze_dim_red, analyze_ttest, analyze_corr, \
                      analyze_regression, analyze_regression_1D, \
                      analyze_regression_exc, analyze_test_seq, proportions]
    
    if args.measure_grad_norm:
        analysis_names.append('analyze_credit_assignment')
        analysis_funcs.append(analyze_credit_assignment)
    
    args.analysis_names = analysis_names
    args.analysis_funcs = analysis_funcs

    print(args)
    main(args)