import argparse
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
parser.add_argument('--use_images', action='store_false',
                    help='Use full face images and CNN for cortical system')
parser.add_argument('--cortical_model', type=str, default='mlp',
                    help='Use a recurrent neural network (LSTM) or MLP for cortical system')
parser.add_argument('--cortical_task', type=str, default='face_task',
                    help='The task for the cortical model - either face_task or wine_task')
parser.add_argument('--image_dir', default='images/',
                    help='Path to directory containing face images')
parser.add_argument('--N_cortical', type=int, default=1000,
                    help='Number of steps for training cortical system')
parser.add_argument('--bs_cortical', type=int, default=32,
                    help='Minibatch size for cortical system')
parser.add_argument('--lr_cortical', type=float, default=0.001,
                    help='Learning rate for cortical system')
parser.add_argument('--nruns_cortical', type=int, default=2, # 20
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
    # train_acc_runs, test_acc_runs = [], []
    for run in range(args.nruns_cortical):
        cortical_run = []
        # train_acc_run, test_acc_run = [], []
        if args.cortical_model=='rnn':
            print('Cortical system is running with an LSTM')
            cortical_system = RecurrentCorticalSystem(use_images=args.use_images, N_contexts=args.N_contexts)
            # Use context/axis as the last one in the sequence
            cortical_system.order_ctx = args.order_ctx
        elif args.cortical_model=='mlp':
            print('Cortical system is running with an MLP')
            cortical_system = CorticalSystem(use_images=args.use_images, 
                                             N_responses=args.N_responses,
                                             N_contexts=args.N_contexts)
        elif args.cortical_model=='stepwisemlp':
            print('Cortical system is running with a StepwiseMLP')
            cortical_system = StepwiseCorticalSystem(use_images=args.use_images) 
            cortical_system.truncated_mlp = args.truncated_mlp
        elif args.cortical_model=='rnncell':
            print('Cortical system is running with a RNNCell')
            cortical_system = RNNCell(use_images=args.use_images, 
                                      N_contexts=args.N_contexts)
                       
        cortical_system.to(device)

        data = get_loaders(batch_size=args.bs_cortical, meta=False,
                          use_images=args.use_images, image_dir=args.image_dir,
                          n_episodes=None, N_responses=args.N_responses, 
                          N_contexts=args.N_contexts, 
                          cortical_task=args.cortical_task)
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
                        loss = loss_fn(y_hat, y)
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
                
                loss.backward()
                optimizer.step()
                # Record loss
                train_losses.append(loss.data.item())
                ave_loss.append(loss.data.item())

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
                    cortical_run.append(cortical_result)

                if i >= N:
                    done = True 
                    break
                i += 1
        cortical_runs.append(cortical_run)
        print("Cortical system training accuracy:", cortical_train_acc)
        print("Cortical system testing accuracy:", cortical_test_acc)
        print("Cortical system analyzing accuracy:", cortical_analyze_acc)

    # End run
    print('num of checkpoints: ', len(cortical_run))
    print('num of runs: ', len(cortical_runs))
    # cortical_train_losses, cortical_system = train(meta, cortical_system, train_loader, args)
    cortical_train_losses = train_losses
    cortical_train_acc, _, _, _ = test(meta, cortical_system, train_loader, args)
    cortical_test_acc, _, _, _ = test(meta, cortical_system, test_loader, args)
    # cortical_system.analyze=True
    # cortical_analyze_acc, cortical_analyze_correct = test(meta, cortical_system, analyze_loader, args)
    # cortical_system.analyze=False

    cortical_mrun_results = run_analyze(args, test_data, cortical_runs)
    cortical_results = {'loss': cortical_train_losses,
                        'train_acc': cortical_train_acc,
                        'test_acc': cortical_test_acc,
                        'analyze_acc': cortical_analyze_acc,
                        'analyze_correct': cortical_analyze_correct,
                        'analysis': cortical_mrun_results}
    if not args.use_em:
        episodic_results = {}
    
    results = {'Episodic' : episodic_results,
               'Cortical' : cortical_results}
    
    with open('../results/'+args.out_file, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    args = parser.parse_args()
    analysis_names = ['analyze_accs', 'hist_data', 'calc_ratio', \
                      'analyze_dim_red', 'analyze_ttest', 'analyze_corr', \
                      'analyze_regression', 'analyze_regression_1D', \
                      'analyze_regression_exc', 'analyze_test_seq', 'proportions']

    analysis_funcs = [analyze_accs, hist_data, calc_ratio, \
                      analyze_dim_red, analyze_ttest, analyze_corr, \
                      analyze_regression, analyze_regression_1D, \
                      analyze_regression_exc, analyze_test_seq, proportions]
    
    args.analysis_names = analysis_names
    args.analysis_funcs = analysis_funcs

    print(args)
    main(args)

    # ToDo: the training day experiment is not complete yet, I only changed the dataset.py
    # I need to change the main.py as well, I created the main_V2.py and changed train.py 
    # The issue is in the main_V2 there is a loop, and the optimizer stuff should be zero_grade outside that loop
    # But now I have train inside the loop and zero_grade stuff happens inside the loop which is wrong, 
    # parser.add_argument('--training_day', type=str, default='day3',
                        # help='Training on day1 or day1_day2 or day3 (all the training data')
