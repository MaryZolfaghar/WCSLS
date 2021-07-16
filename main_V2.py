import argparse
import torch 
import torch.nn as nn
import pickle
import json
import random
import numpy as np 

from dataset import get_loaders
from models import EpisodicSystem, CorticalSystem, RecurrentCorticalSystem
from train import train
from test import test
from analyze import analyze_episodic, analyze_cortical, analyze_cortical_mruns
#

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
parser.add_argument('--cortical_task', type=str, default='wine_task',
                    help='The task for the cortical model - either face_task or wine_task')
parser.add_argument('--image_dir', default='images/',
                    help='Path to directory containing face images')
parser.add_argument('--N_cortical', type=int, default=1000,
                    help='Number of steps for training cortical system')
parser.add_argument('--bs_cortical', type=int, default=32,
                    help='Minibatch size for cortical system')
parser.add_argument('--lr_cortical', type=float, default=0.001,
                    help='Learning rate for cortical system')
parser.add_argument('--nruns_cortical', type=int, default=1, # 20
                    help='Number of runs for cortical system')
parser.add_argument('--checkpoints', type=int, default=50, #50 # the name is confusing, change to something like checkpoint_every or cp_every 
                    help='Number of steps during training before analyzing the results')
parser.add_argument('--analysis_type', type=str, default='all',
                    help='What analysis to do after the multiple runs')
parser.add_argument('--order_ctx', type=str, default='first',
                    help='Use context/axis as first or last input in the recurrent cortical system')
parser.add_argument('--N_responses', type=str, default='one',
                    help='How many responses to perform - multitasking')
parser.add_argument('--N_contexts', type=int, default=8,
                    help='Number of contexts')
parser.add_argument('--training_regime', type=str, default='all_days',
                    help='Training on separate_days or on all_days')
parser.add_argument('--training_day', type=str, default='day3',
                    help='Training on day1 or day1_day2 or day3 (all the training data')

                    
# this is for the experiemnt with different training regime
# Train on day1
# Train on day1 + day2 : training dataset without the hubs
# Train on day1 + day2 + day3 : whole training dataset
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
                        N_responses=None, N_contexts=None, cortical_task=None,
                        training_regime=None)
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
    for run in range(args.nruns_cortical):
        cortical_run = []
        if args.cortical_model=='rnn':
            print('Cortical system is running with an LSTM')
            cortical_system = RecurrentCorticalSystem(use_images=args.use_images).to(device)
            # Use context/axis as the last one in the sequence
            cortical_system.order_ctx = args.order_ctx
            
        elif args.cortical_model=='mlp':
            print('Cortical system is running with an MLP')
            cortical_system = CorticalSystem(use_images=args.use_images, 
                                             N_responses=args.N_responses,
                                             N_contexts=args.N_contexts)
        
        cortical_system.to(device)

        data = get_loaders(batch_size=args.bs_cortical, meta=False,
                          use_images=args.use_images, image_dir=args.image_dir,
                          n_episodes=None, N_responses=args.N_responses, 
                          N_contexts=args.N_contexts, 
                          cortical_task=args.cortical_task,
                          training_day=args.training_regime)
        train_data, train_loader, test_data, test_loader, analyze_data, analyze_loader = data
        # model
        model = cortical_system

        # model.train()
        # loss function
        # loss_fn = nn.CrossEntropyLoss()
        # loss_fn.to(args.device)

        # optimizer
        # lr = args.lr_episodic if meta else args.lr_cortical
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
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
            loss, loss1, loss2, model = train(meta, model, train_loader, args)
                
            if args.N_responses == 'two':
                train_losses1.append(loss1.data.item())
                train_losses2.append(loss2.data.item())
                ave_loss1.append(loss1.data.item())
                ave_loss2.append(loss2.data.item())
                
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
                cortical_train_acc, _ = test(meta, cortical_system, train_loader, args)
                cortical_test_acc, _ = test(meta, cortical_system, test_loader, args)
                cortical_system.analyze=True
                cortical_analyze_acc, cortical_analyze_correct = test(meta, cortical_system, analyze_loader, args)
                cortical_system.analyze=False
                cortical_result = analyze_cortical(cortical_system, test_data, analyze_loader, args)
                cortical_result['train_acc'] = cortical_train_acc
                cortical_result['test_acc'] = cortical_test_acc
                cortical_result['analyze_acc'] = cortical_analyze_acc,
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
    print('done saving')
    # cortical_train_losses, cortical_system = train(meta, cortical_system, train_loader, args)
    cortical_train_losses = train_losses
    cortical_train_acc, _ = test(meta, cortical_system, train_loader, args)
    cortical_test_acc, _ = test(meta, cortical_system, test_loader, args)
    print('done testing')
    cortical_system.analyze=True
    cortical_analyze_acc, cortical_analyze_correct = test(meta, cortical_system, analyze_loader, args)
    cortical_system.analyze=False
    print('done second testing')
    cortical_mrun_results = analyze_cortical_mruns(cortical_runs, test_data, args)
    print('done analyzing')
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
    print('done saving cortical_results')

    # with open('../results/'+'mruns_'+args.out_file, 'wb') as f:
    #     pickle.dump(cortical_runs, f)
    # print('done saving cortical_mruns')

if __name__ == '__main__':
    args = parser.parse_args()
    
    print(args)
    main(args)
