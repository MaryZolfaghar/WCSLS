import argparse
import torch 
import torch.nn as nn
import pickle
import random
import numpy as np 

from dataset import get_loaders
from models import EpisodicSystem, CorticalSystem, RecurrentCorticalSystem
from train import train
from test import test
from analyze import analyze_episodic, analyze_cortical, analyze_cortical_mruns


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
parser.add_argument('--cortical_model', type=str,
                    help='Use a recurrent neural network (LSTM) or MLP for cortical system')
parser.add_argument('--image_dir', default='images/',
                    help='Path to directory containing face images')
parser.add_argument('--N_cortical', type=int, default=1000,
                    help='Number of steps for training cortical system')
parser.add_argument('--bs_cortical', type=int, default=32,
                    help='Minibatch size for cortical system')
parser.add_argument('--lr_cortical', type=float, default=0.001,
                    help='Learning rate for cortical system')
parser.add_argument('--nruns_cortical', type=int, default=10, # 20
                    help='Number of runs for cortical system')
parser.add_argument('--checkpoints', type=int, default=50, #50
                    help='Number of steps during training before analyzing the results')
parser.add_argument('--before_ReLU', action='store_true',
                    help='Whether use hidden reps. of MLP before the ReLU')
# parser.add_argument('--out_type', type=str, default='out_seq',
#                     help='Whether use the last hidd or the whole sequence to fed into the linear output')

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
                        n_episodes=args.N_episodic)
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
        cortical_results = []
        if args.cortical_model=='rnn':
            print('Cortical system is running with an LSTM')
            cortical_system = RecurrentCorticalSystem(use_images=args.use_images).to(device)
        elif args.cortical_model=='mlp':
            print('Cortical system is running with an MLP')
            cortical_system = CorticalSystem(use_images=args.use_images).to(device)
        data = get_loaders(batch_size=args.bs_cortical, meta=False,
                        use_images=args.use_images, image_dir=args.image_dir,
                        n_episodes=None)
        train_data, train_loader, test_data, test_loader, analyze_data, analyze_loader = data
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
        N = args.N_episodic if meta else args.N_cortical 
        i = 0
        done = False
        while not done:
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
                    f1, f2, ax, y, idx1, idx2 = batch # face1, face2, axis, y, index1, index2
                    f1 = f1.to(args.device)
                    f2 = f2.to(args.device)
                    ax = ax.to(args.device)
                    y = y.to(args.device).squeeze(1)
                    y_hat, out = model(f1, f2, ax)
                    
                # Loss
                loss = loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()
                # Record loss
                train_losses.append(loss.data.item())
                ave_loss.append(loss.data.item())

                if i % args.print_every == 0:
                    print("Run: {}, Step: {}, Loss: {}".format(run, i, np.mean(ave_loss)))
                    ave_loss = []
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
                    cortical_results.append(cortical_result)
                    
                if i >= N:
                    done = True 
                    break
                i += 1
        print('num of checkpoints: ', len(cortical_results))
        cortical_runs.append(cortical_results)
        print("Cortical system training accuracy:", cortical_train_acc)
        print("Cortical system testing accuracy:", cortical_test_acc)
        print("Cortical system analyzing accuracy:", cortical_analyze_acc)

    print('num of runs: ', len(cortical_runs))
    with open('results/'+'mruns_'+args.out_file, 'wb') as f:
        pickle.dump(cortical_runs, f)
    print('don saving')
    # cortical_train_losses, cortical_system = train(meta, cortical_system, train_loader, args)
    cortical_train_losses = train_losses
    cortical_train_acc, _ = test(meta, cortical_system, train_loader, args)
    cortical_test_acc, _ = test(meta, cortical_system, test_loader, args)
    print('don testing')
    cortical_system.analyze=True
    cortical_analyze_acc, cortical_analyze_correct = test(meta, cortical_system, analyze_loader, args)
    cortical_system.analyze=False
    print('don testing 2')
    cortical_results = analyze_cortical_mruns(cortical_runs, test_data, args)
    print('don analyzing')
    cortical_results = {'loss': cortical_train_losses,
                        'train_acc': cortical_train_acc,
                        'test_acc': cortical_test_acc,
                        'analyze_acc': cortical_analyze_acc,
                        'analyze_correct': cortical_analyze_correct,
                        'analysis': cortical_results,
                        'cortical_runs': cortical_runs}
    if not args.use_em:
        episodic_results = {}
    # Save results
    results = {'Episodic' : episodic_results,
            'Cortical' : cortical_results}
    with open('results/'+args.out_file, 'wb') as f:
        pickle.dump(results, f)
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
