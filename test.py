import numpy as np
import torch 
from itertools import permutations

def test(meta, model, loader, args):
    model.eval()
    with torch.no_grad():
        correct = []
        correct1, correct2 = ([] for i in range(2))
        for batch in loader:
            if meta:
                m, x_ = batch
                m = m.to(args.device) # [batch, n_train, sample (with y)]
                x = x_[:,:,:-1].to(args.device) # [batch, n_test, input (no y)]
                y = x_[:,:,-1].type(torch.long).to(args.device)
                y = y.squeeze() # y: [batch, n_test]
                y_hat, attention = model(x, m) # yhat: [batch, n_test, 2]
                preds = torch.argmax(y_hat, dim=2).squeeze(0) # [n_test]
            else:
                if args.cortical_task == 'face_task':
                        f1, f2, ctx, y, idx1, idx2 = batch # face1, face2, context, y, index1, index2
                elif args.cortical_task == 'wine_task':
                        f1, f2, ctx, y1, y2, idx1, idx2 = batch # face1, face2, context, y1, y2, index1, index2
                f1 = f1.to(args.device)
                f2 = f2.to(args.device)
                ctx = ctx.to(args.device)
                y_hat, out = model(f1, f2, ctx) # [batch=1, seq_length, 2]
                if (args.N_responses == 'one'):
                        if args.cortical_task == 'wine_task':
                            y = y1
                if model.analyze:
                    # Always batch_size is 1  
                    if (args.N_responses == 'one'):       
                        y = y.cpu().numpy()
                        y = [y, y, y]
                        y = torch.tensor(y).to(args.device).squeeze().unsqueeze(0)
                        preds = torch.argmax(y_hat, dim=-1) # [batch=1, seq_length]
                    elif args.N_responses == 'two':
                        y1 = y1.cpu().numpy()
                        y1 = [y1, y1, y1]
                        y1 = torch.tensor(y1).to(args.device).squeeze().unsqueeze(0)
                        y2 = y2.cpu().numpy()
                        y2 = [y2, y2, y2]
                        y2 = torch.tensor(y2).to(args.device).squeeze().unsqueeze(0)
                        y_hat1 = y_hat[0]
                        y_hat2 = y_hat[1]
                        preds1 = torch.argmax(y_hat1, dim=-1) # [batch=1, seq_length]
                        preds2 = torch.argmax(y_hat2, dim=-1) # [batch=1, seq_length]
                else:
                    if (args.N_responses == 'one'):
                        y = y.to(args.device).squeeze()
                        y_hat, out = model(f1, f2, ctx) # [batch, 2]
                        preds = torch.argmax(y_hat, dim=1).squeeze(0) # [batch]
                    elif args.N_responses == 'two':
                        y1 = y1.to(args.device).squeeze()
                        y2 = y2.to(args.device).squeeze()
                        y_hat, out = model(f1, f2, ctx) # [batch, 2]
                        preds1 = torch.argmax(y_hat[0], dim=1).squeeze(0) # [batch]
                        preds2 = torch.argmax(y_hat[1], dim=1).squeeze(0) # [batch]

            if args.N_responses == 'one':    
                c = (preds == y)
                c = c.cpu().numpy().tolist()
                correct += c
            elif args.N_responses == 'two':
                c1 = (preds1 == y1)
                c1 = c1.cpu().numpy().tolist()
                c2 = (preds2 == y2)
                c2 = c2.cpu().numpy().tolist()
                correct1 += c1
                correct2 += c2
                correct = [correct1, correct2]
    if args.N_responses == 'one':
        acc = np.mean(correct, axis=0)
    elif args.N_responses == 'two':
        acc1 = np.mean(correct1, axis=0)
        acc2 = np.mean(correct2, axis=0)
        acc = [acc1, acc2]
    return acc, correct