import numpy as np
import torch
import torch.nn as nn


def train(meta, model, loader, args):
    # model
    model.train()

    # loss function
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(args.device)

    # optimizer
    lr = args.lr_episodic if meta else args.lr_cortical
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for batch in loader:
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
                loss1 = loss2 = []
            if args.N_responses == 'two':
                y_hat1 = y_hat[0] # [batch, 2]
                y_hat2 = y_hat[1] # [batch, 2]
                loss1 = loss_fn(y_hat1, y1)
                loss2 = loss_fn(y_hat2, y2)
                loss = loss1 + loss2

        loss.backward()
        optimizer.step()

    return loss, loss1, loss2, model


    
