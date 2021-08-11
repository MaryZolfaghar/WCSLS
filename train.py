import numpy as np
import torch
import torch.nn as nn


def train(meta, model, loader, args):
    # model
    model.train()

    # loss function
    # reduction = 'none', them you have a vecot
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(args.device)

    # optimizer
    lr = args.lr_episodic if meta else args.lr_cortical
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n_gradient_ctx, n_gradient_f1, n_gradient_f2 = [], [], []
    
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
                # ToDo: Investigate learning about context
                # by measuring the norm of the gradient w.r.t. the context 
                batch_size = y_hat[0]
                # # jacT = torch.zeros(1,batch_size)
                # gradient is a vector (state_dim (ctx)), no longer a matrix, loss is the sum of each sample in the batch
                # one num per step (norm)
                for i in range(batch_size):
                    output = torch.zeros(batch_size,1)                                                                                                          
                    output[i] = 1.
                    # x = model.ctx_embed
                    y = loss
                    grd = torch.autograd.grad(y, model.ctx_embed, grad_outputs=output, retain_graph=True)[0]
                    n_grd  = torch.linalg.norm(grd)
                    n_gradient_ctx.append(n_grd.numpy())

                    grd = torch.autograd.grad(y, model.f1_embed, grad_outputs=output, retain_graph=True)[0]
                    n_grd  = torch.linalg.norm(grd)
                    n_gradient_f1.append(n_grd.numpy())

                    grd = torch.autograd.grad(y, model.f2_embed, grad_outputs=output, retain_graph=True)[0]
                    n_grd  = torch.linalg.norm(grd)
                    n_gradient_f2.append(n_grd.numpy())
                    # alt: torch.jaccobian[]
                    # jacT[:,i:i+1] = torch.autograd.grad(y, x, grad_outputs=output, retain_graph=True)[0]
                # loss = loss.sum
            if args.N_responses == 'two':
                y_hat1 = y_hat[0] # [batch, 2]
                y_hat2 = y_hat[1] # [batch, 2]
                loss1 = loss_fn(y_hat1, y1)
                loss2 = loss_fn(y_hat2, y2)
                loss = loss1 + loss2

        loss.backward()
        optimizer.step()

    return loss, loss1, loss2, model


    
