from numpy.core.fromnumeric import reshape
import torch 
import numpy as np
import pickle
from itertools import combinations, permutations
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from scipy.stats import pearsonr, ttest_ind
import statsmodels.api as sm
from dataset import get_loaders, WineGrid


def analyze_episodic(model, test_data, args):
    # Collect attention weights for each sample in test set
    model.eval()
    m, x_ = test_data[0] # only 1 episode in test data
    m = m.to(args.device) # m: [1, n_train, sample_dim]
    x = x_[:,:,:-1].to(args.device) # x: [1, n_test, sample_dim]
    y = x_[:,:,-1].type(torch.long).to(args.device)
    y = y.squeeze() # y: [1, n_test]
    with torch.no_grad():
        y_hat, attention = model(x, m) 
        attention = attention[0] # first (only) memory layer
        attention = np.squeeze(attention)
        # attention: [n_train, n_test]
    
    # Check the retrieval weights of relevant vs. irrelevant training samples
    grid = test_data.grid
    train = grid.train # train *samples* in test *episode*
    test = grid.test   # test *samples* in test *episode*
    n_train = len(train)
    n_test = len(test)
    rel_ids = grid.hub_sample_ids # relevant memory ids (train samples)
    attn_ranks = np.zeros_like(attention)
    for i in range(n_test):
        argsorted_attn = np.argsort(attention[i])
        ranks = np.zeros([n_train])
        ranks[argsorted_attn] = np.arange(n_train)
        attn_ranks[i] = ranks
    relevant = []
    irrelevant = []
    for i in range(n_test):
        for j in range(n_train):
            if j in rel_ids[i]:
                relevant.append(attn_ranks[i,j])
            else:
                irrelevant.append(attn_ranks[i,j])
    rank_data = {"relevant": relevant, "irrelevant": irrelevant}

    # Check how often a legitimate "path" was retrieved in the top 5%
    k = 8 # top k memories with highest weights (k = 8 means 5 percent)
    used_hub = []
    for i in range(n_test):
        highest_attn = np.argsort(attention[i])[-k:]
        test_f1, test_f2, test_ctx, test_y = test[i]

        # Get relevant hubs for current test sample
        hubs = []
        for rel_id in rel_ids[i]:
            train_sample = train[rel_id]
            train_f1, train_f2 = train_sample[0], train_sample[1]
            if train_f1 in [test_f1, test_f2]: 
                hubs.append(train_f2)
            if train_f2 in [test_f1, test_f2]:
                hubs.append(train_f1)
        hubs = list(set(hubs))
        hubs_dict = {h:[] for h in hubs}
        assert len(hubs) == 2, "shouldn't be more than 2 hubs?"

        # Check if one of the hubs appears with f1 and f2
        attended_train = [train[idx] for idx in highest_attn]
        for sample in attended_train:
            train_f1, train_f2, train_ctx, train_y = sample
            if train_ctx != test_ctx:
                continue # must be samples testing the same axis to be relevant
            if hubs[0] == train_f1:
                hubs_dict[hubs[0]].append(sample[1])
            if hubs[1] == sample[0]:
                hubs_dict[hubs[1]].append(sample[1])
            if hubs[0] == sample[1]:
                hubs_dict[hubs[0]].append(sample[0])
            if hubs[1] == sample[1]:
                hubs_dict[hubs[1]].append(sample[0])
        if test_f1 in hubs_dict[hubs[0]] and test_f2 in hubs_dict[hubs[0]]:
            used_hub.append(True)
        elif test_f1 in hubs_dict[hubs[1]] and test_f2 in hubs_dict[hubs[1]]:
            used_hub.append(True)
        else:
            used_hub.append(False)
    p_used_hub = np.mean(used_hub)
    print("Proportion that episodic system retrieved a hub path:", p_used_hub)

    results = {"rank_data":rank_data, "p_used_hub": p_used_hub}
    return results

def analyze_cortical(model, test_data, analyze_loader, args):
    # Useful dictionaries from test dataset
    n_states = test_data.n_states 
    loc2idx = test_data.loc2idx 
    idx2loc = {idx:loc for loc, idx in loc2idx.items()}
    idxs = [idx for idx in range(n_states)]
    # locs = [idx2loc[idx] for idx in idxs]
    idx2tensor = test_data.idx2tensor 

    model.eval()
    # Get embeddings from model for each face
    face_embedding = model.face_embedding
    face_embedding.to(args.device)
    embeddings = []
    # Get hiddens from the recurrent model for each face
    
    # if the model was stepwisemlp
    if args.cortical_model=='stepwisemlp':
        hiddens = [[] for i in range(2)]
        hiddens_cong = [[] for i in range(2)]
        hiddens_incong = [[] for i in range(2)] 
        hiddens_ctxs = [[[] for j in range(args.N_contexts)] for i in range(2)]
    else:
        hiddens = [] # hidden reps. for both contexts
        hiddens_incong = []
        hiddens_cong = []
        hiddens_ctxs = [[] for i in range(args.N_contexts)]
    
    idxs1 = []
    idxs2 = []
    idxs1_ctxs = [[] for i in range(args.N_contexts)]
    idxs2_ctxs = [[] for i in range(args.N_contexts)]
    samples = []
    samples_ctxs =[[] for i in range(args.N_contexts)]
    samples_cong = []
    samples_incong = []

    with torch.no_grad():
        for idx in range(n_states):
            face_tensor = idx2tensor[idx].unsqueeze(0).to(args.device) 
            embedding = face_embedding(face_tensor) # [1, state_dim]
            embedding = embedding.cpu().numpy()
            embeddings.append(embedding)
        embeddings = np.concatenate(embeddings, axis=0) # [n_states, state_dim]
        for batch in analyze_loader:
            if args.cortical_task == 'face_task':
                f1, f2, ctx, y, idx1, idx2 = batch
            elif args.cortical_task == 'wine_task':
                f1, f2, ctx, y1, y2, idx1, idx2 = batch
            idx1 = idx1[0]
            idx2 = idx2[0]
            samples.append(batch)
            (x1, y1), (x2, y2) = idx2loc[idx1], idx2loc[idx2]
            f1 = f1.to(args.device) 
            f2 = f2.to(args.device) 
            ctx = ctx.to(args.device)

            # create congruent and incongruent groups
            grid_angle = np.arctan2((y2-y1),(x2-x1))
            phi = np.sin(2*grid_angle)
            if np.abs(phi)<1e-5:
                # for congrunet trials, 
                # zero out those very close to zero angles
                # so it won't turn into 1 or -1 by sign
                cong = 0 
            else:
                cong = np.sign(phi) # 1: congruent, -1:incongruent, 0:none

            # get the hidden reps.    
            y_hat, out = model(f1, f2, ctx) 
            # y_hat: [1, 2]
            # rnn_out: [seq_length, 1, hidden_dim]: [3, 1, 128]
            # mlp_out: [1, hidden_dim]: [1, 128]
            if args.order_ctx == 'first':
                f1_ind = 1
                f2_ind = 2
            elif args.order_ctx == 'last':
                f1_ind = 0
                f2_ind = 1
            if args.cortical_model=='stepwisemlp':
                out1, out2 = out
                out1 = out1.cpu().numpy()
                out2 = out2.cpu().numpy()
                hiddens[0].append(out1)
                hiddens[1].append(out2)
                hiddens_ctxs[0][ctx].append(out1)
                hiddens_ctxs[1][ctx].append(out2)
            else:
                out = out.cpu().numpy()
                hiddens.append(out)
                hiddens_ctxs[ctx].append(out)
            
            ctx = ctx[0].cpu().numpy()
            idxs1.append(idx1)
            idxs2.append(idx2)
            idxs1_ctxs[ctx].append(idx1)
            idxs2_ctxs[ctx].append(idx2)
            samples_ctxs[ctx].append(batch)
            if ((cong==1) and ((ctx==0) or (ctx==1))):
                if args.cortical_model=='stepwisemlp':
                    hiddens_cong[0].append(out1)
                    hiddens_cong[1].append(out2)
                else:
                    hiddens_cong.append(out)
                samples_cong.append(batch)
            elif ((cong==-1) and ((ctx==0) or (ctx==1))):
                if args.cortical_model=='stepwisemlp':
                    hiddens_incong[0].append(out1)
                    hiddens_incong[1].append(out2)
                else:
                    hiddens_incong.append(out)
                samples_incong.append(batch)

    hiddens = np.asarray(hiddens).squeeze() 
    # for n_ctx=2, data_len = 16*12*2=384 (n_states:16, n_states-ties:12, permutation:2)
    # rnn hiddens: [data_len, seq_length, hidden_dim] : [384, 3, 128]
    # mlp hiddens: [data_len, hidden_dim]: [384, 128]
    # stepwisemlp hiddens: [num_hidds, data_len, hidden_dim]: [2, 384, 128]
    # with diagonals - wine task = data_len = (n_ctx-n_diag)*192+n_diag*212 
    # [n_ctx:2, data_len:384], [n_ctx:4, data_len:768], [n_ctx:8, data_len: 1616]
    hiddens_incong = np.asarray(hiddens_incong).squeeze() 
    hiddens_cong = np.asarray(hiddens_cong).squeeze() 
    # rnn hiddens_cong/incong: [144, 3, 128]
    # mlp hiddens_cong/incong: [144, 128]
    # mlp hiddens_cong/incong: [2, 144, 128]
    
    # hiddens_ctx: even tho it is 384, but it is ordered based on the contexts
    if args.cortical_model=='stepwisemlp':
        hiddens_ctx = np.concatenate(np.asarray(hiddens_ctxs).squeeze(), axis=1)
        # hiddens_ctxs: [n_hidds=2, n_ctx, 192, 1, 128]
        # hiddens_ctx: [n_hidds=2, 384, 128]
        hiddens_inc_c =  np.concatenate((hiddens_incong, hiddens_cong), axis=1) 
        # hiddens_inc_c: [n_hidds, 384-ties, 128]: [2, 288, 128]
    else:
        hiddens_ctx = np.concatenate(hiddens_ctxs, axis = 0).squeeze()
        # mlp hiddens_ctxs: [n_ctx, 192, 1, 128]
        # rnn hiddens_ctxs: [n_ctx, n_trials, 3, 1, 128]
        # rnn hiddens_ctx: [384, 3, 128]
        # mlp hiddens_ctx: [384, 128]
        hiddens_inc_c =  np.concatenate((hiddens_incong, hiddens_cong), axis=0) 
        # rnn hiddens_inc_c: [384-ties, seq_length, 128]: [288, 3, 128]
        # mlp hiddens_inc_c: [384-ties, 128]: [288, 128]
    if args.cortical_model=='rnn':
        hiddens_ctx = hiddens_ctx[:, -1, :] # [384, 128]
        hiddens_inc_c = hiddens_inc_c[:, -1, :] #[288, 128]
    samples_inc_c = np.concatenate((samples_incong, samples_cong), axis=0)
    
    if args.cortical_model=='stepwisemlp':
        avg_hidden = np.zeros([2, n_states, hiddens.shape[-1]])
        avg_hidden_ctxs = np.zeros([2, args.N_contexts, n_states, hiddens.shape[-1]])
    else:
        avg_hidden = np.zeros([n_states, hiddens.shape[-1]])
        avg_hidden_ctxs = np.zeros([args.N_contexts, n_states, hiddens.shape[-1]])
    
    if args.cortical_model=='rnn':
        hiddens_ctxs = np.asarray(hiddens_ctxs).squeeze() # [2, 192, 3, 128]
        # Take average for each face based on its location
        for f in range(n_states):
            temp1 = [hiddens[i,f1_ind,:] 
                        for i, idx1 in enumerate(idxs1) if idx1==f]
            temp2 = [hiddens[i,f2_ind,:] 
                        for i, idx2 in enumerate(idxs2) if idx2==f]
            if len(temp1 + temp2)>1:
                    avg_hidden[f] = np.concatenate(temp1 + temp2, axis=0).mean(axis=0)  
            for ctx in range(args.N_contexts):
                temp1_ctxs = [hiddens_ctxs[ctx,i,f1_ind,:] 
                                for i, idx1 in enumerate(idxs1_ctxs[ctx]) if idx1==f]
                temp2_ctxs = [hiddens_ctxs[ctx,i,f2_ind,:] 
                                for i, idx2 in enumerate(idxs2_ctxs[ctx]) if idx2==f]
                if len(temp1_ctxs + temp2_ctxs)>1:
                    avg_hidden_ctxs[ctx, f, :] = np.concatenate(temp1_ctxs + temp2_ctxs, axis=0).mean(axis=0)
                    # avg_hidden_ctxs: [n_contexts, n_states, hidden_dim]: [2, 16, 128] 
        avg_hidden_ctx = np.concatenate(avg_hidden_ctxs, axis=0)
    elif args.cortical_model=='mlp':
        for f in range(n_states):
            temp = [hiddens[i,:] 
                        for i, (idx1, idx2) in enumerate(zip(idxs1, idxs2))
                            if ((idx1==f) | (idx2==f))]
            if len(temp)>1:
                avg_hidden[f] = np.mean(temp, axis=0)
            for ctx in range(args.N_contexts):  
                temp_ctxs = [hiddens_ctxs[ctx][i]
                            for i, (idx1, idx2) in enumerate(zip(idxs1_ctxs[ctx], idxs2_ctxs[ctx]))
                            if ((idx1==f) | (idx2==f))]
                if len(temp_ctxs)>1:
                    avg_hidden_ctxs[ctx, f, :] = np.mean(temp_ctxs, axis=0)
                    # avg_hidden_ctxs: [n_contexts, n_states, hidden_dim]: [2, 16, 128] 
        avg_hidden_ctx = np.concatenate(avg_hidden_ctxs, axis=0)
    elif args.cortical_model=='stepwisemlp':
        # todo: how to do the averaging? over both hidden reps?
        # hiddens_ctxs anf hiddens_inc_c for the pca results should have two dimensions, 
        hiddens_ctxs = np.asarray(hiddens_ctxs).squeeze()
        for f in range(n_states):
            temp1 = [hiddens[0,i,:] 
                        for i, idx1 in enumerate(idxs1) if idx1==f]
            temp2 = [hiddens[1,i,:] 
                        for i, idx2 in enumerate(idxs2) if idx2==f]
            if len(temp1)>1:
                avg_hidden[0,f,:] = np.mean(temp1, axis=0)
            if len(temp2)>1:
                avg_hidden[1,f,:] = np.mean(temp2, axis=0)
            # avg_hidden: [n_hidd, n_states, hidd_dim]: [2,16,128]
            for ctx in range(args.N_contexts):
                temp1_ctxs = [hiddens_ctxs[0,ctx,i,:] 
                             for i, idx1 in enumerate(idxs1_ctxs[ctx]) if idx1==f]
                temp2_ctxs = [hiddens_ctxs[1,ctx,i,:] 
                             for i, idx2 in enumerate(idxs2_ctxs[ctx]) if idx2==f]   
                if len(temp1_ctxs)>1:
                    avg_hidden_ctxs[0,ctx,f,:] = np.mean(temp1_ctxs, axis=0)
                if len(temp2_ctxs)>1:
                    avg_hidden_ctxs[1,ctx,f,:] = np.mean(temp2_ctxs, axis=0)
                # avg_hidden_ctxs: [n_hidd, n_contexts, n_states, hidden_dim]: [2, 2, 16, 128] 
        avg_hidden_ctx = np.concatenate(avg_hidden_ctxs, axis=1)
    samples_res = {'samples': samples, 
                   'samples_ctxs': samples_ctxs,
                   'samples_inc_c': samples_inc_c}

    results = {'samples_res':samples_res,
               'idxs1': idxs1, 'idxs2': idxs2,
               'embeddings': embeddings, # [16, 32]
               'hiddens_ctx':hiddens_ctx, # mlp: [384,128] or in stepwisedmlp: [2,384,128]
               'hiddens_ctxs':hiddens_ctxs, # mlp: [n_ctx, 192, 1, 128]
               'avg_hidden':avg_hidden, # [16, 128] or [n_hidd=2, 16, 128]
               'avg_hidden_ctx':avg_hidden_ctx, # [32, 128] or [n_hidd=2, 32, 128]
               'avg_hidden_ctxs':avg_hidden_ctxs, # [n_ctx, 16, 128] or [n_hidd=2, n_ctx, 16, 128]
               'hiddens_inc_c': hiddens_inc_c} # [288, 128] or [n_hidd=2, 288, 128]
    return results

def proportions(cortical_results):
    hiddens_ctxs = cortical_results['hiddens_ctxs'] # list of len [n_ctx]
    hiddens_ctxs = [np.concatenate(h, axis=0) for h in hiddens_ctxs] # list of len [n_ctx] each has either [192,128] or [224,128]
    # when n_ctx=8, we have diff number of ties, therefore, 
    # in the first 4 contexts we have [192, 128], and in 
    # the second 4 contexts (diagonals) we have [224, 128]
    # that is why we go over each of the hiddens in hiddens_ctxs
    # and then concat them to create [n_trials, hidden_dim] for each
    ps = []
    p_pies = []
    for h in hiddens_ctxs: # h: [n_trials, hidden_dim]
        p_pies.append(np.any(h>0, axis=0)) # list of len [n_ctx], each shape [128,]
        ps.append(np.mean(h>0, axis=0)) # [n_ctx, 128]
    ps = np.asarray(ps) 
    # ps: [n_ctx, 128]
    # avg num of the trials that were active for each unit, and for each context
    s = np.sum(ps, axis=0, keepdims=True) 
    # s: [1, hidden_dim], overall activity of each hidden unit, 
    # if that unit was active at all, over all trials (regardless of the context)
    n = ps / s 
    # n: [n_ctx, hidden_dim] 
    # normalized - how much each unit is active for each ctx over trials 
    # normalized by the overall activity of that unit for all ctx and trials
    # f = n > threshold
    # there are some NaNs
    prop_results = {'hiddens_ctxs': hiddens_ctxs,
                    'p_pies': p_pies, # which trials are active for each hidden unit,  
                    'ps': ps, # on average, how many trials were active for each hidden unit
                    'n': n}
    return prop_results

def calc_dist_ctx(cortical_results, test_data, args):
    N_contexts = 2 #ToDo: for now it works only for x and y, because of the angles
    # Useful dictionaries from test dataset
    n_states = test_data.n_states 
    loc2idx = test_data.loc2idx 
    idx2loc = {idx:loc for loc, idx in loc2idx.items()}
    idxs = [idx for idx in range(n_states)]
    N_contexts = args.N_contexts
    N_responses = args.N_responses
    avg_hidden_ctxs =  cortical_results['avg_hidden_ctxs'] # [2, 16, 128]
    # Correlation
    grid_dists = []
    hidd_dists_ctxs = [[] for i in range(N_contexts)]
    grid_1ds_ctxs = [[] for i in range(N_contexts)]
    grid_angles = []
    samples = []

    for idx1, idx2 in combinations(idxs, 2):
        (x1, y1), (x2, y2) = idx2loc[idx1], idx2loc[idx2]
        samples.append((idx1, idx2))
        grid_dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        grid_dists.append(grid_dist)
        for ctx in range(N_contexts):
            # Euclidean distance between hidden reps. in context ctx
            if args.cortical_model=='stepwisemlp':
                hidd_dist = np.zeros([2])
                hidd1, hidd2 = avg_hidden_ctxs[0,ctx,idx1,:], avg_hidden_ctxs[0,ctx,idx2,:]
                hidd_dist[0] = np.linalg.norm(hidd1 - hidd2)
                hidd1, hidd2 = avg_hidden_ctxs[1,ctx,idx1,:], avg_hidden_ctxs[1,ctx,idx2,:]
                hidd_dist[1] = np.linalg.norm(hidd1 - hidd2)
            else:
                hidd1, hidd2 = avg_hidden_ctxs[ctx][idx1], avg_hidden_ctxs[ctx][idx2]
                hidd_dist = np.linalg.norm(hidd1 - hidd2)
            hidd_dists_ctxs[ctx].append(hidd_dist)
            # 1D rank - Manhattan distance
            loc1 = [x1, y1]
            loc2 = [x2, y2]
            winegrid = WineGrid(N_responses, N_contexts)
            r1, r2 = winegrid.ctx_to_r(ctx, loc1, loc2) 
            grid_1ds_ctxs[ctx].append(np.abs(r1-r2))
            # create on and off diagonal groups
        
        grid_angle = np.arctan2((y2-y1),(x2-x1))
        grid_angles.append(grid_angle)
        
    grid_dists = np.array(grid_dists) # [(n_states*(nstates-1))/2]: [120]
    grid_angles = np.array(grid_angles) # [120]
    samples = np.array(samples)
    hidd_dists_ctxs = np.array(hidd_dists_ctxs) # [n_ctx, sampels, n_hidds]: in mlp: [2,120], in stepwisemlp: [2,120,2]

    phi = np.sin(2*grid_angles)
    binary_phi = np.sign(phi)
    for i, p in enumerate(phi):
        if np.abs(p)<1e-5:
            binary_phi[i] = 0

    angle_results = {'grid_angles': grid_angles,
                     'phi': phi,
                     'binary_phi': binary_phi}
    dist_results = {'samples': samples,
                    'hidd_dists_ctxs': hidd_dists_ctxs,
                    'grid_1ds_ctxs': grid_1ds_ctxs,
                    'grid_dists': grid_dists,
                    'angle_results': angle_results}
    return dist_results

def calc_dist(cortical_results, test_data, args):
    # Useful dictionaries from test dataset
    n_states = test_data.n_states 
    loc2idx = test_data.loc2idx 
    idx2loc = {idx:loc for loc, idx in loc2idx.items()}
    idxs = [idx for idx in range(n_states)]

    # Correlation
    grid_dists = []
    cong_grid_dists = []
    incong_grid_dists = []
    embed_dists = []
    hidd_dists = []
    cong_hidd_dists = []
    incong_hidd_dists = []
    cong_embed_dists = []
    incong_embed_dists = []
    grid_angles = []
    cong_grid_angles = []
    incong_grid_angles = []
    samples = []

    embeddings =  cortical_results['embeddings']
    avg_hidden =  cortical_results['avg_hidden'] # [16, 128]

    for idx1, idx2 in combinations(idxs, 2):
        (x1, y1), (x2, y2) = idx2loc[idx1], idx2loc[idx2]
        samples.append((idx1, idx2))
        grid_dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        grid_dists.append(grid_dist)
        # Euclidean distance between embeddings
        emb1, emb2 = embeddings[idx1], embeddings[idx2]
        embed_dist = np.linalg.norm(emb1 - emb2)
        embed_dists.append(embed_dist)
        # Euclidean distance between hidden reps.
        if args.cortical_model=='stepwisemlp':
            hidd_dist = np.zeros([2])
            hidd1, hidd2 = avg_hidden[0,idx1], avg_hidden[0,idx2]
            hidd_dist[0] = np.linalg.norm(hidd1 - hidd2)
            hidd1, hidd2 = avg_hidden[1,idx1], avg_hidden[1,idx2]
            hidd_dist[1] = np.linalg.norm(hidd1 - hidd2)
        else:
            hidd1, hidd2 = avg_hidden[idx1], avg_hidden[idx2]
            hidd_dist = np.linalg.norm(hidd1 - hidd2)
        hidd_dists.append(hidd_dist)
        # create on and off diagonal groups
        grid_angle = np.arctan2((y2-y1),(x2-x1))
        grid_angles.append(grid_angle)
        phi = np.sin(2*grid_angle)
        if np.abs(phi)<1e-5:
            # for congrunet trials, 
            # zero out those very close to zero angles
            # so it won't turn into 1 or -1 by sign
            cong = 0
        else:
            cong = np.sign(phi) # 1: congruent, -1:incongruent, 0:none
        if cong==1:
            cong_hidd_dists.append(hidd_dist)
            cong_grid_dists.append(grid_dist)
            cong_embed_dists.append(embed_dist)
            cong_grid_angles.append(grid_angle)
        if cong==-1:
            incong_hidd_dists.append(hidd_dist)
            incong_grid_dists.append(grid_dist)
            incong_embed_dists.append(embed_dist)
            incong_grid_angles.append(grid_angle)      
    grid_dists = np.array(grid_dists) # [(n_states*(nstates-1))/2]: [120]
    embed_dists = np.array(embed_dists)
    hidd_dists = np.array(hidd_dists)
    cong_grid_dists = np.array(cong_grid_dists) # [36]
    incong_grid_dists = np.array(incong_grid_dists) # [36]
    cong_hidd_dists = np.array(cong_hidd_dists)
    incong_hidd_dists = np.array(incong_hidd_dists)
    cong_embed_dists = np.array(cong_embed_dists)
    incong_embed_dists = np.array(incong_embed_dists)
    grid_angles = np.array(grid_angles) # [120]
    cong_grid_angles = np.array(cong_grid_angles) # [36]
    incong_grid_angles = np.array(incong_grid_angles) # [36]
    samples = np.array(samples)

    phi = np.sin(2*grid_angles)
    binary_phi = np.sign(phi)
    for i, p in enumerate(phi):
        if np.abs(p)<1e-5:
            binary_phi[i] = 0

    cong_dist_results = {'cong_grid_dists': cong_grid_dists,
                    'cong_hidd_dists': cong_hidd_dists,
                    'cong_embed_dists': cong_embed_dists}
    incong_dist_results = {'incong_grid_dists': incong_grid_dists,
                      'incong_hidd_dists': incong_hidd_dists,
                      'incong_embed_dists': incong_embed_dists}
    angle_results = {'grid_angles': grid_angles,
                    'cong_grid_angles': cong_grid_angles, 
                    'incong_grid_angles': incong_grid_angles,
                    'phi': phi,
                    'binary_phi': binary_phi}
    dist_results = {'samples': samples, 
                    'grid_dists': grid_dists,
                    'embed_dists': embed_dists,
                    'hidd_dists':hidd_dists,
                    'cong_dist_results': cong_dist_results,
                    'incong_dist_results': incong_dist_results,
                    'angle_results': angle_results}
    return dist_results

def hist_data(dist_results):
    # embeddings
    cong_embed_dists = dist_results['cong_dist_results']['cong_embed_dists']
    incong_embed_dists = dist_results['incong_dist_results']['incong_embed_dists']
    
    # hiddens
    cong_hidd_dists = dist_results['cong_dist_results']['cong_hidd_dists']
    incong_hidd_dists = dist_results['incong_dist_results']['incong_hidd_dists']
    
    dist_c_inc_results = {'cong_embed_dist': cong_embed_dists, 
                         'incong_embed_dist': incong_embed_dists,
                         'cong_hidd_dist': cong_hidd_dists,
                         'incong_hidd_dist': incong_hidd_dists}
    
    return dist_c_inc_results

def calc_ratio(dist_results):
    # embeddings
    cong_embed_dists = dist_results['cong_dist_results']['cong_embed_dists']
    incong_embed_dists = dist_results['incong_dist_results']['incong_embed_dists']
    avg_cong_embed = np.mean(cong_embed_dists)
    avg_incong_embed = np.mean(incong_embed_dists)
    ratio_embed = (avg_cong_embed/avg_incong_embed)
    
    # hiddens
    cong_hidd_dists = dist_results['cong_dist_results']['cong_hidd_dists']
    incong_hidd_dists = dist_results['incong_dist_results']['incong_hidd_dists']
    avg_cong_hidd = np.mean(cong_hidd_dists, axis=0)
    avg_incong_hidd = np.mean(incong_hidd_dists, axis=0)
    ratio_hidd = (avg_cong_hidd/avg_incong_hidd)
    
    ratio_results = {'ratio_embed': ratio_embed, 'ratio_hidd': ratio_hidd}
    
    return ratio_results

def analyze_dim_red(args, cortical_results, method, n_components):
    embeddings = cortical_results['embeddings'] # [16, 32]
    hiddens_ctx = cortical_results['hiddens_ctx'] # [384, 128] or in stepwisemlp: [2,384,128]
    avg_hidden = cortical_results['avg_hidden'] # [16, 128] or in stepwisemlp: [2,16,128]
    avg_hidden_ctx = cortical_results['avg_hidden_ctx'] # [32, 128] or in stepwisemlp: [2,32,128]
    hiddens_inc_c = cortical_results['hiddens_inc_c'] # [288, 128] or in stepwisemlp: [2,288,128]
    # hiddens_ctx = np.asarray(hiddens_ctxs)
    # hiddens_ctxs = np.concatenate(hiddens_ctxs, axis=0).squeeze() # [384, 128] or [384, 3, 128]
    # if args.cortical_model == 'rnn':
    #     hiddens_ctxs = hiddens_ctxs[:,-1, :]
    # avg_hidden_ctxs = np.concatenate(avg_hidden_ctxs, axis=0) # [32, 128]
    results = {}
    # PCA
    if method == 'pca':
        pca = PCA(n_components=n_components)
        pca_2d_embed = pca.fit_transform(embeddings)
        if args.cortical_model=='stepwisemlp':
            pca_2d_hidd = np.zeros([hiddens_ctx.shape[0], hiddens_ctx.shape[1], n_components])
            pca_2d_hidd[0,:,:] = pca.fit_transform(hiddens_ctx[0,:,:]) # this is all the hiddens, no averaging for each face
            pca_2d_hidd[1,:,:] = pca.fit_transform(hiddens_ctx[1,:,:])
            pca_2d_avg_hidd = np.zeros([avg_hidden.shape[0], avg_hidden.shape[1], n_components])
            pca_2d_avg_hidd[0,:,:] = pca.fit_transform(avg_hidden[0,:,:]) 
            pca_2d_avg_hidd[1,:,:] = pca.fit_transform(avg_hidden[1,:,:]) 
            pca_2d_ctx_hidd = np.zeros([avg_hidden_ctx.shape[0], avg_hidden_ctx.shape[1], n_components])
            pca_2d_ctx_hidd[0,:,:] = pca.fit_transform(avg_hidden_ctx[0,:,:])
            pca_2d_ctx_hidd[1,:,:] = pca.fit_transform(avg_hidden_ctx[1,:,:])
            pca_2d_incong_cong = np.zeros([hiddens_inc_c.shape[0], hiddens_inc_c.shape[1], n_components])
            pca_2d_incong_cong[0,:,:] = pca.fit_transform(hiddens_inc_c[0,:,:])
            pca_2d_incong_cong[1,:,:] = pca.fit_transform(hiddens_inc_c[1,:,:]) 
        else:
            pca_2d_hidd = pca.fit_transform(hiddens_ctx) # this is all the hiddens, no averaging for each face
            pca_2d_avg_hidd = pca.fit_transform(avg_hidden) # I might need to save this at all
            pca_2d_ctx_hidd = pca.fit_transform(avg_hidden_ctx)
            pca_2d_incong_cong = pca.fit_transform(hiddens_inc_c)
        results = {'embed_2d': pca_2d_embed, 
                   'hidd_2d': pca_2d_hidd,
                   'avg_hidd_2d': pca_2d_avg_hidd,
                   'ctx_hidd_2d': pca_2d_ctx_hidd,
                   'incong_cong_2d': pca_2d_incong_cong}
    elif method == 'mds':
        # MDS
        mds = MDS(n_components=n_components)
        mds_2d_embed = mds.fit_transform(embeddings)
        mds_2d_hidd = mds.fit_transform(hiddens_ctx) # this is all the hiddens, no averaging for each face
        mds_2d_avg_hidd = mds.fit_transform(avg_hidden) # I might need to save this at all
        mds_2d_ctx_hidd = mds.fit_transform(avg_hidden_ctx)
        mds_2d_incong_cong = mds.fit_transform(hiddens_inc_c)
        results = {'embed_2d': mds_2d_embed, 
                    'hidd_2d': mds_2d_hidd,
                    'avg_hidd_2d': mds_2d_avg_hidd,
                    'ctx_hidd_2d': mds_2d_ctx_hidd,
                    'incong_cong_2d': mds_2d_incong_cong}
    elif method == 'tsne':
        # tSNE
        tsne = TSNE(n_components=n_components)
        tsne_2d_embed = tsne.fit_transform(embeddings)
        tsne_2d_hidd = tsne.fit_transform(hiddens_ctx) # this is all the hiddens, no averaging for each face
        tsne_2d_avg_hidd = tsne.fit_transform(avg_hidden) # I might need to save this at all
        tsne_2d_ctx_hidd = tsne.fit_transform(avg_hidden_ctx)
        tsne_2d_incong_cong = tsne.fit_transform(hiddens_inc_c)
        results = {'embed_2d': tsne_2d_embed, 
                    'hidd_2d': tsne_2d_hidd,
                    'avg_hidd_2d': tsne_2d_avg_hidd,
                    'ctx_hidd_2d': tsne_2d_ctx_hidd,
                    'incong_cong_2d': tsne_2d_incong_cong}
    return results

def analyze_ttest(args, dist_results):  
    cong_res = dist_results['cong_dist_results']
    incong_res = dist_results['incong_dist_results']
    
    incong_hidd_dists = incong_res['incong_hidd_dists']
    cong_hidd_dists = cong_res['cong_hidd_dists']
    if args.cortical_model == 'stepwisemlp':
        t_hidd, t_p_val_hidd = np.zeros([2]), np.zeros([2])
        for h in range(2):
            t_hidd[h], t_p_val_hidd[h]   = ttest_ind(cong_hidd_dists[:,h], incong_hidd_dists[:,h])
    else:
        t_hidd, t_p_val_hidd   = ttest_ind(cong_res['cong_hidd_dists'], 
                                       incong_res['incong_hidd_dists'])
    t_embed, t_p_val_embed = ttest_ind(cong_res['cong_embed_dists'], 
                                       incong_res['incong_embed_dists'])
    t_grid, t_p_val_grid   = ttest_ind(cong_res['cong_grid_dists'], 
                                       incong_res['incong_grid_dists'])
    ttest_results = {'t_stat_hidd':t_hidd, 't_p_val_hidd': t_p_val_hidd,
                    't_stat_embed':t_embed, 't_p_val_embed': t_p_val_embed,
                    't_grid':t_grid, 't_p_val_grid': t_p_val_grid}
    return ttest_results

def analyze_corr(args, dist_results):
    grid_dists = dist_results['grid_dists']
    embed_dists = dist_results['embed_dists'] 
    hidd_dists = dist_results['hidd_dists']    
    cong_res = dist_results['cong_dist_results']
    incong_res = dist_results['incong_dist_results']
    r_embed, p_val_embed = pearsonr(grid_dists, embed_dists)
    if args.cortical_model == 'stepwisemlp':
        r_hidd, p_val_hidd = np.zeros([2]), np.zeros([2])
        r_hidd[0], p_val_hidd[0] = pearsonr(grid_dists, hidd_dists[:,0])
        r_hidd[1], p_val_hidd[1] = pearsonr(grid_dists, hidd_dists[:,1])
        r_cong_hidd, p_val_cong_hidd, r_incong_hidd,  p_val_incong_hidd = \
             np.zeros([2]), np.zeros([2]), np.zeros([2]), np.zeros([2])
        cong_hidd_dists, incong_hidd_dists = cong_res['cong_hidd_dists'], \
                                             incong_res['incong_hidd_dists']
        for h in range(2):
            r_cong_hidd[h], p_val_cong_hidd[h] = pearsonr(cong_res['cong_grid_dists'], 
                                                          cong_hidd_dists[:,h])     
        
            r_incong_hidd[h], p_val_incong_hidd[h] = pearsonr(incong_res['incong_grid_dists'],
                                                              incong_hidd_dists[:,h])                                
        
    else:
        r_hidd, p_val_hidd = pearsonr(grid_dists, hidd_dists)
        r_cong_hidd, p_val_cong_hidd = pearsonr(cong_res['cong_grid_dists'], 
                                                cong_res['cong_hidd_dists'])
        r_incong_hidd, p_val_incong_hidd = pearsonr(incong_res['incong_grid_dists'],
                                                    incong_res['incong_hidd_dists'])
    r_cong_embed, p_val_cong_embed = pearsonr(cong_res['cong_grid_dists'], 
                                            cong_res['cong_embed_dists'])
    r_incong_embed, p_val_incong_embed = pearsonr(incong_res['incong_grid_dists'], 
                                                incong_res['incong_embed_dists']) 
    corr_results = {'r_embed': r_embed, 'p_val_embed': p_val_embed,
                           'r_cong_embed': r_cong_embed, 
                           'p_val_cong_embed': p_val_cong_embed,
                           'r_incong_embed': r_incong_embed, 
                           'p_val_incong_embed': p_val_incong_embed,
                           'r_hidd': r_hidd, 'p_val_hidd': p_val_hidd,
                           'r_cong_hidd': r_cong_hidd, 
                           'p_val_cong_hidd': p_val_cong_hidd,
                           'r_incong_hidd': r_incong_hidd, 
                           'p_val_incong_hidd': p_val_incong_hidd}
    return corr_results

def analyze_regression(args, dist_results):
    hidd_dists = dist_results['hidd_dists']
    grid_dists = dist_results['grid_dists']
    
    phi = dist_results['angle_results']['phi']
    binary_phi = dist_results['angle_results']['binary_phi']
           
    # prepare data for the regression analysis
    x_cat = np.concatenate((grid_dists.reshape((-1,1)), binary_phi.reshape((-1,1))),axis=1)
    x_con = np.concatenate((grid_dists.reshape((-1,1)), phi.reshape((-1,1))),axis=1)
    # categorical regression analysis
    x_cat = sm.add_constant(x_cat)
    if args.cortical_model=='stepwisemlp':
        p_val, t_val, param, bse = ([[] for i in range(2)] for i in range(4))
    else:
        p_val, t_val, param, y_hat_E, y, bse = ([] for i in range(6))
    if args.cortical_model == 'stepwisemlp':
        y_hat_E = np.zeros(hidd_dists.shape)
        y = np.zeros(hidd_dists.shape)
        for h in range(2):
            # y = hidd_dists[:,h]
            y[:,h] = hidd_dists[:,h]
            stats_model_cat = sm.OLS(y[:,h],x_cat).fit() 
            y_hat_E[:,h] = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists)
            p_val[h], t_val[h], param[h], bse[h] = stats_model_cat.pvalues, stats_model_cat.tvalues, \
            stats_model_cat.params, stats_model_cat.bse
    else:
        y = hidd_dists
        stats_model_cat = sm.OLS(y,x_cat).fit() 
        y_hat_E = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists)
        p_val, t_val, param, bse = stats_model_cat.pvalues, stats_model_cat.tvalues, \
            stats_model_cat.params, stats_model_cat.bse
    cat_reg = {'p_val': p_val,
               't_val': t_val,
               'param': param,
               'y_hat_E': y_hat_E,
               'y': y,
               'bse': bse}
    # continuous regression analysis
    x_con = sm.add_constant(x_con)
    if args.cortical_model=='stepwisemlp':
        p_val, t_val, param, bse = ([[] for i in range(2)] for i in range(4))
    else:
        p_val, t_val, param, y_hat_E, y, bse = ([] for i in range(6))
    if args.cortical_model == 'stepwisemlp':
        y_hat_E = np.zeros(hidd_dists.shape)
        y = np.zeros(hidd_dists.shape)
        for h in range(2):
            y[:,h] = hidd_dists[:,h]
            stats_model_con = sm.OLS(y[:,h],x_con).fit()
            y_hat_E[:,h] = stats_model_con.params[0] + (stats_model_con.params[1]*grid_dists)
            p_val[h], t_val[h], param[h], bse[h] = stats_model_con.pvalues, stats_model_con.tvalues, \
            stats_model_con.params, stats_model_con.bse
    else:
        y = hidd_dists
        stats_model_con = sm.OLS(y,x_con).fit()
        y_hat_E = stats_model_con.params[0] + (stats_model_con.params[1]*grid_dists)
        p_val, t_val, param, bse = stats_model_con.pvalues, stats_model_con.tvalues, \
            stats_model_con.params, stats_model_con.bse
    con_reg = {'p_val': p_val,
               't_val': t_val,
               'param': param,
               'y_hat_E': y_hat_E,
               'y': y,
               'bse': bse}

    reg_results = {'cat_reg': cat_reg, 
                   'con_reg': con_reg}
    return reg_results

def analyze_regression_1D(args,dist_ctx_results):
    hidd_dists_ctxs = dist_ctx_results['hidd_dists_ctxs']
    hidd_dists_ctx0 = hidd_dists_ctxs[0]
    hidd_dists_ctx1 = hidd_dists_ctxs[1]
    grid_1ds_ctxs = dist_ctx_results['grid_1ds_ctxs']
    grid_1ds_ctx0 = grid_1ds_ctxs[0]
    grid_1ds_ctx1 = grid_1ds_ctxs[1]
    grid_dists = dist_ctx_results['grid_dists']
    
    phi = dist_ctx_results['angle_results']['phi']
    binary_phi = dist_ctx_results['angle_results']['binary_phi']
    
    hidd_dists_ctx = np.concatenate((hidd_dists_ctx0, hidd_dists_ctx1), axis=0)
    grid_1ds_ctx = np.concatenate((grid_1ds_ctx0, grid_1ds_ctx1), axis=0)
    grid_dists_ctx = np.concatenate((grid_dists, grid_dists), axis=0)
    binary_phi_ctx = np.concatenate((binary_phi, binary_phi), axis=0)
    phi_ctx = np.concatenate((phi, phi), axis=0)
    # prepare data for the regression analysis
    x_cat = np.concatenate((grid_dists_ctx.reshape((-1,1)), grid_1ds_ctx.reshape((-1,1)),
                            binary_phi_ctx.reshape((-1,1))),axis=1) # [240, 3]
    x_con = np.concatenate((grid_dists_ctx.reshape((-1,1)), grid_1ds_ctx.reshape((-1,1)),
                            phi_ctx.reshape((-1,1))),axis=1)
    
    # categorical regression analysis
    x_cat = sm.add_constant(x_cat)
    if args.cortical_model=='stepwisemlp':
        p_val, t_val, param, y_hat_E, y, bse = ([[] for i in range(2)] for i in range(6))
    else:
        p_val, t_val, param, y_hat_E, y, bse = ([] for i in range(6))
    if args.cortical_model == 'stepwisemlp':
        y_hat_E = np.zeros(hidd_dists_ctx.shape)
        for h in range(2):
            y = hidd_dists_ctx[:,h]
            stats_model_cat = sm.OLS(y,x_cat).fit() 
            y_hat_E[:,h] = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists_ctx)        
            p_val[h], t_val[h], param[h], bse[h] = stats_model_cat.pvalues, stats_model_cat.tvalues, \
            stats_model_cat.params, stats_model_cat.bse
    else:
        y = hidd_dists_ctx
        stats_model_cat = sm.OLS(y,x_cat).fit() 
        y_hat_E = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists_ctx)
        p_val, t_val, param, bse = stats_model_cat.pvalues, stats_model_cat.tvalues, \
            stats_model_cat.params, stats_model_cat.bse
    cat_reg = {'p_val': p_val,
               't_val': t_val,
               'param': param,
               'y_hat_E': y_hat_E,
               'y': y,
               'bse': bse}
    # continuous regression analysis
    x_con = sm.add_constant(x_con)
    if args.cortical_model=='stepwisemlp':
        p_val, t_val, param, y_hat_E, y, bse = ([[] for i in range(2)] for i in range(6))
    else:
        p_val, t_val, param, y_hat_E, y, bse = ([] for i in range(6))
    if args.cortical_model == 'stepwisemlp':
        y_hat_E = np.zeros(hidd_dists_ctx.shape)
        for h in range(2):
            y = hidd_dists_ctx[:,h]
            stats_model_con = sm.OLS(y,x_con).fit()
            y_hat_E[:,h] = stats_model_con.params[0] + (stats_model_con.params[1]*grid_dists_ctx)        
            p_val[h], t_val[h], param[h], bse[h] = stats_model_con.pvalues, stats_model_con.tvalues, \
            stats_model_con.params, stats_model_con.bse
    else:
        y = hidd_dists_ctx
        stats_model_con = sm.OLS(y,x_con).fit()
        y_hat_E = stats_model_con.params[0] + (stats_model_con.params[1]*grid_dists_ctx)
        p_val, t_val, param, bse = stats_model_con.pvalues, stats_model_con.tvalues, \
            stats_model_con.params, stats_model_con.bse
    con_reg = {'p_val': p_val,
               't_val': t_val,
               'param': param,
               'y_hat_E': y_hat_E,
               'y': y,
               'bse': bse}

    reg_results = {'cat_reg': cat_reg, 
                   'con_reg': con_reg}
    return reg_results

def analyze_regression_exc(args, dist_results, test_data):
    # Useful dictionaries from test dataset
    n_states = test_data.n_states 
    hidd_dists = dist_results['hidd_dists'] #[n_combinations]: [120]
    grid_dists = dist_results['grid_dists']
    binary_phi = dist_results['angle_results']['binary_phi'] # [120]
    samples = dist_results['samples'] # [120, 2]
    states=[]
    if args.cortical_model=='stepwisemlp':
        p_vals, t_vals, params, bses = ([[] for i in range(2)] for i in range(4))
    else:
        p_vals, t_vals, params, y_hat_Es, ys, bses = ([] for i in range(6))

    for state in range(n_states):
        s_idxs = [i for i, sample in enumerate(samples) if state not in sample] # [105]
        # prepare data for the regression analysis
        x_cat = np.concatenate((grid_dists[s_idxs].reshape((-1,1)), binary_phi[s_idxs].reshape((-1,1))),axis=1)
        # regression analysis
        x_cat = sm.add_constant(x_cat)
        if args.cortical_model == 'stepwisemlp':
            y_hat_Es = np.zeros(hidd_dists.shape)
            ys = np.zeros(hidd_dists.shape)
            for h in range(2):
                y = hidd_dists[s_idxs,h]
                ys[s_idxs,h] = y
                stats_model_cat = sm.OLS(y,x_cat).fit() 
                y_hat_Es[:,h] = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists)
                p_vals[h].append(stats_model_cat.pvalues)
                t_vals[h].append(stats_model_cat.tvalues)
                params[h].append(stats_model_cat.params)
                # y_hat_Es[h].append(y_hat_E)
                bses[h].append(stats_model_cat.bse)
        else:
            y = hidd_dists[s_idxs]
            ys.append(y)
            stats_model_cat = sm.OLS(y,x_cat).fit() 
            y_hat_E = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists)
            p_vals.append(stats_model_cat.pvalues)
            t_vals.append(stats_model_cat.tvalues)
            params.append(stats_model_cat.params)
            y_hat_Es.append(y_hat_E)
            bses.append(stats_model_cat.bse)
        states.append(state)
        
    # regression analysis - after removing (0,0) and (3,3)
    s_idxs = [i for i, sample in enumerate(samples) if ((0 not in sample) & (15 not in sample))] # [91]
    x_cat = np.concatenate((grid_dists[s_idxs].reshape((-1,1)), binary_phi[s_idxs].reshape((-1,1))),axis=1)
    x_cat = sm.add_constant(x_cat)
    if args.cortical_model == 'stepwisemlp':
            y_hat_Es = np.zeros(hidd_dists.shape)
            ys = np.zeros(hidd_dists.shape)
            for h in range(2):
                y = hidd_dists[s_idxs,h]
                ys[s_idxs,h] = y
                stats_model_cat = sm.OLS(y,x_cat).fit() 
                y_hat_Es[:,h] = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists)
                p_vals[h].append(stats_model_cat.pvalues)
                t_vals[h].append(stats_model_cat.tvalues)
                params[h].append(stats_model_cat.params)
                # y_hat_Es[h].append(y_hat_E)
                bses[h].append(stats_model_cat.bse)
    else:
        y = hidd_dists[s_idxs]
        ys.append(y)
        stats_model_cat = sm.OLS(y,x_cat).fit() 
        y_hat_E = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists)
        p_vals.append(stats_model_cat.pvalues)
        t_vals.append(stats_model_cat.tvalues)
        params.append(stats_model_cat.params)
        y_hat_Es.append(y_hat_E)
        bses.append(stats_model_cat.bse)
    states.append(16)
    
    # regression analysis - after removing (0,0) and (3,3), (3,0) and (0.3)
    s_idxs = [i for i, sample in enumerate(samples) if ((0 not in sample) & (15 not in sample) &
                                                        (3 not in sample) & (12 not in sample))] #[66]
    x_cat = np.concatenate((grid_dists[s_idxs].reshape((-1,1)), binary_phi[s_idxs].reshape((-1,1))),axis=1)
    x_cat = sm.add_constant(x_cat)
    if args.cortical_model == 'stepwisemlp':
            y_hat_Es = np.zeros(hidd_dists.shape)
            ys = np.zeros(hidd_dists.shape)
            for h in range(2):
                y = hidd_dists[s_idxs,h]  
                ys[s_idxs,h] = y
                y_hat_Es[:,h] = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists)
                p_vals[h].append(stats_model_cat.pvalues)
                t_vals[h].append(stats_model_cat.tvalues)
                params[h].append(stats_model_cat.params)
                # y_hat_Es[h].append(y_hat_E)
                bses[h].append(stats_model_cat.bse)
    else:
        y = hidd_dists[s_idxs]
        ys.append(y)
        stats_model_cat = sm.OLS(y,x_cat).fit() 
        y_hat_E = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists)
        p_vals.append(stats_model_cat.pvalues)
        t_vals.append(stats_model_cat.tvalues)
        params.append(stats_model_cat.params)
        y_hat_Es.append(y_hat_E)
        bses.append(stats_model_cat.bse)
    states.append(17)

    states = np.array(states)
    p_vals = np.array(p_vals)
    t_vals = np.array(t_vals)
    params = np.array(params)
    y_hat_Es = np.array(y_hat_Es)
    ys = np.array(ys)
    bses = np.array(bses)
    
    exc_reg_results = {'excluded_states': states,
                       'p_vals': p_vals,
                       't_vals': t_vals,
                       'params': params,
                       'y_hat_Es': y_hat_Es,
                       'ys': ys,
                       'bses': bses}                   

    return exc_reg_results

def analyze_test_seq(cortical_result):
    import sys
    sys.path.append("..")
    data = get_loaders(batch_size=32, meta=False,
                      use_images=True, image_dir='./images/',
                      n_episodes=None,
                      N_responses=None, N_contexts=None,
                      cortical_task='face_task')
    train_data, train_loader, test_data, test_loader, analyze_data, analyze_loader = data

    idx2loc = {idx:loc for loc, idx in test_data.loc2idx.items()}

    # ctx_order = 'first'
    # ctx_order_str = 'ctxF'
    
    analyze_correct = cortical_result['analyze_correct'] # [n_trials, time_steps]: [384, 3]
    analyze_correct = np.asarray(analyze_correct).squeeze()

    hidd_t_idx = 1 # at what time step, t = 1 means at the time of face1 
                                # and t = 2 means at the time of face2
                                # in axis First, it should be t = 1
    # create groups based on the row or columns
    # e.g, for context0 (xaxis), first column is group 1, sec col is group 2, and so on.
    # 4 groups for each axis/context; total 8 groups

    ctx0_g0=[]
    ctx0_g1=[]
    ctx0_g2=[]
    ctx0_g3=[]

    ctx1_g0=[]
    ctx1_g1=[]
    ctx1_g2=[]
    ctx1_g3=[]

    for i, batch in enumerate(analyze_loader):
        f1, f2, ax, y, idx1, idx2 = batch
        acc = analyze_correct[i][hidd_t_idx]
        ax = ax.cpu().numpy().squeeze()
        idx1 = idx1[0]
        idx2 = idx2[0]
        loc1 = idx2loc[idx1]
        loc2 = idx2loc[idx2]
        if ax==0:
            if loc1[ax]==0: ctx0_g0.append(acc) # (len(all_perms)/2) / 4 = [48]
            elif loc1[ax]==1: ctx0_g1.append(acc)
            elif loc1[ax]==2: ctx0_g2.append(acc)
            elif loc1[ax]==3: ctx0_g3.append(acc)
        elif ax==1:
            if loc1[ax]==0: ctx1_g0.append(acc)
            elif loc1[ax]==1: ctx1_g1.append(acc)
            elif loc1[ax]==2: ctx1_g2.append(acc)
            elif loc1[ax]==3: ctx1_g3.append(acc)
    ctx0_accs = [np.mean(ctx0_g0), np.mean(ctx0_g1), 
                np.mean(ctx0_g2), np.mean(ctx0_g3) ]
    ctx1_accs = [np.mean(ctx1_g0), np.mean(ctx1_g1), 
                np.mean(ctx1_g2), np.mean(ctx1_g3) ]
         
    # print('Accuracy at t=%s (face%s) contex 0:' %(hidd_t_idx,hidd_t_idx), ctx0_accs)
    # print('Accuracy at t=%s (face%s) contex 1:' %(hidd_t_idx,hidd_t_idx), ctx1_accs)
    return ctx0_accs, ctx1_accs

# do the analysis over multiple runs
def analyze_cortical_mruns(cortical_results, test_data, args):
    n_states = test_data.n_states 
    loc2idx = test_data.loc2idx 
    idx2loc = {idx:loc for loc, idx in loc2idx.items()}
    idxs = [idx for idx in range(n_states)]
    locs = [idx2loc[idx] for idx in idxs]
    analysis_type = args.analysis_type

    checkpoints = len(cortical_results[0])
    
    proportion_ress = [[] for i in range(args.nruns_cortical)]
    ratio_ress = [[] for i in range(args.nruns_cortical)]
    corr_ress = [[] for i in range(args.nruns_cortical)]
    dist_ress = [[] for i in range(args.nruns_cortical)]
    ttest_ress = [[] for i in range(args.nruns_cortical)]
    test_seq_ress = [[] for i in range(args.nruns_cortical)]
    reg_ress = [[] for i in range(args.nruns_cortical)]
    pca_results, tsne_results, mds_results = ({} for i in range(3))

    if ((analysis_type=='pca') | (analysis_type=='all')):
        res = cortical_results[-1][-1]
        samples_res = res['samples_res']
        pca_results = analyze_dim_red(args, res, 'pca', n_components=2)
        # mds_results = analyze_dim_red(args, res, 'mds', n_components=2)
        # tsne_results = analyze_dim_red(args, res, 'tsne', n_components=2)
        
        pca_results['grid_locations']  = locs
        pca_results['samples_res']  = samples_res
        # mds_results['grid_locations']  = locs
        # mds_results['samples_res']  = samples_res
        # tsne_results['grid_locations'] = locs
        # tsne_results['samples_res']  = samples_res
    
    for run in range(args.nruns_cortical):
        proportion_res = [[] for i in range(4)]
        ratio_res = [[] for i in range(2)]
        corr_res = [[] for i in range(4)]
        dist_res = [[] for i in range(8)]
        ttest_res = [[] for i in range(4)]
        reg_con_res = [[] for i in range(6)]
        # reg_con_res = {}
        reg_cat_res = [[] for i in range(6)]
        reg_exc_res = [[] for i in range(6)]
        reg_1D_res = [[] for i in range(6)]
        test_seq_res = [[] for i in range(2)]

        for cp in range(checkpoints):
            # load the results
            cortical_result = cortical_results[run][cp]
            dist_result = calc_dist(cortical_result, test_data, args)
            dist_ctx_result = calc_dist_ctx(cortical_result, test_data, args)
            dist_c_inc_result = hist_data(dist_result) 
            # distance results
            dist_res[0].append(dist_result['embed_dists'])
            dist_res[1].append(dist_result['hidd_dists'])
            dist_res[2].append(dist_result['grid_dists'])
            dist_res[3].append(dist_result['angle_results']['grid_angles'])
            dist_res[4].append(dist_c_inc_result['cong_embed_dist'])
            dist_res[5].append(dist_c_inc_result['incong_embed_dist'])
            dist_res[6].append(dist_c_inc_result['cong_hidd_dist'])
            dist_res[7].append(dist_c_inc_result['incong_hidd_dist'])
            
            # ratio analysis 
            if ((analysis_type=='ratio') | (analysis_type=='all')):
                ratio_result = calc_ratio(dist_result)
                ratio_res[0].append(ratio_result['ratio_embed'])
                ratio_res[1].append(ratio_result['ratio_hidd'])
            # t-test analysis
            if ((analysis_type=='ttest') | (analysis_type=='all')):
                ttest_result = analyze_ttest(args, dist_result)                
                ttest_res[0].append(ttest_result['t_stat_embed'])
                ttest_res[1].append(ttest_result['t_p_val_embed'])
                ttest_res[2].append(ttest_result['t_stat_hidd'])
                ttest_res[3].append(ttest_result['t_p_val_hidd'])
            # corretion analysis
            if ((analysis_type=='corr') | (analysis_type=='all')):
                corr_result = analyze_corr(args, dist_result)
                corr_res[0].append(corr_result['r_embed'])
                corr_res[1].append(corr_result['p_val_embed'])
                corr_res[2].append(corr_result['r_hidd'])
                corr_res[3].append(corr_result['p_val_hidd'])
            # proportion analysis
            if ((analysis_type=='proportions') | (analysis_type=='all')):
                prop_result = proportions(cortical_result)
                proportion_res[0].append(prop_result['hiddens_ctxs'])
                proportion_res[1].append(prop_result['p_pies'])
                proportion_res[2].append(prop_result['ps'])
                proportion_res[3].append(prop_result['n'])
            # regression analysis
            if ((analysis_type=='regs') | (analysis_type=='all')):
                reg_result = analyze_regression(args, dist_result)
                # categorical
                cat_reg = reg_result['cat_reg']
                reg_cat_res[0].append(cat_reg['p_val'])
                reg_cat_res[1].append(cat_reg['t_val'])
                reg_cat_res[2].append(cat_reg['param'])
                reg_cat_res[3].append(cat_reg['y'])
                reg_cat_res[4].append(cat_reg['y_hat_E'])
                reg_cat_res[5].append(cat_reg['bse'])
                # conitnuous
                con_reg = reg_result['con_reg']
                reg_con_res[0].append(con_reg['p_val'])
                reg_con_res[1].append(con_reg['t_val'])
                reg_con_res[2].append(con_reg['param'])
                reg_con_res[3].append(con_reg['y'])
                reg_con_res[4].append(con_reg['y_hat_E'])
                reg_con_res[5].append(con_reg['bse'])
                # reg_con_res['p_val'] = 
                # Excluded - Regression (do reg after removing each state)
                exc_reg = analyze_regression_exc(args, dist_result, test_data)
                reg_exc_res[0].append(exc_reg['p_vals'])
                reg_exc_res[1].append(exc_reg['t_vals'])
                reg_exc_res[2].append(exc_reg['params'])
                reg_exc_res[3].append(exc_reg['ys'])
                reg_exc_res[4].append(exc_reg['y_hat_Es'])
                reg_exc_res[5].append(exc_reg['bses'])
                # Including 1D rank difference in the model
                oneD_reg = analyze_regression_1D(args, dist_ctx_result)
                oneD_reg = oneD_reg['cat_reg']
                reg_1D_res[0].append(oneD_reg['p_val'])
                reg_1D_res[1].append(oneD_reg['t_val'])
                reg_1D_res[2].append(oneD_reg['param'])
                reg_1D_res[3].append(oneD_reg['y'])
                reg_1D_res[4].append(oneD_reg['y_hat_E'])
                reg_1D_res[5].append(oneD_reg['bse'])
            if ((analysis_type=='test_seq') | (analysis_type=='all')):
                ctx0_accs, ctx1_accs = analyze_test_seq(cortical_result)
                test_seq_res[0].append(ctx0_accs)
                test_seq_res[1].append(ctx1_accs)
        if ((analysis_type=='proportions') | (analysis_type=='all')):
            proportion_ress[run].append(proportion_res)
        if ((analysis_type=='corr') | (analysis_type=='all')):
            corr_ress[run].append(corr_res)
        if ((analysis_type=='ratio') | (analysis_type=='all')):
            ratio_ress[run].append(ratio_res)
        if ((analysis_type=='ttest') | (analysis_type=='all')):
            ttest_ress[run].append(ttest_res)
        if ((analysis_type=='regs') | (analysis_type=='all')):
            reg_res = [[] for i in range(4)]
            reg_res[0] = reg_cat_res
            reg_res[1] = reg_con_res
            reg_res[2] = reg_exc_res
            reg_res[3] = reg_1D_res
            reg_ress[run].append(reg_res)
        if ((analysis_type=='test_seq') | (analysis_type=='all')):
                test_seq_ress[run].append(test_seq_res)
        dist_ress[run].append(dist_res)
    dist_results={}
    dist_results['embed_dists'] = np.asarray([r[0][0] for r in dist_ress])
    dist_results['hidd_dists'] = np.asarray([r[0][1] for r in dist_ress])
    dist_results['grid_dists'] = np.asarray([r[0][2] for r in dist_ress])
    dist_results['grid_angles'] = np.asarray([r[0][3] for r in dist_ress])
    dist_results['cong_embed_dist'] = np.asarray([r[0][4] for r in dist_ress])
    dist_results['incong_embed_dist'] = np.asarray([r[0][5] for r in dist_ress])
    dist_results['cong_hidd_dist'] = np.asarray([r[0][6] for r in dist_ress])
    dist_results['incong_hidd_dist'] = np.asarray([r[0][7] for r in dist_ress])
    if ((analysis_type=='corr') | (analysis_type=='all')):
        corr_results = {}
        corr_results['r_embeds'] = np.asarray([r[0][0] for r in corr_ress])
        corr_results['p_val_embeds'] = np.asarray([r[0][1] for r in corr_ress])
        corr_results['r_hidds'] = np.asarray([r[0][2] for r in corr_ress])
        corr_results['p_val_hidds'] = np.asarray([r[0][3] for r in corr_ress])
        results = {'corr_results': corr_results, 
                   'dist_results': dist_results}
        with open('../results/'+'corr_'+args.out_file, 'wb') as f:
            pickle.dump(results, f)
    if ((analysis_type=='ratio') | (analysis_type=='all')):
        ratio_results = {}
        ratio_results['ratio_embeds'] = np.asarray([r[0][0] for r in ratio_ress])
        ratio_results['ratio_hidds'] = np.asarray([r[0][1] for r in ratio_ress])
        results = {'ratio_results': ratio_results, 
                   'dist_results': dist_results}
        with open('../results/'+'ratio_'+args.out_file, 'wb') as f:
            pickle.dump(results, f)
    if ((analysis_type=='ttest') | (analysis_type=='all')):
        ttest_results = {}
        ttest_results['t_stat_embed'] = np.asarray([r[0][0] for r in ttest_ress])
        ttest_results['t_p_val_embed'] = np.asarray([r[0][1] for r in ttest_ress])
        ttest_results['t_stat_hidd'] = np.asarray([r[0][2] for r in ttest_ress])
        ttest_results['t_p_val_hidd'] = np.asarray([r[0][3] for r in ttest_ress])
        results = {'ttest_results': ttest_results,
                   'dist_results': dist_results}
        with open('../results/'+'ttest_'+args.out_file, 'wb') as f:
            pickle.dump(results, f)
    if ((analysis_type=='pca') | (analysis_type=='all')): 
        results = {'pca_results': pca_results,
                   'dist_results': dist_results}
        with open('../results/'+'pca_'+args.out_file, 'wb') as f:
            pickle.dump(results, f)
    if ((analysis_type=='proportions') | (analysis_type=='all')): 
        proportion_results={}
        proportion_results['hiddens_ctxs'] = np.asarray([r[0][0] for r in proportion_ress])
        proportion_results['p_pies'] = np.asarray([r[0][1] for r in proportion_ress])
        proportion_results['ps'] = np.asarray([r[0][2] for r in proportion_ress])
        proportion_results['n'] = np.asarray([r[0][3] for r in proportion_ress])
        results = {'proportion_results': proportion_results,
                   'dist_results': dist_results}
        with open('../results/'+'proportion_'+args.out_file, 'wb') as f:
            pickle.dump(results, f)
    if ((analysis_type=='test_seq') | (analysis_type=='all')): 
        test_seq_results={}
        test_seq_results['ctx0_accs'] = np.asarray([r[0][0] for r in test_seq_ress])
        test_seq_results['ctx1_accs'] = np.asarray([r[0][1] for r in test_seq_ress])
        results = {'test_seq_results': test_seq_results,
                   'dist_results': dist_results}
        with open('../results/'+'test_seq_'+args.out_file, 'wb') as f:
            pickle.dump(results, f)
    if ((analysis_type=='regs') | (analysis_type=='all')):
        reg_ress = np.asarray(reg_ress).squeeze(1)

        cat_regs = {}
        cat_regs['p_vals'] = reg_ress[:,0,0,:]
        cat_regs['t_vals'] = reg_ress[:,0,1,:]
        cat_regs['params'] = reg_ress[:,0,2,:]
        cat_regs['ys'] = reg_ress[:,0,3,:]
        cat_regs['y_hat_Es'] = reg_ress[:,0,4,:]
        cat_regs['bses'] = reg_ress[:,0,5,:]
        con_regs = {}
        con_regs['p_vals'] = reg_ress[:,1,0,:]
        con_regs['t_vals'] = reg_ress[:,1,1,:]
        con_regs['params'] = reg_ress[:,1,2,:]
        con_regs['ys'] = reg_ress[:,1,3,:]
        con_regs['y_hat_Es'] = reg_ress[:,1,4,:]
        con_regs['bses'] = reg_ress[:,1,5,:]
        exc_regs = {}
        exc_regs['p_vals'] = reg_ress[:,2,0,:]
        exc_regs['t_vals'] = reg_ress[:,2,1,:]
        exc_regs['params'] = reg_ress[:,2,2,:]
        exc_regs['ys'] = reg_ress[:,2,3,:]
        exc_regs['y_hat_Es'] = reg_ress[:,2,4,:]
        exc_regs['bses'] = reg_ress[:,2,5,:]
        oneD_regs = {}
        oneD_regs['p_vals'] = reg_ress[:,3,0,:]
        oneD_regs['t_vals'] = reg_ress[:,3,1,:]
        oneD_regs['params'] = reg_ress[:,3,2,:]
        oneD_regs['ys'] = reg_ress[:,3,3,:]
        oneD_regs['y_hat_Es'] = reg_ress[:,3,4,:]
        oneD_regs['bses'] = reg_ress[:,3,5,:]
        reg_results = {'cat_regs': cat_regs,
                       'con_regs': con_regs,
                       'exc_regs': exc_regs,
                       'oneD_regs': oneD_regs}
        results = {'dist_results': dist_results,
                   'reg_results': reg_results}
        with open('../results/'+'reg_'+args.out_file, 'wb') as f:
            pickle.dump(results, f)
    
    return results




# def analyze_cortical_mruns1(cortical_results, test_data, args):
#     n_states = test_data.n_states 
#     loc2idx = test_data.loc2idx 
#     idx2loc = {idx:loc for loc, idx in loc2idx.items()}
#     idxs = [idx for idx in range(n_states)]
#     locs = [idx2loc[idx] for idx in idxs]
#     analysis_type = args.analysis_type

#     checkpoints = len(cortical_results[0])
#     r_hidds, p_val_hidds, r_embeds, p_val_embeds = ([] for i in range(4))
#     ratio_hidds, ratio_embeds = ([] for i in range(2))
#     t_stat_embeds, t_p_val_embeds, t_stat_hidds, t_p_val_hidds = \
#         ([] for i in range(4))
#     p_val_cat_regs, t_val_cat_regs, param_cat_regs, bse_cat_regs, \
#         y_cat_regs, y_hat_E_cat_regs = ([] for i in range(6))
#     p_val_con_regs, t_val_con_regs, param_con_regs, bse_con_regs, \
#         y_con_regs, y_hat_E_con_regs = ([] for i in range(6))
#     p_val_exc_regs, t_val_exc_regs, param_exc_regs, y_exc_regs, \
#             y_hat_E_exc_regs, bse_exc_regs = ([] for i in range(6))
#     p_val_1D_regs, t_val_1D_regs, param_1D_regs, y_1D_regs, \
#             y_hat_E_1D_regs, bse_1D_regs  = ([] for i in range(6))
#     cong_embed_dists, incong_embed_dists, cong_hidd_dists, \
#         incong_hidd_dists = ([] for i in range(4))
#     embed_dists, hidd_dists, grid_dists,  grid_angles \
#         = ([] for i in range(4))
#     proportion_results = [[] for i in range(args.nruns_cortical)]

#     corr_results, reg_results, ttest_results, ratio_results, = ({} for i in range(4))
#     pca_results, tsne_results, mds_results = ({} for i in range(3))

#     if ((analysis_type=='pca') | (analysis_type=='all')):
#         res = cortical_results[-1][-1]
#         samples_res = res['samples_res']
#         pca_results = analyze_dim_red(args, res, 'pca', n_components=3)
#         # mds_results = analyze_dim_red(args, res, 'mds', n_components=2)
#         # tsne_results = analyze_dim_red(args, res, 'tsne', n_components=2)
        
#         pca_results['grid_locations']  = locs
#         pca_results['samples_res']  = samples_res
#         # mds_results['grid_locations']  = locs
#         # mds_results['samples_res']  = samples_res
#         # tsne_results['grid_locations'] = locs
#         # tsne_results['samples_res']  = samples_res
    
#     for run in range(args.nruns_cortical):
#         r_hidd, p_val_hidd, r_embed, p_val_embed = ([] for i in range(4))
#         ratio_hidd, ratio_embed = ([] for i in range(2))
#         t_stat_embed, t_p_val_embed, t_stat_hidd, t_p_val_hidd \
#             = ([] for i in range(4))
#         p_val_cat_reg, t_val_cat_reg, param_cat_reg, bse_cat_reg, \
#             y_cat_reg, y_hat_E_cat_reg = ([] for i in range(6))
#         p_val_con_reg, t_val_con_reg, param_con_reg, bse_con_reg, \
#             y_con_reg, y_hat_E_con_reg  = ([] for i in range(6))
#         p_val_exc_reg, t_val_exc_reg, param_exc_reg, y_exc_reg, \
#             y_hat_E_exc_reg, bse_exc_reg  = ([] for i in range(6))
#         p_val_1D_reg, t_val_1D_reg, param_1D_reg, y_1D_reg, \
#             y_hat_E_1D_reg, bse_1D_reg  = ([] for i in range(6))
#         cong_embed_dist, incong_embed_dist, cong_hidd_dist, \
#              incong_hidd_dist = ([] for i in range(4))
#         embed_dist, hidd_dist, grid_dist, grid_angle \
#             = ([] for i in range(4))
#         proportion_result = [[] for i in range(4)]
        
#         for cp in range(checkpoints):
#             # load the results
#             cortical_res = cortical_results[run][cp]
#             dist_results = calc_dist(cortical_res, test_data, args)
#             dist_ctx_results = calc_dist_ctx(cortical_res, test_data, args)
#             dist_c_inc_results = hist_data(dist_results) 
#             # distance results
#             embed_dist.append(dist_results['embed_dists'])
#             hidd_dist.append(dist_results['hidd_dists'])
#             grid_dist.append(dist_results['grid_dists'])
#             grid_angle.append(dist_results['angle_results']['grid_angles'])
#             cong_embed_dist.append(dist_c_inc_results['cong_embed_dist'])
#             incong_embed_dist.append(dist_c_inc_results['incong_embed_dist'])
#             cong_hidd_dist.append(dist_c_inc_results['cong_hidd_dist'])
#             incong_hidd_dist.append(dist_c_inc_results['incong_hidd_dist'])

#             # ratio analysis 
#             if ((analysis_type=='ratio') | (analysis_type=='all')):
#                 ratio_results = calc_ratio(dist_results)
#                 # embeddings
#                 ratio_embed.append(ratio_results['ratio_embed'])
#                 # hiddens
#                 ratio_hidd.append(ratio_results['ratio_hidd'])

#             # t-test analysis
#             if ((analysis_type=='ttest') | (analysis_type=='all')):
#                 ttest_results = analyze_ttest(dist_results)                
#                 t_stat_embed.append(ttest_results['t_stat_embed'])
#                 t_p_val_embed.append(ttest_results['t_p_val_embed'])
#                 t_stat_hidd.append(ttest_results['t_stat_hidd'])
#                 t_p_val_hidd.append(ttest_results['t_p_val_hidd'])

#             # corretion analysis
#             if ((analysis_type=='corr') | (analysis_type=='all')):
#                 corr_results = analyze_corr(args, dist_results)
#                 # embeddings
#                 r_embed.append(corr_results['r_embed'])
#                 p_val_embed.append(corr_results['p_val_embed'])
#                 # hiddens
#                 r_hidd.append(corr_results['r_hidd'])
#                 p_val_hidd.append(corr_results['p_val_hidd'])
#             # proportion analysis
#             if ((analysis_type=='proportions') | (analysis_type=='all')):
#                 prop_res = proportions(cortical_res)
#                 proportion_result[0].append(prop_res['hiddens_ctxs'])
#                 proportion_result[1].append(prop_res['p_pies'])
#                 proportion_result[2].append(prop_res['ps'])
#                 proportion_result[3].append(prop_res['n'])

#             # regression analysis
#             if ((analysis_type=='regs') | (analysis_type=='all')):
#                 reg_results = analyze_regression(dist_results)
#                 # categorical
#                 cat_reg = reg_results['cat_reg']
#                 p_val_cat_reg.append(cat_reg['p_val'])
#                 t_val_cat_reg.append(cat_reg['t_val'])
#                 param_cat_reg.append(cat_reg['param'])
#                 y_cat_reg.append(cat_reg['y'])
#                 y_hat_E_cat_reg.append(cat_reg['y_hat_E'])
#                 bse_cat_reg.append(cat_reg['bse'])
#                 # conitnuous
#                 con_reg = reg_results['con_reg']
#                 p_val_con_reg.append(con_reg['p_val'])
#                 t_val_con_reg.append(con_reg['t_val'])
#                 param_con_reg.append(con_reg['param'])
#                 y_con_reg.append(con_reg['y'])
#                 y_hat_E_con_reg.append(con_reg['y_hat_E'])
#                 bse_con_reg.append(con_reg['bse'])
#                 # Excluded - Regression (do reg after removing each state)
#                 exc_reg = analyze_regression_exc(dist_results, test_data)
#                 p_val_exc_reg.append(exc_reg['p_vals'])
#                 t_val_exc_reg.append(exc_reg['t_vals'])
#                 param_exc_reg.append(exc_reg['params'])
#                 y_exc_reg.append(exc_reg['ys'])
#                 y_hat_E_exc_reg.append(exc_reg['y_hat_Es'])
#                 bse_exc_reg.append(exc_reg['bses'])
#                 # Including 1D rank difference in the model
#                 oneD_reg = analyze_regression_1D(dist_ctx_results)
#                 oneD_reg = oneD_reg['cat_reg']
#                 p_val_1D_reg.append(oneD_reg['p_val'])
#                 t_val_1D_reg.append(oneD_reg['t_val'])
#                 param_1D_reg.append(oneD_reg['param'])
#                 y_1D_reg.append(oneD_reg['y'])
#                 y_hat_E_1D_reg.append(oneD_reg['y_hat_E'])
#                 bse_1D_reg.append(oneD_reg['bse'])

#         if ((analysis_type=='proportions') | (analysis_type=='all')):
#             proportion_results[run].append(proportion_result)
#         if ((analysis_type=='corr') | (analysis_type=='all')):
#             r_hidds.append(r_hidd)
#             p_val_hidds.append(p_val_hidd)
#             r_embeds.append(r_embed)
#             p_val_embeds.append(p_val_embed)
#         if ((analysis_type=='ratio') | (analysis_type=='all')):
#             ratio_hidds.append(ratio_hidd)
#             ratio_embeds.append(ratio_embed)
#         if ((analysis_type=='ttest') | (analysis_type=='all')):
#             t_stat_embeds.append(t_stat_embed)
#             t_p_val_embeds.append(t_p_val_embed)
#             t_stat_hidds.append(t_stat_hidd)
#             t_p_val_hidds.append(t_p_val_hidd)
#         if ((analysis_type=='regs') | (analysis_type=='all')):
#             p_val_cat_regs.append(p_val_cat_reg)
#             t_val_cat_regs.append(t_val_cat_reg)
#             param_cat_regs.append(param_cat_reg)
#             y_cat_regs.append(y_cat_reg)
#             y_hat_E_cat_regs.append(y_hat_E_cat_reg)
#             bse_cat_regs.append(bse_cat_reg)
#             p_val_con_regs.append(p_val_con_reg)
#             t_val_con_regs.append(t_val_con_reg)
#             param_con_regs.append(param_con_reg)
#             y_con_regs.append(y_con_reg)
#             y_hat_E_con_regs.append(y_hat_E_con_reg)
#             bse_con_regs.append(bse_con_reg)
#             p_val_exc_regs.append(p_val_exc_reg)
#             t_val_exc_regs.append(t_val_exc_reg)
#             param_exc_regs.append(param_exc_reg)
#             y_exc_regs.append(y_exc_reg)
#             y_hat_E_exc_regs.append(y_hat_E_exc_reg)
#             bse_exc_regs.append(bse_exc_reg)
#             p_val_1D_regs.append(p_val_1D_reg)
#             t_val_1D_regs.append(t_val_1D_reg)
#             param_1D_regs.append(param_1D_reg)
#             y_1D_regs.append(y_1D_reg)
#             y_hat_E_1D_regs.append(y_hat_E_1D_reg)
#             bse_1D_regs.append(bse_1D_reg)    
#         cong_embed_dists.append(cong_embed_dist)
#         incong_embed_dists.append(incong_embed_dist)
#         cong_hidd_dists.append(cong_hidd_dist)
#         incong_hidd_dists.append(incong_hidd_dist)
#         embed_dists.append(embed_dist)
#         hidd_dists.append(hidd_dist)
#         grid_dists.append(grid_dist)
#         grid_angles.append(grid_angle)

#     cong_embed_dists = np.array(cong_embed_dists)
#     incong_embed_dists = np.array(incong_embed_dists)
#     cong_hidd_dists = np.array(cong_hidd_dists)
#     incong_hidd_dists = np.array(incong_hidd_dists)
#     embed_dists = np.array(embed_dists)
#     hidd_dists = np.array(hidd_dists)
#     grid_dists = np.array(grid_dists)
#     grid_angles = np.array(grid_angles)
#     dist_results = {'embed_dists': embed_dists, 'hidd_dists': hidd_dists,
#                     'grid_dists': grid_dists, 'grid_angles': grid_angles,
#                     'cong_embed_dists': cong_embed_dists, 'incong_embed_dists': incong_embed_dists,
#                     'cong_hidd_dists': cong_hidd_dists, 'incong_hidd_dists': incong_hidd_dists}

#     if ((analysis_type=='corr') | (analysis_type=='all')):
#         r_hidds = np.array(r_hidds)
#         p_val_hidds = np.array(p_val_hidds)
#         r_embeds = np.array(r_embeds)
#         p_val_embeds = np.array(p_val_embeds)
#         corr_results = {'r_embeds': r_embeds, 
#                         'p_val_embeds': p_val_embeds,
#                         'r_hidds': r_hidds, 
#                         'p_val_hidds': p_val_hidds}
#         results = {'corr_results': corr_results, 
#                    'dist_results': dist_results}
#         with open('../results/'+'corr_'+args.out_file, 'wb') as f:
#             pickle.dump(results, f)
#     if ((analysis_type=='ratio') | (analysis_type=='all')):
#         ratio_hidds = np.array(ratio_hidds)
#         ratio_embeds = np.array(ratio_embeds)
#         ratio_results = {'ratio_embeds': ratio_embeds, 
#                          'ratio_hidds': ratio_hidds}
#         results = {'ratio_results': ratio_results, 
#                    'dist_results': dist_results}
#         with open('../results/'+'ratio_'+args.out_file, 'wb') as f:
#             pickle.dump(results, f)
#     if ((analysis_type=='ttest') | (analysis_type=='all')):
#         t_stat_embeds = np.array(t_stat_embeds)
#         t_p_val_embeds = np.array(t_p_val_embeds)
#         t_stat_hidds = np.array(t_stat_hidds)
#         t_p_val_hidds = np.array(t_p_val_hidds)
#         ttest_results = {'t_stat_hidds': t_stat_hidds,
#                          't_p_val_hidds': t_p_val_hidds,
#                          't_stat_embeds': t_stat_embeds, 
#                          't_p_val_embeds': t_p_val_embeds}
#         results = {'ttest_results': ttest_results,
#                    'dist_results': dist_results}
#         with open('../results/'+'ttest_'+args.out_file, 'wb') as f:
#             pickle.dump(results, f)
#     if ((analysis_type=='pca') | (analysis_type=='all')): 
#         results = {'pca_results': pca_results,
#                    'dist_results': dist_results}
#         with open('../results/'+'pca_'+args.out_file, 'wb') as f:
#             pickle.dump(results, f)
#     if ((analysis_type=='proportions') | (analysis_type=='all')): 
#         prop_results={}
#         prop_results['hiddens_ctxs'] = np.asarray([r[0][0] for r in proportion_results])
#         prop_results['p_pies'] = np.asarray([r[0][1] for r in proportion_results])
#         prop_results['ps'] = np.asarray([r[0][2] for r in proportion_results])
#         prop_results['n'] = np.asarray([r[0][3] for r in proportion_results])
#         results = {'proportion_results': prop_results,
#                    'dist_results': dist_results}
#         with open('../results/'+'proportion_'+args.out_file, 'wb') as f:
#             pickle.dump(results, f)

#     if ((analysis_type=='regs') | (analysis_type=='all')):
#         p_val_cat_regs = np.array(p_val_cat_regs)
#         t_val_cat_regs = np.array(t_val_cat_regs)
#         param_cat_regs = np.array(param_cat_regs)
#         y_cat_regs = np.array(y_cat_regs)
#         y_hat_E_cat_regs = np.array(y_hat_E_cat_regs)
#         bse_cat_regs = np.array(bse_cat_regs)
#         p_val_con_regs = np.array(p_val_con_regs)
#         t_val_con_regs = np.array(t_val_con_regs)
#         param_con_regs = np.array(param_con_regs)
#         y_con_regs = np.array(y_con_regs)
#         y_hat_E_con_regs = np.array(y_hat_E_con_regs)
#         bse_con_regs = np.array(bse_con_regs)
#         p_val_exc_regs = np.array(p_val_exc_regs)
#         t_val_exc_regs = np.array(t_val_exc_regs)
#         param_exc_regs = np.array(param_exc_regs)
#         y_exc_regs = np.array(y_exc_regs)
#         y_hat_E_exc_regs = np.array(y_hat_E_exc_regs)
#         bse_exc_regs = np.array(bse_exc_regs)
#         p_val_1D_regs = np.array(p_val_1D_regs)
#         t_val_1D_regs = np.array(t_val_1D_regs)
#         param_1D_regs = np.array(param_1D_regs)
#         y_1D_regs = np.array(y_1D_regs)
#         y_hat_E_1D_regs = np.array(y_hat_E_1D_regs)
#         bse_1D_regs = np.array(bse_1D_reg)
#         cat_regs = {'p_vals': p_val_cat_regs,
#                     't_vals': t_val_cat_regs,
#                     'params': param_cat_regs,
#                     'ys': y_cat_regs,
#                     'y_hat_Es': y_hat_E_cat_regs,
#                     'bses': bse_cat_regs}
#         con_regs = {'p_vals': p_val_con_regs,
#                     't_vals': t_val_con_regs,
#                     'params': param_con_regs,
#                     'ys': y_con_regs,
#                     'y_hat_Es': y_hat_E_con_regs,
#                     'bses': bse_con_regs}
#         exc_regs = {'p_vals': p_val_exc_regs,
#                     't_vals': t_val_exc_regs,
#                     'params': param_exc_regs,
#                     'ys': y_exc_regs,
#                     'y_hat_Es': y_hat_E_exc_regs,
#                     'bses': bse_exc_regs}
#         oneD_regs = {'p_vals': p_val_1D_regs,
#                     't_vals': t_val_1D_regs,
#                     'params': param_1D_regs,
#                     'ys': y_1D_regs,
#                     'y_hat_Es': y_hat_E_1D_regs,
#                     'bses': bse_1D_regs}
#         reg_results = {'cat_regs': cat_regs,
#                        'con_regs': con_regs,
#                        'exc_regs': exc_regs,
#                        'oneD_regs': oneD_regs}
#         results = {'dist_results': dist_results,
#                    'reg_results': reg_results}
#         with open('../results/'+'reg_'+args.out_file, 'wb') as f:
#             pickle.dump(results, f)
    
    

#     return results


#     # do the analysis over multiple runs
