from numpy.core.fromnumeric import reshape
import torch 
import numpy as np
import pickle
from itertools import combinations, permutations
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from scipy.stats import pearsonr, ttest_ind
import statsmodels.api as sm

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
        test_f1, test_f2, test_ax, test_y = test[i]

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
            train_f1, train_f2, train_ax, train_y = sample
            if train_ax != test_ax:
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
    hiddens_ctx0 = [] # hidden reps. for context0 (when axis = 0)
    hiddens_ctx1 = [] # hidden reps. for context1 (when axis = 1)
    hiddens = [] # hidden reps. for both contexts
    hiddens_incong = []
    hiddens_cong = []
    idxs1 = []
    idxs2 = []
    idxs1_ctx0 = []
    idxs1_ctx1 = []
    idxs2_ctx0 = []
    idxs2_ctx1 = []
    samples = []
    samples_ctx0 = []
    samples_ctx1 = []
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
            # if model.analyze:
            f1, f2, ax, y, idx1, idx2 = batch
            idx1 = idx1[0]
            idx2 = idx2[0]
            samples.append(batch)
            (x1, y1), (x2, y2) = idx2loc[idx1], idx2loc[idx2]
            f1 = f1.to(args.device) 
            f2 = f2.to(args.device) 
            ax = ax.to(args.device)

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
            y_hat, out = model(f1, f2, ax) 
            # y_hat: [1, 2]
            # rnn_out: [seq_length, 1, hidden_dim]
            # mlp_out: dict len [2]: hidd_b/a: [1, hidden_dim]: [1, 128]
            if args.order_ax == 'first':
                f1_ind = 1
                f2_ind = 2
                ax_ind = 0
            elif args.order_ax == 'last':
                f1_ind = 0
                f2_ind = 1
                ax_ind = 2

            if args.cortical_model=='rnn':
                out = out.squeeze().unsqueeze(0).unsqueeze(0)
            elif args.cortical_model=='mlp':
                if args.before_ReLU:
                    out = out['hidd_b'] # hidden reps. befor the ReLU
                else:
                    out = out['hidd_a'] # hidden reps. after the ReLU
                out = out.unsqueeze(0)
            out = out.cpu().numpy()
            ax = ax.cpu().numpy()
            idxs1.append(idx1)
            idxs2.append(idx2)
            hiddens.append(out)
            if ax==0:
                hiddens_ctx0.append(out)
                idxs1_ctx0.append(idx1)
                idxs2_ctx0.append(idx2)
                samples_ctx0.append(batch)
            elif ax==1:
                hiddens_ctx1.append(out)
                idxs1_ctx1.append(idx1)
                idxs2_ctx1.append(idx2)
                samples_ctx1.append(batch)
            if cong==1:
                hiddens_cong.append(out)
                samples_cong.append(batch)
            elif cong==-1:
                hiddens_incong.append(out)
                samples_incong.append(batch)


    hiddens = np.concatenate(hiddens, axis = 0) 
    # data_len = 16*12*2=384 (n_states:16, n_states-ties:12, permutation:2)
    # rnn hiddens: [data_len, 1, seq_length, hidden_dim] : [384, 1, 3, 128]
    # mlp hiddens: [data_len, 1, hidden_dim] : [384, 1, 128]
    hiddens_ctx0 = np.concatenate(hiddens_ctx0, axis = 0)
    hiddens_ctx1 = np.concatenate(hiddens_ctx1, axis = 0)
    # rnn hiddens_ctx0/_ctx1: [data_len/2, 1, seq_length, hidden_dim] : [192, 1, 3, 128]
    # mlp hiddens_ctx0/_ctx1: [data_len/2, 1, hidden_dim] : [192, 1, 128]
    hiddens_incong = np.concatenate(hiddens_incong, axis = 0) 
    hiddens_cong = np.concatenate(hiddens_cong, axis = 0) 
    # rnn hiddens_ctx0/_ctx1: [data_len/2-ties, 1, seq_length, hidden_dim] : [144, 1, 3, 128]
    # mlp hiddens_ctx0/_ctx1: [data_len/2-ties, 1, hidden_dim] : [144, 1, 128]
    
    if args.cortical_model=='rnn':
        hiddens_ctx = np.concatenate((hiddens_ctx0, hiddens_ctx1), axis=0)
        hiddens_ctx = hiddens_ctx[:, :, -1, :].squeeze()
        hiddens_inc_c =  np.concatenate((hiddens_incong, hiddens_cong), axis=0)
        hiddens_inc_c = hiddens_inc_c[:, :, -1, :].squeeze()
    elif args.cortical_model=='mlp':
        hiddens_ctx = np.concatenate((hiddens_ctx0, hiddens_ctx1), axis=0).squeeze()
        hiddens_inc_c =  np.concatenate((hiddens_incong, hiddens_cong), axis=0).squeeze()
    samples_ctx = np.concatenate((samples_ctx0, samples_ctx1), axis=0)
    samples_inc_c = np.concatenate((samples_incong, samples_cong), axis=0)
    # hiddens_ctx: [384, 128]
    # hiddens_inc_c: [384-ties, 128]: [288, 128]
    
    avg_hidden = np.zeros([n_states, hiddens.shape[-1]])
    avg_hidden_ctx0 = np.zeros([n_states, hiddens_ctx0.shape[-1]])
    avg_hidden_ctx1 = np.zeros([n_states, hiddens_ctx1.shape[-1]])

    if args.cortical_model=='rnn':
        # Take average for each face based on its location
        for f in range(n_states):
            temp1 = [hiddens[i,:,f1_ind,:] 
                        for i, idx1 in enumerate(idxs1) if idx1==f]
            temp2 = [hiddens[i,:,f2_ind,:] 
                        for i, idx2 in enumerate(idxs2) if idx2==f]         
            temp1_ctx0 = [hiddens_ctx0[i,:,f1_ind,:] 
                            for i, idx1 in enumerate(idxs1_ctx0) if idx1==f]
            temp2_ctx0 = [hiddens_ctx0[i,:,f2_ind,:] 
                            for i, idx2 in enumerate(idxs2_ctx0) if idx2==f]
            temp1_ctx1 = [hiddens_ctx1[i,:,f1_ind,:] 
                            for i, idx1 in enumerate(idxs1_ctx1) if idx1==f]
            temp2_ctx1 = [hiddens_ctx1[i,:,f2_ind,:] 
                            for i, idx2 in enumerate(idxs2_ctx1) if idx2==f]
            if len(temp1 + temp2)>1:
                avg_hidden[f] = np.concatenate(temp1 + temp2, axis=0).mean(axis=0)
            if len(temp1_ctx0 + temp2_ctx0)>1:
                avg_hidden_ctx0[f] = np.concatenate(temp1_ctx0 + temp2_ctx0, axis=0).mean(axis=0)
            if len(temp1_ctx1 + temp2_ctx1)>1:
                avg_hidden_ctx1[f] = np.concatenate(temp1_ctx1 + temp2_ctx1, axis=0).mean(axis=0)
        avg_hidden_ctx = np.concatenate((avg_hidden_ctx0, avg_hidden_ctx1))
        # avg_hidden_ctx: [n_states*2, hidden_dim]: [32, 128] 
    elif args.cortical_model=='mlp':
        for f in range(n_states):
            temp = [hiddens[i,:,:] 
                        for i, (idx1, idx2) in enumerate(zip(idxs1, idxs2))
                            if ((idx1==f) | (idx2==f))]
            temp_ctx0 = [hiddens_ctx0[i,:,:] 
                            for i, (idx1, idx2) in enumerate(zip(idxs1_ctx0, idxs2_ctx0))
                            if ((idx1==f) | (idx2==f))]
            temp_ctx1 = [hiddens_ctx1[i,:,:] 
                            for i, (idx1, idx2) in enumerate(zip(idxs1_ctx1, idxs2_ctx1))
                            if ((idx1==f) | (idx2==f))]
            if len(temp)>1:
                avg_hidden[f] = np.mean(temp, axis=0)
            if len(temp_ctx0)>1:
                avg_hidden_ctx0[f] = np.mean(temp_ctx0, axis=0)
            if len(temp_ctx1)>1:
                avg_hidden_ctx1[f] = np.mean(temp_ctx1, axis=0)
        avg_hidden_ctx = np.concatenate((avg_hidden_ctx0, avg_hidden_ctx1))
        # avg_hidden_ctx: [n_states*2, hidden_dim]: [32, 128] 
    samples_res = {'samples': samples, 
                  'samples_ctx': samples_ctx,
                  'samples_inc_c': samples_inc_c}

    results = {'samples_res':samples_res,
                'idxs1': idxs1, 'idxs2': idxs2,
                'embeddings': embeddings, 
                'hiddens_ctx':hiddens_ctx,
                'avg_hidden':avg_hidden, 
                'avg_hidden_ctx':avg_hidden_ctx,
                'avg_hidden_ctx0': avg_hidden_ctx0, 
                'avg_hidden_ctx1': avg_hidden_ctx1,
                'hiddens_inc_c': hiddens_inc_c}
    
    return results

def calc_dist_ctx(cortical_results, test_data):
    # Useful dictionaries from test dataset
    n_states = test_data.n_states 
    loc2idx = test_data.loc2idx 
    idx2loc = {idx:loc for loc, idx in loc2idx.items()}
    idxs = [idx for idx in range(n_states)]

    # Correlation
    grid_dists = []
    hidd_dists_ctx0 = []
    hidd_dists_ctx1 = []
    grid_1ds_ctx0 = []
    grid_1ds_ctx1 = []
    grid_angles = []
    samples = []

    avg_hidden_ctx0 =  cortical_results['avg_hidden_ctx0']
    avg_hidden_ctx1 =  cortical_results['avg_hidden_ctx1']

    for idx1, idx2 in combinations(idxs, 2):
        (x1, y1), (x2, y2) = idx2loc[idx1], idx2loc[idx2]
        samples.append((idx1, idx2))
        grid_dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        grid_dists.append(grid_dist)
        # Euclidean distance between hidden reps. in context0
        hidd1, hidd2 = avg_hidden_ctx0[idx1], avg_hidden_ctx0[idx2]
        hidd_dist = np.linalg.norm(hidd1 - hidd2)
        hidd_dists_ctx0.append(hidd_dist)
        # Euclidean distance between hidden reps. in context1
        hidd1, hidd2 = avg_hidden_ctx1[idx1], avg_hidden_ctx1[idx2]
        hidd_dist = np.linalg.norm(hidd1 - hidd2)
        hidd_dists_ctx1.append(hidd_dist)
        # 1D rank - Manhattan distance
        grid_1ds_ctx0.append(np.abs(x1-x2))
        grid_1ds_ctx1.append(np.abs(y1-y2))
        # create on and off diagonal groups
        grid_angle = np.arctan2((y2-y1),(x2-x1))
        grid_angles.append(grid_angle)
        
    grid_dists = np.array(grid_dists) # [(n_states*(nstates-1))/2]: [120]
    grid_angles = np.array(grid_angles) # [120]
    samples = np.array(samples)
    hidd_dists_ctx0 = np.array(hidd_dists_ctx0)
    hidd_dists_ctx1 = np.array(hidd_dists_ctx1)
    grid_1ds_ctx0 = np.array(grid_1ds_ctx0)
    grid_1ds_ctx1 = np.array(grid_1ds_ctx1)

    phi = np.sin(2*grid_angles)
    binary_phi = np.sign(phi)
    for i, p in enumerate(phi):
        if np.abs(p)<1e-5:
            binary_phi[i] = 0

    angle_results = {'grid_angles': grid_angles,
                     'phi': phi,
                     'binary_phi': binary_phi}
    dist_results = {'samples': samples,
                    'hidd_dists_ctx0': hidd_dists_ctx0,
                    'hidd_dists_ctx1': hidd_dists_ctx1,
                    'grid_1ds_ctx0': grid_1ds_ctx0,
                    'grid_1ds_ctx1': grid_1ds_ctx1,
                    'grid_dists': grid_dists,
                    'angle_results': angle_results}
    return dist_results

def calc_dist(cortical_results, test_data):
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
    avg_hidden =  cortical_results['avg_hidden']

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
    avg_cong_hidd = np.mean(cong_hidd_dists)
    avg_incong_hidd = np.mean(incong_hidd_dists)
    ratio_hidd = (avg_cong_hidd/avg_incong_hidd)
    
    ratio_results = {'ratio_embed': ratio_embed, 'ratio_hidd': ratio_hidd}
    
    return ratio_results

def analyze_dim_red(cortical_results, method, n_components):
    embeddings = cortical_results['embeddings']
    hiddens_ctx = cortical_results['hiddens_ctx']
    avg_hidden = cortical_results['avg_hidden']
    avg_hidden_ctx = cortical_results['avg_hidden_ctx']
    avg_hidden_ctx0 = cortical_results['avg_hidden_ctx0']
    avg_hidden_ctx1 = cortical_results['avg_hidden_ctx1']
    hiddens_inc_c = cortical_results['hiddens_inc_c']

    results = {}
    # PCA
    if method == 'pca':
        pca = PCA(n_components=n_components)
        pca_2d_embed = pca.fit_transform(embeddings)
        pca_2d_hidd = pca.fit_transform(hiddens_ctx) # this is all the hiddens, no averaging for each face
        pca_2d_avg_hidd = pca.fit_transform(avg_hidden) # I might need to save this at all
        pca_2d_ctx_hidd = pca.fit_transform(avg_hidden_ctx)
        pca_2d_ctx0_hidd = pca.fit_transform(avg_hidden_ctx0)
        pca_2d_ctx1_hidd = pca.fit_transform(avg_hidden_ctx1)
        pca_2d_incong_cong = pca.fit_transform(hiddens_inc_c)
        results = {'embed_2d': pca_2d_embed, 
                   'hidd_2d': pca_2d_hidd,
                   'avg_hidd_2d': pca_2d_avg_hidd,
                   'ctx_hidd_2d': pca_2d_ctx_hidd,
                   'ctx0_hidd_2d': pca_2d_ctx0_hidd, 
                   'ctx1_hidd_2d': pca_2d_ctx1_hidd, 
                   'incong_cong_2d': pca_2d_incong_cong}
    elif method == 'mds':
        # MDS
        mds = MDS(n_components=n_components)
        mds_2d_embed = mds.fit_transform(embeddings)
        mds_2d_hidd = mds.fit_transform(hiddens_ctx) # this is all the hiddens, no averaging for each face
        mds_2d_avg_hidd = mds.fit_transform(avg_hidden) # I might need to save this at all
        mds_2d_ctx_hidd = mds.fit_transform(avg_hidden_ctx)
        mds_2d_ctx0_hidd = mds.fit_transform(avg_hidden_ctx0)
        mds_2d_ctx1_hidd = mds.fit_transform(avg_hidden_ctx1)
        mds_2d_incong_cong = mds.fit_transform(hiddens_inc_c)
        results = {'embed_2d': mds_2d_embed, 
                    'hidd_2d': mds_2d_hidd,
                    'avg_hidd_2d': mds_2d_avg_hidd,
                    'ctx_hidd_2d': mds_2d_ctx_hidd,
                    'ctx0_hidd_2d': mds_2d_ctx0_hidd, 
                    'ctx1_hidd_2d': mds_2d_ctx1_hidd, 
                    'incong_cong_2d': mds_2d_incong_cong}
    elif method == 'tsne':
        # tSNE
        tsne = TSNE(n_components=n_components)
        tsne_2d_embed = tsne.fit_transform(embeddings)
        tsne_2d_hidd = tsne.fit_transform(hiddens_ctx) # this is all the hiddens, no averaging for each face
        tsne_2d_avg_hidd = tsne.fit_transform(avg_hidden) # I might need to save this at all
        tsne_2d_ctx_hidd = tsne.fit_transform(avg_hidden_ctx)
        tsne_2d_ctx0_hidd = tsne.fit_transform(avg_hidden_ctx0)
        tsne_2d_ctx1_hidd = tsne.fit_transform(avg_hidden_ctx1)
        tsne_2d_incong_cong = tsne.fit_transform(hiddens_inc_c)
        results = {'embed_2d': tsne_2d_embed, 
                    'hidd_2d': tsne_2d_hidd,
                    'avg_hidd_2d': tsne_2d_avg_hidd,
                    'ctx_hidd_2d': tsne_2d_ctx_hidd,
                    'ctx0_hidd_2d': tsne_2d_ctx0_hidd, 
                    'ctx1_hidd_2d': tsne_2d_ctx1_hidd, 
                    'incong_cong_2d': tsne_2d_incong_cong}
    return results

def analyze_ttest(dist_results):  
    cong_res = dist_results['cong_dist_results']
    incong_res = dist_results['incong_dist_results']
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

def analyze_corr(dist_results):
    grid_dists = dist_results['grid_dists']
    embed_dists = dist_results['embed_dists'] 
    hidd_dists = dist_results['hidd_dists']    
    cong_res = dist_results['cong_dist_results']
    incong_res = dist_results['incong_dist_results']
    r_embed, p_val_embed = pearsonr(grid_dists, embed_dists)
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

def analyze_regression(dist_results):
    hidd_dists = dist_results['hidd_dists']
    grid_dists = dist_results['grid_dists']
    
    phi = dist_results['angle_results']['phi']
    binary_phi = dist_results['angle_results']['binary_phi']
    
            
    # prepare data for the regression analysis
    x_cat = np.concatenate((grid_dists.reshape((-1,1)), binary_phi.reshape((-1,1))),axis=1)
    x_con = np.concatenate((grid_dists.reshape((-1,1)), phi.reshape((-1,1))),axis=1)

    y = hidd_dists

    # categorical regression analysis
    x_cat = sm.add_constant(x_cat)
    stats_model_cat = sm.OLS(y,x_cat).fit() 
    y_hat_E = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists)
    cat_reg = {'p_val': stats_model_cat.pvalues,
               't_val': stats_model_cat.tvalues,
               'param': stats_model_cat.params,
               'y_hat_E': y_hat_E,
               'y': y,
               'bse': stats_model_cat.bse}
    # continuous regression analysis
    x_con = sm.add_constant(x_con)
    stats_model_con = sm.OLS(y,x_con).fit()
    y_hat_E = stats_model_con.params[0] + (stats_model_con.params[1]*grid_dists)
    con_reg = {'p_val': stats_model_con.pvalues,
               't_val': stats_model_con.tvalues,
               'param': stats_model_con.params,
               'y_hat_E': y_hat_E,
               'y': y,
               'bse': stats_model_con.bse}

    reg_results = {'cat_reg': cat_reg, 
                   'con_reg': con_reg}
    return reg_results

def analyze_regression_1D(dist_ctx_results):
    hidd_dists_ctx0 = dist_ctx_results['hidd_dists_ctx0']
    hidd_dists_ctx1 = dist_ctx_results['hidd_dists_ctx1']
    grid_1ds_ctx0 = dist_ctx_results['grid_1ds_ctx0']
    grid_1ds_ctx1 = dist_ctx_results['grid_1ds_ctx1']
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

    y = hidd_dists_ctx

    # categorical regression analysis
    x_cat = sm.add_constant(x_cat)
    stats_model_cat = sm.OLS(y,x_cat).fit() 
    y_hat_E = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists)
    cat_reg = {'p_val': stats_model_cat.pvalues,
               't_val': stats_model_cat.tvalues,
               'param': stats_model_cat.params,
               'y_hat_E': y_hat_E,
               'y': y,
               'bse': stats_model_cat.bse}
    # continuous regression analysis
    x_con = sm.add_constant(x_con)
    stats_model_con = sm.OLS(y,x_con).fit()
    y_hat_E = stats_model_con.params[0] + (stats_model_con.params[1]*grid_dists)
    con_reg = {'p_val': stats_model_con.pvalues,
               't_val': stats_model_con.tvalues,
               'param': stats_model_con.params,
               'y_hat_E': y_hat_E,
               'y': y,
               'bse': stats_model_con.bse}

    reg_results = {'cat_reg': cat_reg, 
                   'con_reg': con_reg}
    return reg_results

def analyze_regression_exc(dist_results, test_data):
    # Useful dictionaries from test dataset
    n_states = test_data.n_states 
    hidd_dists = dist_results['hidd_dists'] #[n_combinations]: [120]
    grid_dists = dist_results['grid_dists']
    binary_phi = dist_results['angle_results']['binary_phi'] # [120]
    samples = dist_results['samples'] # [120, 2]
    states=[]
    p_vals, t_vals, params, y_hat_Es, ys, bses = ([] for i in range(6))
    
    for state in range(n_states):
        s_idxs = [i for i, sample in enumerate(samples) if state not in sample] # [105]
        # prepare data for the regression analysis
        x_cat = np.concatenate((grid_dists[s_idxs].reshape((-1,1)), binary_phi[s_idxs].reshape((-1,1))),axis=1)
        y = hidd_dists[s_idxs]
        # regression analysis
        x_cat = sm.add_constant(x_cat)
        stats_model_cat = sm.OLS(y,x_cat).fit() 
        y_hat_E = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists)
        states.append(state)
        p_vals.append(stats_model_cat.pvalues)
        t_vals.append(stats_model_cat.tvalues)
        params.append(stats_model_cat.params)
        y_hat_Es.append(y_hat_E)
        bses.append(stats_model_cat.bse)
    # regression analysis - after removing (0,0) and (3,3)
    s_idxs = [i for i, sample in enumerate(samples) if ((0 not in sample) & (15 not in sample))] # [91]
    x_cat = np.concatenate((grid_dists[s_idxs].reshape((-1,1)), binary_phi[s_idxs].reshape((-1,1))),axis=1)
    y = hidd_dists[s_idxs]
    x_cat = sm.add_constant(x_cat)
    stats_model_cat = sm.OLS(y,x_cat).fit() 
    y_hat_E = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists)
    states.append(16)
    p_vals.append(stats_model_cat.pvalues)
    t_vals.append(stats_model_cat.tvalues)
    params.append(stats_model_cat.params)
    y_hat_Es.append(y_hat_E)
    bses.append(stats_model_cat.bse)
    # regression analysis - after removing (0,0) and (3,3), (3,0) and (0.3)
    s_idxs = [i for i, sample in enumerate(samples) if ((0 not in sample) & (15 not in sample) &
                                                        (3 not in sample) & (12 not in sample))] #[66]
    x_cat = np.concatenate((grid_dists[s_idxs].reshape((-1,1)), binary_phi[s_idxs].reshape((-1,1))),axis=1)
    y = hidd_dists[s_idxs]
    x_cat = sm.add_constant(x_cat)
    stats_model_cat = sm.OLS(y,x_cat).fit() 
    y_hat_E = stats_model_cat.params[0] + (stats_model_cat.params[1]*grid_dists)
    states.append(17)
    p_vals.append(stats_model_cat.pvalues)
    t_vals.append(stats_model_cat.tvalues)
    params.append(stats_model_cat.params)
    y_hat_Es.append(y_hat_E)
    bses.append(stats_model_cat.bse)

    states = np.array(states)
    p_vals = np.array(p_vals)
    t_vals = np.array(t_vals)
    params = np.array(params)
    y_hat_Es = np.array(y_hat_Es)
    bses = np.array(bses)
    
    exc_reg_results = {'excluded_states': states,
                       'p_vals': p_vals,
                       't_vals': t_vals,
                       'params': params,
                       'y_hat_Es': y_hat_Es,
                       'ys': ys,
                       'bses': bses}                   

    return exc_reg_results

# do the analysis over multiple runs
def analyze_cortical_mruns(cortical_results, test_data, args):
    n_states = test_data.n_states 
    loc2idx = test_data.loc2idx 
    idx2loc = {idx:loc for loc, idx in loc2idx.items()}
    idxs = [idx for idx in range(n_states)]
    locs = [idx2loc[idx] for idx in idxs]
    analysis_type = args.analysis_type

    checkpoints = len(cortical_results[0])
    r_hidds, p_val_hidds, r_embeds, p_val_embeds = ([] for i in range(4))
    ratio_hidds, ratio_embeds = ([] for i in range(2))
    t_stat_embeds, t_p_val_embeds, t_stat_hidds, t_p_val_hidds = \
        ([] for i in range(4))
    p_val_cat_regs, t_val_cat_regs, param_cat_regs, bse_cat_regs, \
        y_cat_regs, y_hat_E_cat_regs = ([] for i in range(6))
    p_val_con_regs, t_val_con_regs, param_con_regs, bse_con_regs, \
        y_con_regs, y_hat_E_con_regs = ([] for i in range(6))
    p_val_exc_regs, t_val_exc_regs, param_exc_regs, y_exc_regs, \
            y_hat_E_exc_regs, bse_exc_regs = ([] for i in range(6))
    p_val_1D_regs, t_val_1D_regs, param_1D_regs, y_1D_regs, \
            y_hat_E_1D_regs, bse_1D_regs  = ([] for i in range(6))
    cong_embed_dists, incong_embed_dists, cong_hidd_dists, \
        incong_hidd_dists = ([] for i in range(4))
    embed_dists, hidd_dists, grid_dists,  grid_angles \
        = ([] for i in range(4))
    
    corr_results, reg_results, ttest_results, ratio_results, = ({} for i in range(4))
    pca_results, tsne_results, mds_results = ({} for i in range(3))

    if ((analysis_type=='pca') | (analysis_type=='all')):
        res = cortical_results[-1][-1]
        samples_res = res['samples_res']
        pca_results = analyze_dim_red(res, 'pca', n_components=3)
        # mds_results = analyze_dim_red(res, 'mds', n_components=2)
        # tsne_results = analyze_dim_red(res, 'tsne', n_components=2)
        
        pca_results['grid_locations']  = locs
        pca_results['samples_res']  = samples_res
        # mds_results['grid_locations']  = locs
        # mds_results['samples_res']  = samples_res
        # tsne_results['grid_locations'] = locs
        # tsne_results['samples_res']  = samples_res

    for run in range(args.nruns_cortical):
        r_hidd, p_val_hidd, r_embed, p_val_embed = ([] for i in range(4))
        ratio_hidd, ratio_embed = ([] for i in range(2))
        t_stat_embed, t_p_val_embed, t_stat_hidd, t_p_val_hidd \
            = ([] for i in range(4))
        p_val_cat_reg, t_val_cat_reg, param_cat_reg, bse_cat_reg, \
            y_cat_reg, y_hat_E_cat_reg = ([] for i in range(6))
        p_val_con_reg, t_val_con_reg, param_con_reg, bse_con_reg, \
            y_con_reg, y_hat_E_con_reg  = ([] for i in range(6))
        p_val_exc_reg, t_val_exc_reg, param_exc_reg, y_exc_reg, \
            y_hat_E_exc_reg, bse_exc_reg  = ([] for i in range(6))
        p_val_1D_reg, t_val_1D_reg, param_1D_reg, y_1D_reg, \
            y_hat_E_1D_reg, bse_1D_reg  = ([] for i in range(6))
        cong_embed_dist, incong_embed_dist, cong_hidd_dist, \
             incong_hidd_dist = ([] for i in range(4))
        embed_dist, hidd_dist, grid_dist, grid_angle \
            = ([] for i in range(4))

        for cp in range(checkpoints):
            # load the results
            cortical_res = cortical_results[run][cp]
            dist_results = calc_dist(cortical_res, test_data)
            dist_ctx_results = calc_dist_ctx(cortical_res, test_data)
            dist_c_inc_results = hist_data(dist_results) 
            # distance results
            embed_dist.append(dist_results['embed_dists'])
            hidd_dist.append(dist_results['hidd_dists'])
            grid_dist.append(dist_results['grid_dists'])
            grid_angle.append(dist_results['angle_results']['grid_angles'])
            cong_embed_dist.append(dist_c_inc_results['cong_embed_dist'])
            incong_embed_dist.append(dist_c_inc_results['incong_embed_dist'])
            cong_hidd_dist.append(dist_c_inc_results['cong_hidd_dist'])
            incong_hidd_dist.append(dist_c_inc_results['incong_hidd_dist'])

            # ratio analysis 
            if ((analysis_type=='ratio') | (analysis_type=='all')):
                ratio_results = calc_ratio(dist_results)
                # embeddings
                ratio_embed.append(ratio_results['ratio_embed'])
                # hiddens
                ratio_hidd.append(ratio_results['ratio_hidd'])

            # t-test analysis
            if ((analysis_type=='ttest') | (analysis_type=='all')):
                ttest_results = analyze_ttest(dist_results)                
                t_stat_embed.append(ttest_results['t_stat_embed'])
                t_p_val_embed.append(ttest_results['t_p_val_embed'])
                t_stat_hidd.append(ttest_results['t_stat_hidd'])
                t_p_val_hidd.append(ttest_results['t_p_val_hidd'])

            # corretion analysis
            if ((analysis_type=='corr') | (analysis_type=='all')):
                corr_results = analyze_corr(dist_results)
                # embeddings
                r_embed.append(corr_results['r_embed'])
                p_val_embed.append(corr_results['p_val_embed'])
                # hiddens
                r_hidd.append(corr_results['r_hidd'])
                p_val_hidd.append(corr_results['p_val_hidd'])

            # regression analysis
            if ((analysis_type=='regs') | (analysis_type=='all')):
                reg_results = analyze_regression(dist_results)
                # categorical
                cat_reg = reg_results['cat_reg']
                p_val_cat_reg.append(cat_reg['p_val'])
                t_val_cat_reg.append(cat_reg['t_val'])
                param_cat_reg.append(cat_reg['param'])
                y_cat_reg.append(cat_reg['y'])
                y_hat_E_cat_reg.append(cat_reg['y_hat_E'])
                bse_cat_reg.append(cat_reg['bse'])
                # conitnuous
                con_reg = reg_results['con_reg']
                p_val_con_reg.append(con_reg['p_val'])
                t_val_con_reg.append(con_reg['t_val'])
                param_con_reg.append(con_reg['param'])
                y_con_reg.append(con_reg['y'])
                y_hat_E_con_reg.append(con_reg['y_hat_E'])
                bse_con_reg.append(con_reg['bse'])
                # Excluded - Regression (do reg after removing each state)
                exc_reg = analyze_regression_exc(dist_results, test_data)
                p_val_exc_reg.append(exc_reg['p_vals'])
                t_val_exc_reg.append(exc_reg['t_vals'])
                param_exc_reg.append(exc_reg['params'])
                y_exc_reg.append(exc_reg['ys'])
                y_hat_E_exc_reg.append(exc_reg['y_hat_Es'])
                bse_exc_reg.append(exc_reg['bses'])
                # Including 1D rank difference in the model
                oneD_reg = analyze_regression_1D(dist_ctx_results)
                oneD_reg = oneD_reg['cat_reg']
                p_val_1D_reg.append(oneD_reg['p_val'])
                t_val_1D_reg.append(oneD_reg['t_val'])
                param_1D_reg.append(oneD_reg['param'])
                y_1D_reg.append(oneD_reg['y'])
                y_hat_E_1D_reg.append(oneD_reg['y_hat_E'])
                bse_1D_reg.append(oneD_reg['bse'])


        if ((analysis_type=='corr') | (analysis_type=='all')):
            r_hidds.append(r_hidd)
            p_val_hidds.append(p_val_hidd)
            r_embeds.append(r_embed)
            p_val_embeds.append(p_val_embed)
        if ((analysis_type=='ratio') | (analysis_type=='all')):
            ratio_hidds.append(ratio_hidd)
            ratio_embeds.append(ratio_embed)
        if ((analysis_type=='ttest') | (analysis_type=='all')):
            t_stat_embeds.append(t_stat_embed)
            t_p_val_embeds.append(t_p_val_embed)
            t_stat_hidds.append(t_stat_hidd)
            t_p_val_hidds.append(t_p_val_hidd)
        if ((analysis_type=='regs') | (analysis_type=='all')):
            p_val_cat_regs.append(p_val_cat_reg)
            t_val_cat_regs.append(t_val_cat_reg)
            param_cat_regs.append(param_cat_reg)
            y_cat_regs.append(y_cat_reg)
            y_hat_E_cat_regs.append(y_hat_E_cat_reg)
            bse_cat_regs.append(bse_cat_reg)
            p_val_con_regs.append(p_val_con_reg)
            t_val_con_regs.append(t_val_con_reg)
            param_con_regs.append(param_con_reg)
            y_con_regs.append(y_con_reg)
            y_hat_E_con_regs.append(y_hat_E_con_reg)
            bse_con_regs.append(bse_con_reg)
            p_val_exc_regs.append(p_val_exc_reg)
            t_val_exc_regs.append(t_val_exc_reg)
            param_exc_regs.append(param_exc_reg)
            y_exc_regs.append(y_exc_reg)
            y_hat_E_exc_regs.append(y_hat_E_exc_reg)
            bse_exc_regs.append(bse_exc_reg)
            p_val_1D_regs.append(p_val_1D_reg)
            t_val_1D_regs.append(t_val_1D_reg)
            param_1D_regs.append(param_1D_reg)
            y_1D_regs.append(y_1D_reg)
            y_hat_E_1D_regs.append(y_hat_E_1D_reg)
            bse_1D_regs.append(bse_1D_reg)    
        cong_embed_dists.append(cong_embed_dist)
        incong_embed_dists.append(incong_embed_dist)
        cong_hidd_dists.append(cong_hidd_dist)
        incong_hidd_dists.append(incong_hidd_dist)
        embed_dists.append(embed_dist)
        hidd_dists.append(hidd_dist)
        grid_dists.append(grid_dist)
        grid_angles.append(grid_angle)

    cong_embed_dists = np.array(cong_embed_dists)
    incong_embed_dists = np.array(incong_embed_dists)
    cong_hidd_dists = np.array(cong_hidd_dists)
    incong_hidd_dists = np.array(incong_hidd_dists)
    embed_dists = np.array(embed_dists)
    hidd_dists = np.array(hidd_dists)
    grid_dists = np.array(grid_dists)
    grid_angles = np.array(grid_angles)
    dist_results = {'embed_dists': embed_dists, 'hidd_dists': hidd_dists,
                    'grid_dists': grid_dists, 'grid_angles': grid_angles,
                    'cong_embed_dists': cong_embed_dists, 'incong_embed_dists': incong_embed_dists,
                    'cong_hidd_dists': cong_hidd_dists, 'incong_hidd_dists': incong_hidd_dists}

    if ((analysis_type=='corr') | (analysis_type=='all')):
        r_hidds = np.array(r_hidds)
        p_val_hidds = np.array(p_val_hidds)
        r_embeds = np.array(r_embeds)
        p_val_embeds = np.array(p_val_embeds)
        corr_results = {'r_embeds': r_embeds, 
                        'p_val_embeds': p_val_embeds,
                        'r_hidds': r_hidds, 
                        'p_val_hidds': p_val_hidds}
        results = {'corr_results': corr_results, 
                   'dist_results': dist_results}
        with open('../results/'+'corr_'+args.out_file, 'wb') as f:
            pickle.dump(results, f)
    if ((analysis_type=='ratio') | (analysis_type=='all')):
        ratio_hidds = np.array(ratio_hidds)
        ratio_embeds = np.array(ratio_embeds)
        ratio_results = {'ratio_embeds': ratio_embeds, 
                         'ratio_hidds': ratio_hidds}
        results = {'ratio_results': ratio_results, 
                   'dist_results': dist_results}
        with open('../results/'+'ratio_'+args.out_file, 'wb') as f:
            pickle.dump(results, f)
    if ((analysis_type=='ttest') | (analysis_type=='all')):
        t_stat_embeds = np.array(t_stat_embeds)
        t_p_val_embeds = np.array(t_p_val_embeds)
        t_stat_hidds = np.array(t_stat_hidds)
        t_p_val_hidds = np.array(t_p_val_hidds)
        ttest_results = {'t_stat_hidds': t_stat_hidds,
                         't_p_val_hidds': t_p_val_hidds,
                         't_stat_embeds': t_stat_embeds, 
                         't_p_val_embeds': t_p_val_embeds}
        results = {'ttest_results': ttest_results,
                   'dist_results': dist_results}
        with open('../results/'+'ttest_'+args.out_file, 'wb') as f:
            pickle.dump(results, f)
    if ((analysis_type=='pca') | (analysis_type=='all')): 
        results = {'pca_results': pca_results,
                   'dist_results': dist_results}
        with open('../results/'+'pca_'+args.out_file, 'wb') as f:
            pickle.dump(results, f)

    if ((analysis_type=='regs') | (analysis_type=='all')):
        p_val_cat_regs = np.array(p_val_cat_regs)
        t_val_cat_regs = np.array(t_val_cat_regs)
        param_cat_regs = np.array(param_cat_regs)
        y_cat_regs = np.array(y_cat_regs)
        y_hat_E_cat_regs = np.array(y_hat_E_cat_regs)
        bse_cat_regs = np.array(bse_cat_regs)
        p_val_con_regs = np.array(p_val_con_regs)
        t_val_con_regs = np.array(t_val_con_regs)
        param_con_regs = np.array(param_con_regs)
        y_con_regs = np.array(y_con_regs)
        y_hat_E_con_regs = np.array(y_hat_E_con_regs)
        bse_con_regs = np.array(bse_con_regs)
        p_val_exc_regs = np.array(p_val_exc_regs)
        t_val_exc_regs = np.array(t_val_exc_regs)
        param_exc_regs = np.array(param_exc_regs)
        y_exc_regs = np.array(y_exc_regs)
        y_hat_E_exc_regs = np.array(y_hat_E_exc_regs)
        bse_exc_regs = np.array(bse_exc_regs)
        p_val_1D_regs = np.array(p_val_1D_regs)
        t_val_1D_regs = np.array(t_val_1D_regs)
        param_1D_regs = np.array(param_1D_regs)
        y_1D_regs = np.array(y_1D_regs)
        y_hat_E_1D_regs = np.array(y_hat_E_1D_regs)
        bse_1D_regs = np.array(bse_1D_reg)
        cat_regs = {'p_vals': p_val_cat_regs,
                    't_vals': t_val_cat_regs,
                    'params': param_cat_regs,
                    'ys': y_cat_regs,
                    'y_hat_Es': y_hat_E_cat_regs,
                    'bses': bse_cat_regs}
        con_regs = {'p_vals': p_val_con_regs,
                    't_vals': t_val_con_regs,
                    'params': param_con_regs,
                    'ys': y_con_regs,
                    'y_hat_Es': y_hat_E_con_regs,
                    'bses': bse_con_regs}
        exc_regs = {'p_vals': p_val_exc_regs,
                    't_vals': t_val_exc_regs,
                    'params': param_exc_regs,
                    'ys': y_exc_regs,
                    'y_hat_Es': y_hat_E_exc_regs,
                    'bses': bse_exc_regs}
        oneD_regs = {'p_vals': p_val_1D_regs,
                    't_vals': t_val_1D_regs,
                    'params': param_1D_regs,
                    'ys': y_1D_regs,
                    'y_hat_Es': y_hat_E_1D_regs,
                    'bses': bse_1D_regs}
        reg_results = {'cat_regs': cat_regs,
                       'con_regs': con_regs,
                       'exc_regs': exc_regs,
                       'oneD_regs': oneD_regs}
        results = {'dist_results': dist_results,
                   'reg_results': reg_results}
        with open('../results/'+'reg_'+args.out_file, 'wb') as f:
            pickle.dump(results, f)
    
    

    return results