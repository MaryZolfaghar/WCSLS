import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTMCell
# from torch.nn.modules.rnn import LSTMCell

class MemoryLayer(nn.Module):
    def __init__(self, input_dim, memory_dim, model_dim, mlp_dim, dropout_p):
        super(MemoryLayer, self).__init__()
        
        # Hyperparameters
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.model_dim = model_dim
        self.mlp_dim = mlp_dim
        
        # Parameters
        self.W_q = nn.Linear(input_dim, model_dim)
        self.W_k = nn.Linear(memory_dim, model_dim)
        self.W_v = nn.Linear(memory_dim, model_dim)
        self.lin1 = nn.Linear(model_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        
    def forward(self, x, m):
        # Generate queries, keys, values
        Q = self.W_q(x) # [batch, n_test, model_dim]
        K = self.W_k(m) # [batch, n_memories, model_dim]
        V = self.W_v(m) # [batch, n_memories, model_dim]
        
        # Get attention distributions over memories for each sample
        attn = torch.matmul(Q, K.permute(0,2,1)) # [batch, n_test, n_memories]
        attn = attn/np.sqrt(self.model_dim)
        attn = self.softmax(attn) # [batch, n_test, n_memories]
        
        # Get weighted average of values
        V_bar = torch.matmul(attn, V) # [batch, n_test, model_dim]
        
        # Feedforward
        out = self.lin1(V_bar) # [batch, n_test, mlp_dim]
        out = self.dropout(out)
        out = self.relu(out)   # [batch, n_test, mlp_dim]
        out = self.lin2(out)   # [batch, n_test, input_dim]
        
        return out, attn
        
class EpisodicSystem(nn.Module):
    def __init__(self):
        super(EpisodicSystem, self).__init__()
    
        # Hyperparameters
        self.n_states = 16    # number of faces in 4x4 grid
        self.ctx_dim = 2     # dimension of context/axis (2d one-hot vectors)
        self.y_dim = 1        # dimension of y (binary)
        self.model_dim = 32   # dimension of Q, K, V
        self.mlp_dim = 64     # dimension of mlp hidden layer
        self.n_layers = 1     # number of layers
        self.dropout_p = 0.0  # dropout probability
        self.input_dim = 2*self.n_states + self.ctx_dim # y not given in input
        self.memory_dim = self.input_dim + self.y_dim    # y given in memories
        self.output_dim = 2   # number of choices (binary)
        
        # Memory system
        memory_layers = []
        for l_i in range(self.n_layers):
            layer = MemoryLayer(self.input_dim, self.memory_dim, self.model_dim, 
                                self.mlp_dim, self.dropout_p)
            memory_layers.append(layer)
        self.memory_layers = nn.ModuleList(memory_layers)
        
        # Output
        self.lin1 = nn.Linear(2*self.input_dim, self.mlp_dim)
        self.lin2 = nn.Linear(self.mlp_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_p)
        
    def forward(self, x, m):
        out = x
        # Memory system
        attention = []
        for l_i in range(self.n_layers):
            out, attn = self.memory_layers[l_i](out, m) 
            # out = [batch, n_test, model_dim]
            # attn = [batch, n_test, n_memories]
            attention.append(attn.detach().cpu().numpy())
        
        # MLP
        out = torch.cat([x, out], dim=2) # [batch, n_test, 2*input_dim]
        out = self.lin1(out) # [batch, n_test, mlp_dim]
        out = self.dropout(out) # [batch, n_test, mlp_dim]
        out = self.relu(out) # [batch, n_test, mlp_dim]
        out = self.lin2(out) # [batch, n_test, out_dim]
        
        return out, attention

class CNN(nn.Module):
    def __init__(self, state_dim):
        super(CNN, self).__init__()

        # Hyperparameters
        self.state_dim = state_dim  # size of final embeddings
        self.image_size = 64        # height and width of images
        self.in_channels = 1        # channels in inputs (grey-scaled)
        self.kernel_size = 3        # kernel size of convolutions
        self.padding = 0            # padding in conv layers
        self.stride = 2             # stride of conv layers
        self.pool_kernel = 2        # kernel size of max pooling
        self.pool_stride = 2        # stride of max pooling
        self.out_channels1 = 4      # number of channels in conv1
        self.out_channels2 = 8      # number of channels in conv2
        self.num_layers = 2         # number of conv layers

        # Conv layers
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels1, 
                               self.kernel_size, self.stride, self.padding)
        self.maxpool1 = nn.MaxPool2d(self.pool_kernel, self.pool_stride)

        self.conv2 = nn.Conv2d(self.out_channels1, self.out_channels2, 
                               self.kernel_size, self.stride, self.padding)
        self.maxpool2 = nn.MaxPool2d(self.pool_kernel, self.pool_stride)

        # Linear layer
        self.cnn_out_dim = self.calc_cnn_out_dim()
        self.linear = nn.Linear(self.cnn_out_dim, self.state_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv 1
        x = self.conv1(x)          # [batch, 4, 31, 31]
        x = self.relu(x)           # [batch, 4, 31, 31]
        x = self.maxpool1(x)       # [batch, 4, 15, 15]

        # Conv 2
        x = self.conv2(x)          # [batch, 8, 7, 7]
        x = self.relu(x)           # [batch, 8, 7, 7]
        x = self.maxpool2(x)       # [batch, 8, 3, 3]

        # Linear
        x = x.view(x.shape[0], -1) # [batch, 72]
        x = self.linear(x)         # [batch, 32]
        
        return x
        
    def calc_cnn_out_dim(self):
        w = self.image_size
        h = self.image_size 
        for l in range(self.num_layers):
            new_w = np.floor(((w - self.kernel_size)/self.stride) + 1)
            new_h = np.floor(((h - self.kernel_size)/self.stride) + 1)
            new_w = np.floor(new_w / self.pool_kernel)
            new_h = np.floor(new_h / self.pool_kernel)
            w = new_w
            h = new_h
        return int(w*h*8)

class CorticalSystem(nn.Module):
    def __init__(self, args):
        super(CorticalSystem, self).__init__()
        self.use_images = args.use_images
        self.N_contexts = args.N_contexts
        self.N_responses = args.N_responses
        self.is_lesion = args.is_lesion
        self.lesion_p = args.lesion_p
        self.measure_grad_norm = args.measure_grad_norm
        
        # Hyperparameters
        self.n_states = 16
        self.state_dim = 32
        if self.N_responses=='one':
            self.mlp_in_dim = 3*self.state_dim # (f1 + f2 + context/axis)
        elif self.N_responses=='two':
            self.mlp_in_dim = 2*self.state_dim # (f1 + f2)
        self.hidden_dim = 128
        self.output_dim = 2
        self.analyze = False
        
        # Input embedding (images or one-hot)
        if self.use_images:
            self.face_embedding = CNN(self.state_dim)
        else:
            self.face_embedding = nn.Embedding(self.n_states, self.state_dim)
            nn.init.xavier_normal_(self.face_embedding.weight)
            
        self.ctx_embedding = nn.Embedding(self.N_contexts, self.state_dim)
        nn.init.xavier_normal_(self.ctx_embedding.weight)

        # MLP
        self.hidden = nn.Linear(self.mlp_in_dim, self.hidden_dim)
        self.resp1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.resp2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, f1, f2, ctx):

        # Embed inputs
        f1_embed  = self.face_embedding(f1) # [batch, state_dim]
        f2_embed  = self.face_embedding(f2) # [batch, state_dim]
        ctx_embed = self.ctx_embedding(ctx) # [batch, state_dim]
        
        if self.is_lesion:
            ctx_embed = torch.tensor(self.lesion_p) * ctx_embed
        
        if self.measure_grad_norm:
            self.f1_embed = f1_embed
            self.f2_embed = f2_embed
            self.ctx_embed = ctx_embed

        x = torch.cat([f1_embed, f2_embed, ctx_embed], dim=1) 
        # if self.N_responses == 'one':
        #     ctx_embed = self.ctx_embedding(ctx) # [batch, state_dim]
        #     # MLP
        #     x = torch.cat([f1_embed, f2_embed, ctx_embed], dim=1) 
        #     # x: [batch, 3*state_dim]: [32, 96]
        # elif self.N_responses == 'two':
        #     # MLP
        #     x = torch.cat([f1_embed, f2_embed], dim=1) 
        #     # x: [batch, 2*state_dim]: [32, 64]
        
        hidd = self.hidden(x) # [batch, hidden_dim]
        hidd = self.relu(hidd)    # [batch, hidden_dim]
        x1 = self.resp1(hidd) # [batch, output_dim]
        if self.N_responses == 'one':
            x = x1
        elif self.N_responses == 'two':
            x2 = self.resp2(hidd) # [batch, output_dim]
            x = [x1, x2]
        
        return x, hidd

class RecurrentCorticalSystem(nn.Module):
    def __init__(self, args):
        super(RecurrentCorticalSystem, self).__init__()
        self.use_images = args.use_images
        self.N_contexts = args.N_contexts
        self.order_ctx = args.order_ctx
        self.N_responses = args.N_responses
        self.is_lesion = args.is_lesion
        self.lesion_p = args.lesion_p
        self.measure_grad_norm = args.measure_grad_norm

        # Hyperparameters
        self.n_states = 16
        self.state_dim = 32
        self.hidden_dim = 128
        self.output_dim = 2
        self.analyze = False
        
        # Input embedding (images or one-hot)
        if self.use_images:
            self.face_embedding = CNN(self.state_dim)
        else:
            self.face_embedding = nn.Embedding(self.n_states, self.state_dim)
            nn.init.xavier_normal_(self.face_embedding.weight)
            
        self.ctx_embedding = nn.Embedding(self.N_contexts, self.state_dim)
        nn.init.xavier_normal_(self.ctx_embedding.weight)
        # ctx_bias = torch.cat([1*torch.ones([1, self.state_dim]), -1*torch.ones([1, self.state_dim])], dim=0)
        # self.ctx_embedding.weight.data = self.ctx_embedding.weight.data + ctx_bias
        

        # LSTM
        self.lstm = nn.LSTM(self.state_dim, self.hidden_dim)

        # MLP
        self.resp1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.resp2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, f1, f2, ctx):

        # Embed inputs
        f1_embed = self.face_embedding(f1) # [batch, state_dim]
        f2_embed = self.face_embedding(f2) # [batch, state_dim]
        ctx_embed = self.ctx_embedding(ctx)# [batch, state_dim]

        if self.is_lesion:
            ctx_embed = torch.tensor(self.lesion_p) * ctx_embed

        if self.measure_grad_norm:
            self.f1_embed = f1_embed
            self.f2_embed = f2_embed
            self.ctx_embed = ctx_embed

        # LSTM
        if self.order_ctx == 'last':
            x = torch.cat([f1_embed.unsqueeze(0), f2_embed.unsqueeze(0),
                           ctx_embed.unsqueeze(0)], dim=0)
        elif self.order_ctx == 'first':
            x = torch.cat([ctx_embed.unsqueeze(0), f1_embed.unsqueeze(0), 
                           f2_embed.unsqueeze(0)], dim=0)
            
        
        # MLP
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [seq_length, batch, hidden_dim]
        # h: [1, batch, hidden_dim]
        # c: [1, batch, hidden_dim]
        if self.analyze:
            lstm_out = lstm_out.permute(1,0,2)
            # lstm_out: [batch, seq_length, hidden_dim]
            x1 = self.resp1(lstm_out)
            # x1: [batch, seq_length, output_dim] 
            if self.N_responses == 'one':
                x = x1
            elif self.N_responses == 'two':
                x2 = self.resp2(lstm_out)
                x = [x1, x2]
        else:
            x1 = self.resp1(h_n.squeeze(0))
            # x1: [batch, output_dim] 
            if self.N_responses == 'one':
                x = x1
            elif self.N_responses == 'two':
                x2 = self.resp2(h_n.squeeze(0))
                x = [x1, x2]
        
        return x, lstm_out

class RNNCell(nn.Module):
    def __init__(self, args):
        super(RNNCell, self).__init__()
        self.use_images = args.use_images
        self.N_contexts = args.N_contexts
        self.order_ctx = args.order_ctx
        self.N_responses = args.N_responses
        self.is_lesion = args.is_lesion
        self.lesion_p = args.lesion_p
        self.measure_grad_norm = args.measure_grad_norm

        # Hyperparameters
        self.n_states = 16
        self.state_dim = 32
        self.hidden_dim = 128
        self.output_dim = 2
        self.analyze = False
        
        # Input embedding (images or one-hot)
        if self.use_images:
            self.face_embedding = CNN(self.state_dim)
        else:
            self.face_embedding = nn.Embedding(self.n_states, self.state_dim)
            nn.init.xavier_normal_(self.face_embedding.weight)
        
        self.ctx_embedding = nn.Embedding(self.N_contexts, self.state_dim)
        nn.init.xavier_normal_(self.ctx_embedding.weight)
        ctx_bias = torch.cat([1*torch.ones([1, self.state_dim]), -1*torch.ones([1, self.state_dim])], dim=0)
        self.ctx_embedding.weight.data = self.ctx_embedding.weight.data + ctx_bias
        

        # LSTM Cell
        self.lstmcell = nn.LSTMCell(self.state_dim, self.hidden_dim)

        # MLP
        self.resp1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.resp2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()


    def forward(self, f1, f2, ctx):
        # Embed inputs
        f1_embed  = self.face_embedding(f1).unsqueeze(0) # [1, batch, state_dim]
        f2_embed  = self.face_embedding(f2).unsqueeze(0) # [1, batch, state_dim]
        ctx_embed = self.ctx_embedding(ctx).unsqueeze(0) # [1, batch, state_dim]
        
        if self.is_lesion:
            ctx_embed = torch.tensor(self.lesion_p) * ctx_embed

        if self.measure_grad_norm:
            self.f1_embed = f1_embed
            self.f2_embed = f2_embed
            self.ctx_embed = ctx_embed

        # LSTMCell
        if self.order_ctx == 'last':
            x = torch.cat([f1_embed, f2_embed, ctx_embed], dim=0)
        elif self.order_ctx == 'first':
            x = torch.cat([ctx_embed, f1_embed, f2_embed], dim=0)

        lstm_out = []
        n_times = len(x)
        # h0 and c0
        h_n = torch.zeros([f1_embed.size(1), self.hidden_dim]) # [1, batch, hidden_dim]
        c_n = torch.zeros([f1_embed.size(1), self.hidden_dim]) # [1, batch, hidden_dim]
        for t in range(n_times):
            xt = x[t]
            h_n, c_n = self.lstmcell(xt, (h_n.detach(), c_n.detach())) # h_n/c_n: [1,batch, hidden_dim]
            lstm_out.append(h_n)
        lstm_out = torch.stack(lstm_out, dim=0) # [seq_length, batch, hidden_dim]

        if self.analyze:
            lstm_out = lstm_out.permute(1,0,2)
            # lstm_out: [batch, seq_length, hidden_dim]
            x1 = self.resp1(lstm_out)
            # x1: [batch, seq_length, output_dim] 
            if self.N_responses == 'one':
                x = x1
            elif self.N_responses == 'two':
                x2 = self.resp2(lstm_out)
                x = [x1, x2]
        else:
            # r = torch.cat([h_n.unsqueeze(0), ctx_embed], dim=0)
            x1 = self.resp1(h_n)
            # x1: [batch, output_dim] 
            if self.N_responses == 'one':
                x = x1
            elif self.N_responses == 'two':
                x2 = self.resp2(h_n)
                x = [x1, x2]
        
        return x, lstm_out

class StepwiseCorticalSystem(nn.Module):
    def __init__(self, args):
        super(StepwiseCorticalSystem,self).__init__()
        self.use_images = args.use_images
        self.order_ctx = args.order_ctx
        self.truncated_mlp = args.truncated_mlp
        self.is_lesion = args.is_lesion
        self.lesion_p = args.lesion_p
        self.measure_grad_norm = args.measure_grad_norm

        # Hyperparameters
        self.n_states = 16
        self.state_dim = 32
        self.hidden1_dim = 128
        self.hidden2_dim = 128
        self.mlp_in1_dim = 2*self.state_dim
        self.mlp_in2_dim = self.hidden1_dim+self.state_dim
        self.output_dim = 2
        self.analyze = False
        
        # Input Embedding (images or one-hot)
        if self.use_images:
            self.face_embedding = CNN(self.state_dim)
        else:
            self.face_embedding = nn.Embedding(self.n_states, self.state_dim)
            nn.init.xavier_normal_(self.face_embedding.weight)
        self.ctx_embedding = nn.Embedding(2, self.state_dim)
        nn.init.xavier_normal_(self.ctx_embedding.weight)

        # MLP
        self.hidden1 = nn.Linear(self.mlp_in1_dim, self.hidden1_dim)
        self.hidden2 = nn.Linear(self.mlp_in2_dim, self.hidden2_dim)
        self.resp1 = nn.Linear(self.hidden2_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, f1, f2, ctx):
        f1_embed  = self.face_embedding(f1) # [batch, state_dim]
        f2_embed  = self.face_embedding(f2) # [batch, state_dim]
        ctx_embed = self.ctx_embedding(ctx)
        
        if self.is_lesion:
            ctx_embed = torch.tensor(self.lesion_p) * ctx_embed

        if self.measure_grad_norm:
            self.f1_embed = f1_embed
            self.f2_embed = f2_embed
            self.ctx_embed = ctx_embed

        # if self.order_ctx == 'last':
        #     x1 = torch.cat([ctx_embed, f1_embed], dim=1)
        #     # x = torch.cat([f1_embed, f2_embed, ctx_embed], dim=0)
        # elif self.order_ctx == 'first':
        #     x1 = torch.cat([ctx_embed, f1_embed], dim=1)
        #     # x = torch.cat([ctx_embed, f1_embed, f2_embed], dim=0)

        x1 = torch.cat([ctx_embed, f1_embed], dim=1)
        hidd1 = self.hidden1(x1) # [batch, hidden1_dim]
        hidd1 = self.relu(hidd1) # [batch, hidden1_dim]
        
        if self.truncated_mlp=='true':
            x2 = torch.cat([hidd1.detach(), f2_embed], dim=1) # [batch, state_dim+hidden1_dim]
        else:
            x2 = torch.cat([hidd1, f2_embed], dim=1) # [batch, state_dim+hidden1_dim]

        hidd2 = self.hidden2(x2) # [batch, hidden2_dim]
        hidd2 = self.relu(hidd2) # [batch, hidden2_dim]
        x = self.resp1(hidd2)  # [batch, output_dim]
        hidd = [hidd1, hidd2]
        return x, hidd

class CognitiveController(nn.Module):
    def __init__(self, args):
        super(CognitiveController, self).__init__()
        self.use_images = args.use_images
        self.n_rsp = args.N_responses
        self.n_ctx = args.N_contexts
        self.is_lesion = args.is_lesion
        self.lesion_p = args.lesion_p

        # Hyperparameters
        self.n_states = 16
        self.state_dim = 32
        self.mlp_in_dim = 2*self.state_dim # f1+f2 (context treated separately)
        self.hidden_dim = 128
        msg = "hidden_dim must be divisible by N_contexts"
        assert self.hidden_dim % self.N_contexts == 0, msg
        self.h_dim = self.hidden_dim // self.N_contexts # neurons per group in hidden
        self.output_dim = 2
        self.analyze = False
        

        # Input embedding (images or one-hot)
        if self.use_images:
            self.face_embedding = CNN(self.state_dim)
        else:
            self.face_embedding = nn.Embedding(self.n_states, self.state_dim)
            nn.init.xavier_normal_(self.face_embedding.weight)
            
        self.ctx_embedding = nn.Embedding(self.N_contexts, self.state_dim)
        nn.init.xavier_normal_(self.ctx_embedding.weight)

        # MLP
        self.control = nn.Linear(self.state_dim, self.N_contexts)
        self.linear = nn.Linear(self.mlp_in_dim, self.hidden_dim)
        self.out1 = nn.Linear(self.hidden_dim, self.output_dim)
        if self.N_responses == 'two':
            self.out2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, f1, f2, ctx):
        batch = f1.shape[0]

        # Embed inputs
        f1_embed  = self.face_embedding(f1) # [batch, state_dim]
        f2_embed  = self.face_embedding(f2) # [batch, state_dim]
        ctx_embed = self.ctx_embedding(ctx) # [batch, state_dim]
        
        
        if self.is_lesion:
            ctx_embed = torch.tensor(self.lesion_p) * ctx_embed

        if self.measure_grad_norm:
            self.f1_embed = f1_embed
            self.f2_embed = f2_embed
            self.ctx_embed = ctx_embed

        # Hidden
        x = torch.cat([f1_embed, f2_embed], dim=1) # [batch, 2*state_dim]
        hidden = self.relu(self.linear(x)) # [batch, hidden_dim]
        hidden = hidden.view(batch, self.h_dim, self.n_ctx) 
        # hidden: [batch, hidden_dim // n_ctx, n_ctx]

        # Control
        control_signal = self.softmax(self.control(ctx_embed)) # [batch, n_ctx]
        control_signal = control_signal.unsqueeze(1) # [batch, 1, n_ctx]
        hidden = hidden * control_signal # [batch, hidden_dim // n_ctx, n_ctx]
        
        # Output
        hidden = hidden.view(batch,-1) # [batch, hidden_dim]
        output = self.out1(hidden) # [batch, output_dim]
        if self.n_rsp == 'two':
            output2 = self.out2(hidden) # [batch, output_dim]
            output = [output, output2]
    
        return output, hidden
