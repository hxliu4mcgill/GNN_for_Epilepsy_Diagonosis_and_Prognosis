import torch
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat


class ModuleTimestamping(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, t, sampling_endpoints):
        return self.rnn(t[:sampling_endpoints[-1]])[0][[p-1 for p in sampling_endpoints]] # t * b * c


class LayerGCN(nn.Module):
    """
    A: (b t) n c
    V: (b t) n c
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.threshold = 0.1
        self.feat_drop = nn.Dropout(0.2)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())


    def forward(self, a, v):

        a_th = a >= self.threshold
        I = torch.eye(a_th.shape[1]).unsqueeze(0).expand(a_th.shape[0], -1, -1).to(a_th.device)
        a_th = a_th.float() # + I
        d = torch.sum(a_th, dim=-1, keepdim=False)
        d_ = torch.diag_embed(torch.pow(d, -0.5))
        a_bar = torch.bmm(d_, a_th)
        a_bar = torch.bmm(a_bar, d_)

        h = torch.bmm(a_bar, v)

        feat_bridge = rearrange(h, 'b n c -> (b n) c')
        out = self.mlp(feat_bridge)
        out = self.feat_drop(out)
        return out


class ModuleMeanReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1,1,1], dtype=torch.float32)

class ModuleSumReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.sum(node_axis), torch.zeros(size=[1,1,1], dtype=torch.float32)

class ModuleSERO(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale*hidden_dim)), nn.BatchNorm1d(round(upscale*hidden_dim)), nn.GELU())
        self.attend = nn.Linear(round(upscale*hidden_dim), input_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_readout = x.mean(node_axis)
        x_shape = x_readout.shape
        x_embed = self.embed(x_readout.reshape(-1,x_shape[-1]))
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1],-1)
        permute_idx = list(range(node_axis))+[len(x_graphattention.shape)-1]+list(range(node_axis,len(x_graphattention.shape)-1))
        x_graphattention = x_graphattention.permute(permute_idx)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)

class ModuleGARO(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n'))/np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)



class ModuleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))


    def forward(self, x):
        x_attend, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x_attend) # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix


class ModelSTAGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads, num_layers, sparsity, dropout=0.5, cls_token='sum', readout='sero', hop_num=4, garo_upscale=1.0):
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token=='sum': self.cls_token = lambda x: x.sum(0)
        elif cls_token=='mean': self.cls_token = lambda x: x.mean(0)
        elif cls_token=='param': self.cls_token = lambda x: x[-1]
        else: raise
        if readout=='garo': readout_module = ModuleGARO
        elif readout=='sero': readout_module = ModuleSERO
        elif readout=='mean': readout_module = ModuleMeanReadout
        elif readout =='sum': readout_module = ModuleSumReadout
        else: raise

        self.token_parameter = nn.Parameter(torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token=='param' else None

        self.num_classes = num_classes
        self.sparsity = sparsity

        # define modules
        self.percentile = Percentile()
        self.timestamp_encoder = ModuleTimestamping(input_dim, hidden_dim, hidden_dim)
        # self.initial_linear = nn.Sequential(nn.Linear(input_dim+hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim))
        self.initial_linear = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim))
        self.gcn = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            self.gcn.append(LayerGCN(hidden_dim, hidden_dim, hidden_dim))
            self.readout_modules.append(readout_module(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1))
            self.transformer_modules.append(ModuleTransformer(hidden_dim, 2*hidden_dim, num_heads=num_heads, dropout=0.1))
            self.linear_layers.append(nn.Linear(hidden_dim, num_classes))

        self.head = nn.Linear(hidden_dim * num_layers, int(hidden_dim / 2))
        self.classifier=nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * num_layers, num_classes))


    def _collate_adjacency(self, a, sparsity, sparse=False):
        i_list = []
        v_list = []

        if sparse is True:
            for sample, _dyn_a in enumerate(a):
                for timepoint, _a in enumerate(_dyn_a):
                    thresholded_a = (_a > self.percentile(_a, 100-sparsity))
                    _i = thresholded_a.nonzero(as_tuple=False)
                    _v = torch.ones(len(_i))
                    _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                    i_list.append(_i)
                    v_list.append(_v)
            _i = torch.cat(i_list).T.to(a.device)
            _v = torch.cat(v_list).to(a.device)

            return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3]))

        else:
            a_ = a.clone()
            for sample, _dyn_a in enumerate(a):
                for timepoint, _a in enumerate(_dyn_a):
                    thresholded_a = (_a > self.percentile(_a, 100 - sparsity))
                    a_[sample][timepoint] = thresholded_a.float()

            return a_


    def forward(self, v, a, t, sampling_endpoints):
        # assumes shape [minibatch x time x node x feature] for v
        # assumes shape [minibatch x time x node x node] for a
        logit = 0.0
        attention = {'edge-attention': [], 'time-attention': []}
        latent_list = []
        minibatch_size, num_timepoints, num_nodes = a.shape[:3]

        time_encoding = self.timestamp_encoder(t, sampling_endpoints)
        time_encoding = repeat(time_encoding, 'b t c -> t b n c', n=num_nodes)

        h = torch.cat([v, time_encoding], dim=3)
        h = rearrange(h, 'b t n c -> (b t n) c')
        h = self.initial_linear(h)
        dyn_v = h.clone()
        # a = self._collate_adjacency(a, self.sparsity)

        # h = time_encoding
        # h = rearrange(h, 'b t n c -> (b t n) c')
        # h = self.initial_linear(h)
        # a = self._collate_adjacency(a, self.sparsity)

        for layer, (G, R, T, L) in enumerate(zip(self.gcn, self.readout_modules, self.transformer_modules, self.linear_layers)):
            h_bridge = rearrange(h, '(b t n) c -> (b t) n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
            a_bridge = rearrange(a, 'b t n c -> (b t) n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
            h = G(a_bridge, h_bridge)

            h2_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
            h_readout, node_attn = R(h2_bridge, node_axis=2)

            if self.token_parameter is not None: h_readout = torch.cat([h_readout, self.token_parameter[layer].expand(-1,h_readout.shape[1],-1)])

            h_attend, time_attn = T(h_readout)
            ortho_latent = rearrange(h2_bridge, 't b n c -> (t b) n c')
            # matrix_inner = torch.bmm(ortho_latent, ortho_latent.permute(0,2,1))
            # reg_ortho += (matrix_inner/matrix_inner.max(-1)[0].unsqueeze(-1) - torch.eye(num_nodes, device=matrix_inner.device)).triu().norm(dim=(1,2)).mean()

            latent = self.cls_token(h_attend)
            # logit += self.dropout(L(latent))
            logit += L(latent)

            # attention['node-attention'].append(node_attn)
            # edge_attn = rearrange(edge_attn, '(b t) n c -> b t n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
            attention['edge-attention'].append(time_attn)
            attention['time-attention'].append(time_attn)
            latent_list.append(latent)

        logit = logit.squeeze(1)
        # attention['node-attention'] = torch.stack(attention['node-attention'], dim=1).detach().cpu()
        attention['edge-attention'] = torch.stack(attention['edge-attention'], dim=1).detach().cpu()
        attention['time-attention'] = torch.stack(attention['time-attention'], dim=1).detach().cpu()
        latent = torch.stack(latent_list, dim=1)
        latent = rearrange(latent, 'b l c -> b (l c)')
        logit2 = self.classifier(latent)
        latent = nn.functional.normalize(self.head(latent), dim=1)

        return logit2, attention, latent, 0, dyn_v #reg_ortho
        # return logit, attention, latent, 0  # reg_ortho



# Percentile class based on
# https://github.com/aliutkus/torchpercentile
class Percentile(torch.autograd.Function):
    def __init__(self):
        super().__init__()


    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)


    def forward(self, input, percentiles):
        input = torch.flatten(input) # find percentiles for flattened axis
        input_dtype = input.dtype
        input_shape = input.shape
        if isinstance(percentiles, int):
            percentiles = (percentiles,)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles, dtype=torch.double)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles)
        input = input.double()
        percentiles = percentiles.to(input.device).double()
        input = input.view(input.shape[0], -1)
        in_sorted, in_argsort = torch.sort(input, dim=0)
        positions = percentiles * (input.shape[0]-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        result = (d0+d1).view(-1, *input_shape[1:])
        return result.type(input_dtype)


    def backward(self, grad_output):
        """
        backward the gradient is basically a lookup table, but with weights
        depending on the distance between each point and the closest
        percentiles
        """
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        # the argsort in the flattened in vector

        cols_offsets = (
            torch.arange(
                    0, input_shape[1], device=in_argsort.device)
            )[None, :].long()
        in_argsort = (in_argsort*input_shape[1] + cols_offsets).view(-1).long()
        floored = (
            floored[:, None]*input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
            ceiled[:, None]*input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)