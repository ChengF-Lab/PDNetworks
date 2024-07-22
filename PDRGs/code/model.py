import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, uniform_
import math



class PolyGraphConvolution(nn.Module):

    def __init__(self, n_nei, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_nei = n_nei
        self.weight = nn.Parameter(torch.empty(n_nei, in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, adj_norm):
        out = x @ self.weight[0, :, :]
        for idx in range(1, self.n_nei):
            curr_agg = x @ self.weight[idx,:,:]
            for _ in range(idx):
                curr_agg = adj_norm @ curr_agg
            out += curr_agg
        return out


class PolyGCN(nn.Module):

    def __init__(self, adj, input_dim, hidden_dims, output_dim, n_nei, dropout, agg, batch_norm=False):
        super().__init__()

        self.dropout = dropout
        self.adj = adj
        self.agg = agg
        self.act = nn.ReLU()
        self.layers = nn.ModuleList()
        self.create_layers(n_nei, input_dim, hidden_dims, output_dim)

        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None

    def create_layers(self, n_nei, input_dim, hidden_dims, output_dim):

        if self.agg == None:
            layer_dims = [input_dim, *hidden_dims, output_dim]
            for dim1, dim2 in zip(layer_dims[:-1], layer_dims[1:]):
                self.layers.append(PolyGraphConvolution(n_nei=n_nei, in_features=dim1,
                                                        out_features=dim2))

        elif self.agg == "concat":
            # layer_dims = np.concatenate((hidden_dims, [sum(hidden_dims)], [output_dim])).astype(np.int32)
            # self.layers = nn.ModuleList([PolyGraphConvolution(n_nei, input_dim, layer_dims[0])])
            # for idx in range(len(layer_dims) - 3):
            #     self.layers.append(PolyGraphConvolution(n_nei, layer_dims[idx], layer_dims[idx + 1]))
            # self.layers.append(PolyGraphConvolution(n_nei, layer_dims[-2], layer_dims[-1]))
            layer_dims = [input_dim, *hidden_dims, sum(hidden_dims), output_dim]
            for dim1, dim2 in zip(layer_dims[:-1], layer_dims[1:]):
                if dim2 != sum(hidden_dims):
                    self.layers.append(PolyGraphConvolution(n_nei=n_nei, in_features=dim1,
                                                            out_features=dim2))



    def gcn_forward(self, x, adj_norm):
        for idx, poly_gcn in enumerate(self.layers):
            if self.dropout != 0:
                x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
            x = poly_gcn(x, adj_norm)
            if idx != len(self.layers) - 1:
                x = self.act(x)
                if self.batch_norm is not None:
                    x = self.batch_norm[idx](x)
        return x


    def concat_forward(self, x, adj_norm):

        hidden_embed = []

        for idx in range(len(self.layers) - 1):
            x = self.layers[idx](x, adj_norm)
            x = self.act(x)
            if self.batch_norm is not None:
                x = self.batch_norm[idx](x)
            hidden_embed.append(x)

        x = torch.cat(hidden_embed, dim = 1)
        x = self.layers[-1](x, adj_norm)

        return x


    def forward(self, x, adj_norm):
        if self.agg == None:
            out = self.gcn_forward(x, adj_norm)
            return out
        elif self.agg == "concat":
            out = self.concat_forward(x, adj_norm)
            return out
        else:
            pass

        

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]