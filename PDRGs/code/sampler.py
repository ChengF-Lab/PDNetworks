import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data as data_utils
import networkx as nx
import random



def sample_edges(A_sel):
    num_sel_nodes = A_sel.shape[0]
    edges = np.transpose(sp.tril(A_sel, 1).nonzero())
    num_neg = edges.shape[0]

    # Select num_neg non-edges
    generated = False
    while not generated:
        candidate_ne = np.random.randint(0, num_sel_nodes, size=(2*num_neg, 2), dtype=np.int64)
        cne1, cne2 = candidate_ne[:, 0], candidate_ne[:, 1]
        to_keep = (1 - A_sel[cne1, cne2]).astype(np.bool).A1 * (cne1 != cne2)
        next_nonedges = candidate_ne[to_keep][:num_neg]
        generated = to_keep.sum() >= num_neg

    return torch.LongTensor(edges), torch.LongTensor(next_nonedges)






