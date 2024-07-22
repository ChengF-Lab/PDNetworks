import numpy as np
import torch


def loss_batch(emb, ones_idx, zeros_idx, num_edges):

    num_nodes = emb.shape[0]

    edge_prob = num_edges / (num_nodes ** 2 - num_nodes)

    eps = -np.log(1 - edge_prob)

    # Loss for edges
    e1, e2 = ones_idx[:, 0], ones_idx[:, 1]
    edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
    loss_edges = -torch.mean(torch.log(-torch.expm1(-eps - edge_dots)))

    # Loss for non-edges
    ne1, ne2 = zeros_idx[:, 0], zeros_idx[:, 1]
    loss_nonedges = torch.mean(torch.sum(emb[ne1] * emb[ne2], dim=1))

    # sampled #s connected edges = #s non-connected edges
    return (loss_edges + loss_nonedges) / 2


def loss_cpu(emb, adj):

    e1, e2 = adj.nonzero()

    num_nodes = adj.shape[0]

    num_edges = e1.shape[0]

    edge_prob = num_edges / (num_nodes ** 2 - num_nodes)

    eps = -np.log(1 - edge_prob)

    edge_dots = np.sum(emb[e1] * emb[e2], axis=1)
    loss_edges = -np.sum(np.log(-np.expm1(-eps - edge_dots)))

    # Correct for overcounting F_u * F_v for edges and nodes with themselves
    self_dots_sum = np.sum(emb * emb)
    correction = self_dots_sum + np.sum(edge_dots)
    sum_emb = np.transpose(np.sum(emb, axis = 0))
    loss_nonedges = np.sum(emb @ sum_emb) - correction

    pos_loss = loss_edges / num_edges

    num_nonedges = num_nodes ** 2 - num_nodes - num_edges

    neg_loss = loss_nonedges / num_nonedges

    true_ratio = num_nonedges / num_edges

    # return pos_loss, neg_loss, (pos_loss + neg_loss) / 2
    return pos_loss, neg_loss, (pos_loss + true_ratio * neg_loss) / (1 + true_ratio)
    