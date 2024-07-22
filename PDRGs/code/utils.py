"""Various utility functions."""
import numpy as np
import scipy.sparse as sp
import torch
from typing import Union
import networkx as nx
import random
from sklearn.preprocessing import normalize, StandardScaler
from collections import Counter
from scipy.stats import norm
from sklearn.decomposition import TruncatedSVD
from itertools import islice
import os
import logging
from random import sample
from collections import defaultdict
logger = logging.getLogger(__name__)


def load_dataset(dir_net):
    G = nx.read_edgelist(dir_net)
    G.remove_edges_from(nx.selfloop_edges(G))
    lcc_nodes = sorted(max(nx.connected_components(G), key=len))
    G_lcc = G.subgraph(lcc_nodes)
    NODE2ID = {n: i for i, n in enumerate(lcc_nodes)}
    ID2NODE = {i: n for i, n in enumerate(lcc_nodes)}
    row_idx = []
    col_idx = []
    for n_a, n_b in G_lcc.edges:
        row_idx.append(NODE2ID[n_a])
        col_idx.append(NODE2ID[n_b])
        row_idx.append(NODE2ID[n_b])
        col_idx.append(NODE2ID[n_a])

    A = sp.csr_matrix(
        (np.ones_like(np.array(row_idx)), (np.array(row_idx), np.array(col_idx))),
        shape=(len(lcc_nodes), len(lcc_nodes))
    )

    assert np.all(A.diagonal() == 0), 'All diagonal elements of A should be zero.'

    node_neis = {NODE2ID[n]: set([NODE2ID[nb] for nb in G_lcc.neighbors(n)]) for n in G_lcc.nodes()}

    return A, G_lcc, NODE2ID, ID2NODE, node_neis


def l2_reg_loss(model, scale=1e-5):

    loss = 0.0
    for w in model.get_weights():
        loss += w.pow(2.).sum()
    return loss * scale



def to_sparse_tensor(matrix: Union[sp.spmatrix, torch.Tensor, np.array],
                     device
                     ) -> Union[torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor]:


    coo = matrix.tocoo()
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    if device:
        sparse_tensor = sparse_tensor.to(device)
    return sparse_tensor.coalesce()


def feature_generator(A, n_comp, preprocess, device):
    if preprocess.lower() == "none":
        feat = to_sparse_tensor(A, device=device)
    elif preprocess.lower() == "svd":
        svd = TruncatedSVD(n_components=n_comp)
        feat = torch.FloatTensor(svd.fit_transform(A)).to(device)
        logger.info('svd explained variance ratio = {}'.format(svd.explained_variance_ratio_.sum()))
    else:
        raise NotImplementedError('Only support None and svd.')
    return feat

def normalize_adj(adj, sparse=True):

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    if sparse:
        # return res
        return res.tocsr()
    else:
        return res.todense()


def _volume(S, n, degG):
    '''
    volume(node) = sum(neighbors' degrees) + degree of itself
    paper link: https://arxiv.org/pdf/1112.0031.pdf
    :param S: node set S = node + node 1st-order neighbors
    :param degG: dictionary, key = node, value = node degree
    :return: volume(S) (numeric)
    '''

    S.add(n)
    vol = 0
    for m in S:
        vol += degG[m]
    return vol


def _edge(nxG, S, n):
    '''
    paper link: https://arxiv.org/pdf/1112.0031.pdf
    :param nxG:
    :param S: node set S = node + node 1st-order neighbors
    :return: edges(S)
    '''
    S.add(n)
    return 2 * len(nxG.subgraph(S).edges)


def graph_property(nxG):
    '''
    compute degree(node), volume(set), edges(set)
    input: networkx graph object
    :return: condG
    '''
    _degG = {n: len(set(nxG[n])) for n in set(nxG.nodes)} # node degree
    _volG = {n: _volume(set(nxG[n]), n, _degG) for n in set(nxG.nodes)} # volume(node's neighborhood)
    _edgeG = {n: _edge(nxG, set(nxG[n]), n) for n in set(nxG.nodes)} # edges(node's neighborhood)
    _cutG = {n: _volG[n] - _edgeG[n] for n in set(nxG.nodes)} # cut(node's neighborhood)
    _cvolG = {n: _volume(set(nxG.nodes) - set(nxG[n]), n, _degG) for n in set(nxG.nodes)} #vol(S bar)
    condG = {n: _cutG[n] / min(_volG[n], _cvolG[n]) for n in set(nxG.nodes)} # conductance(node's neighborhood)

    return condG



def cluster_number(G):
    '''
    determine the community number
    input: networkx object G
    output: community number
    '''
    cond_G = graph_property(G)
    clustercenter = [n for n in set(G.nodes) if
                      cond_G[n] < min([cond_G[m] for m in G[n]])]
    return len(clustercenter)



def cluster_infer(Z_pred, ID2NODE):
    clust_results = dict()
    for idx in range(Z_pred.shape[0]):
        clust_results[ID2NODE[idx]] = []
        clust_sets = np.where(Z_pred[idx, ] > 0)[0]
        for cidx in clust_sets:
            clust_results[ID2NODE[idx]].append(cidx)
    return clust_results



def load_snp(dir_net, header = False):

    with open(dir_net, mode='r') as f:
        if header:
            next(f)
        # snp_id = []
        gene_id = set()
        for line in f:
            _, id = line.strip("\n").split("\t")
            gene_id.add(id)

    return sorted(list(gene_id))


# def load_snp_pf(dir_file, header = False, use_weight = False):
#     g_weight = dict()
#     pf_collect = defaultdict(set)
#
#     with open(dir_file, mode='r') as f:
#         if header:
#             next(f)
#
#         for line in f:
#             rs_id, g_id = line.strip("\n").split("\t")
#             pf_collect[pf].add(gid)
#
#     for pf in pf_collect:
#         if pf not in ["missing", "No protein family in Uniprot"]:
#             for gid in pf_collect[pf]:
#                 g_weight[gid] = float(1 / len(pf_collect[pf]))
#         else:
#             for gid in pf_collect[pf]:
#                 g_weight[gid] = 1.0
#
#     return g_weight


def evaluate_protein_weight(protein_set, protein_family):
    protein_weight = dict()
    pf_collect = defaultdict(set)
    for p in protein_set:
        curr_family = protein_family[p] if p in protein_family else "unknown"
        pf_collect[curr_family].add(p)

    for p in protein_set:
        if p not in protein_weight:
            curr_family = protein_family[p] if p in protein_family else "unknown"
            if curr_family != "unknown":
                for nds in pf_collect[curr_family]:
                    protein_weight[nds] = float(1/len(pf_collect[curr_family]))
            else:
                protein_weight[p] = 1.0

    return protein_weight


def null_dist_score1(non_snp_genes, rand_num, node_num, curr_gene, cluster_results, protein_family, shared_seed, use_weight):
    random.seed(shared_seed)
    rand_gene_score = []
    gene_clust = set(cluster_results[curr_gene])
    for _ in range(rand_num):
        rand_set = random.sample(non_snp_genes, node_num)
        if use_weight:
            rand_weight = evaluate_protein_weight(rand_set, protein_family)
        rand_score = 0.0
        for rg in rand_set:
            rg_clust = set(cluster_results[rg])
            if len(rg_clust) > 0 and len(gene_clust) > 0:
                curr_weight = rand_weight[rg] if use_weight else 1.0
                rand_score += curr_weight * len(gene_clust & rg_clust) / len(rg_clust)
        rand_gene_score.append(rand_score)
    return rand_gene_score


def sig_score(gene, cluster_results, snp_genes, protein_family, shared_seed, use_weight = False):
    non_snp_genes = sorted(list(set(cluster_results.keys()) - snp_genes))
    if use_weight:
        snp_genes_weight = evaluate_protein_weight(snp_genes, protein_family)
    gene_clust = set(cluster_results[gene])
    gene_score = 0.0
    for sp in snp_genes:
        sp_clust = set(cluster_results[sp])
        if len(sp_clust) > 0 and len(gene_clust) > 0:
            curr_weight = snp_genes_weight[sp] if use_weight else 1.0
            gene_score += curr_weight * len(gene_clust & sp_clust) / len(sp_clust)


    rand_gene_score = null_dist_score1(non_snp_genes, 1000, len(snp_genes), gene, cluster_results, protein_family, shared_seed, use_weight)
    sig_num = sum([s > gene_score for s in rand_gene_score])
    if sig_num / 1000 < 0.05:
        return gene_score
    else:
        return 0



# def null_dist_score1(non_snp_genes, rand_num, node_num, curr_gene, cluster_results, input_weights, shared_seed):
#     random.seed(shared_seed)
#     rand_gene_score = []
#     gene_clust = set(cluster_results[curr_gene])
#     temp_non_snp_genes = set(non_snp_genes)
#     temp_non_snp_genes = sorted(list(temp_non_snp_genes))
#     for _ in range(rand_num):
#         rand_set = random.sample(temp_non_snp_genes, node_num)
#         rand_score = 0.0
#         for idx, rg in enumerate(rand_set):
#             rg_clust = set(cluster_results[rg])
#             if len(rg_clust) > 0 and len(gene_clust) > 0:
#                 rand_score += input_weights[idx] * len(gene_clust & rg_clust) / len(rg_clust)
#         rand_gene_score.append(rand_score)
#     return rand_gene_score
#
#
# def sig_score(gene, cluster_results, snp_clust, non_snp_genes, snp_weight, shared_seed):
#     gene_clust = set(cluster_results[gene])
#     gene_score = 0.0
#     input_weights = sorted(list(snp_weight.values()))
#     assert len(input_weights) == len(snp_clust), 'In utils.py/sig_score, length mismatch!'
#     # print("=="*50)
#     for sp in snp_clust:
#         sp_clust = set(cluster_results[sp])
#         if len(sp_clust) > 0 and len(gene_clust) > 0:
#             gene_score += snp_weight[sp] * len(gene_clust & sp_clust) / len(sp_clust)
#             # temp_score = len(gene_clust & sp_clust) / len(sp_clust)
#             # if temp_score > 0:
#             #     print("gene = {}, sp = {}, temp_score = {}, len(gene_clust) = {}, len(sp_clust) = {}".format(gene, sp, temp_score, len(gene_clust), len(sp_clust)))
#     rand_gene_score = null_dist_score1(non_snp_genes, 1000, len(snp_clust), gene, cluster_results, input_weights, shared_seed)
#     sig_num = sum([s > gene_score for s in rand_gene_score])
#     if sig_num / 1000 < 0.05:
#         return gene_score
#     else:
#         return 0


def sample_epoch_subgraph(N, node_neis, batch_size):

    epoch_nodes = set()
    sample_nodes = set(sample(list(range(N)), batch_size))
    for n in sample_nodes:
        if len(node_neis[n].intersection(sample_nodes)) > 0:
            curr_nei = node_neis[n].intersection(sample_nodes)
            curr_nei.add(n)
            epoch_nodes |= curr_nei
        
    epoch_nodes = list(epoch_nodes)
    return epoch_nodes


def drop_edges(A, percent):

    A = A.tocoo()
    row_idx = A.row
    col_idx = A.col
    nnz = A.nnz
    perm = np.random.permutation(nnz)
    preserve_nnz = int(nnz*percent)
    perm = perm[:preserve_nnz]
    # new_row_idx = np.concatenate((row_idx[perm], col_idx[perm]))
    # new_col_idx = np.concatenate((col_idx[perm], row_idx[perm]))
    A_drop = sp.csr_matrix((np.ones_like(row_idx[perm]), (row_idx[perm], col_idx[perm])), shape=A.shape)
    
    return A_drop


# def drop_edges(epoch, A, percent):

#     upper_A = sp.triu(A, k = 1).tocoo()
#     row_idx = upper_A.row
#     col_idx = upper_A.col
#     nnz = upper_A.nnz
#     perm = np.random.permutation(nnz)
#     preserve_nnz = int(nnz*percent)
#     perm = perm[:preserve_nnz]
#     new_row_idx = np.concatenate((row_idx[perm], col_idx[perm]))
#     new_col_idx = np.concatenate((col_idx[perm], row_idx[perm]))
#     A_drop = sp.csr_matrix((np.ones_like(new_row_idx), (new_row_idx, new_col_idx)), shape=A.shape)
    
#     return A_drop


