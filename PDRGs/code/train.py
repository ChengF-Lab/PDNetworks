import torch
import torch.nn.functional as F
from utils import load_dataset, feature_generator, l2_reg_loss, cluster_infer, cluster_number, \
    to_sparse_tensor, normalize_adj, sample_epoch_subgraph, drop_edges
# from sampler import get_edge_sampler
from sampler import sample_edges
import scipy.sparse as sp
from model import PolyGCN
from loss import loss_batch, loss_cpu
import stopping
from torch.optim import NAdam
import numpy as np
import math
import pickle
import os
from numpy.linalg import norm
from torch.nn.utils import clip_grad_norm_
import logging
logger = logging.getLogger(__name__)



def retrieve_clusters(args):

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
        torch.cuda.set_device(device)


    A, G_lcc, NODE2ID, ID2NODE, node_neis  = load_dataset(args.dirnet)

    NODE_NUM = len(G_lcc.nodes())
    EDGE_NUM = len(G_lcc.edges())

    node_cutoff = float(1 / NODE_NUM)
    edge_cutoff = float((2 * EDGE_NUM) / (NODE_NUM * (NODE_NUM - 1)))

    N = A.shape[0]  # total nodes

    logger.info('There are {} of nodes in the largest connected component (LCC) of PPI!'.format(N))

    batch_size = int(N * args.p_bs)

    K = cluster_number(G_lcc)
    
    if args.K > K:
        K = args.K
 
    feat = feature_generator(A, args.n_comp, preprocess=args.preprocess, device = device)

    adj_norm = to_sparse_tensor(normalize_adj(A, sparse = True), device = device)

    hidden_size = [int(s) for s in args.hidden_size]

    gnn = PolyGCN(adj = A, input_dim=feat.shape[1], hidden_dims=hidden_size, output_dim=K,
                  batch_norm=args.batch_norm, n_nei=(args.n_nei + 1), dropout=args.dropout, agg = "concat").to(device)

    opt = NAdam(gnn.parameters(), lr=args.lr)

    val_loss = np.inf
    validation_fn = lambda: val_loss
    early_stopping = stopping.NoImprovementStopping(validation_fn, patience=args.patience)

    temp_hs = "_hidden-size"
    for idx in range(len(hidden_size)):
        temp_hs += ("_" + str(hidden_size[idx]))

    save_file_name = f"NAdam_A_poly{args.n_nei}_wd{args.weight_decay}_dropout{args.dropout}_lr_{args.lr}_K{K}_bs{batch_size}_patience{args.patience}_lrmin{args.lr_min}_max-epochs{args.epochs}_preprocess{args.preprocess}"
    save_file_name += temp_hs

    model_out = args.dirresult + save_file_name + ".pth"
    model_saver = stopping.ModelSaver(model = gnn, optimizer = opt, dir_model = model_out)


    f_out = open(args.dirresult + save_file_name + ".txt", "a")
    # f_grad_out = open(args.dirresult + save_file_name + "_grad.txt", "a")
    lr = args.lr

    if args.pretrained:

        # model_saver.restore(device)
        #
        # gnn.eval()
        # Z = F.relu(gnn(feat, adj_norm))
        #
        # Z_cpu = Z.cpu().detach().numpy()
        #
        # _, _, full_loss = loss_cpu(Z_cpu, A)
        #
        # logger.info(f'loss.full = {full_loss:.4f}')
        #
        # thresh = math.sqrt(-math.log(1 - edge_cutoff))
        # Z_pred = Z_cpu > thresh
        #
        # clust_results = cluster_infer(Z_pred, ID2NODE)
        #
        # with open(args.dirresult + save_file_name + '_cluster_results_edge.pickle', 'wb') as handle:
        #     pickle.dump(clust_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # handle.close()

        with open(args.dirresult + save_file_name + '_cluster_results.pickle', 'rb') as handle:
            clust_results = pickle.load(handle)
        handle.close()

    else:
        # for epoch in range(1):
        for epoch in range(args.epochs):

            if (epoch + 1) % args.val_step == 0:
                lr = max(lr * args.lr_decay, args.lr_min)
                for param_group in opt.param_groups:
                    param_group['lr'] = lr

            epoch_nodes = sample_epoch_subgraph(N = N, node_neis = node_neis, batch_size = batch_size)

            A_batch = A[epoch_nodes][:,epoch_nodes]

            A_batch_drop = drop_edges(A = A_batch, percent = args.p_edgedrop)

            adj_drop_epoch_norm = to_sparse_tensor(normalize_adj(A_batch_drop, sparse = True), device = device)

            ones_idx, zeros_idx = sample_edges(A_sel = A_batch)

            # Training step
            gnn.train()
            opt.zero_grad()
            Z_batch = F.relu(gnn(x = feat[epoch_nodes], adj_norm = adj_drop_epoch_norm))
            # Z_batch = F.relu(Z_batch)

            loss = loss_batch(Z_batch, ones_idx, zeros_idx, A_batch.nnz)
            loss += l2_reg_loss(gnn, scale=args.weight_decay)
            loss.backward()
            clip_grad_norm_(gnn.parameters(), 1)
            opt.step()


            if epoch == 0 or (epoch + 1) % args.val_step == 0:

                logger.info('*' * 100)

                with torch.no_grad():

                    gnn.eval()

                    Z = F.relu(gnn(feat, adj_norm))
                    # Z = F.relu(Z)
                    pos_full, neg_full, full_loss = loss_cpu(Z.cpu().detach().numpy(), A)

                    adj_norm_epoch = to_sparse_tensor(normalize_adj(A_batch, sparse = True), device = device)

                    Z_batch = F.relu(gnn(feat[epoch_nodes], adj_norm_epoch))
                    # Z_batch = F.relu(Z_batch)

                    pos_batch, neg_batch, batch_loss = loss_cpu(Z_batch.cpu().detach().numpy(), A_batch)
                    pos_val = (pos_full * A.nnz - pos_batch * A_batch.nnz) / (A.nnz - A_batch.nnz)
                    neg_val = (neg_full * (A.shape[0] * A.shape[0] - A.shape[0] - A.nnz) - neg_batch * (A_batch.shape[0] * A_batch.shape[0] - A_batch.shape[0] - A_batch.nnz)) / (
                            (A.shape[0] * A.shape[0] - A.shape[0] - A.nnz) - (A_batch.shape[0] * A_batch.shape[0] - A_batch.shape[0] - A_batch.nnz))

                    val_ratio = ((A.shape[0] * A.shape[0] - A.shape[0] - A.nnz) - (A_batch.shape[0] * A_batch.shape[0] - A_batch.shape[0] - A_batch.nnz)) / (
                                A.nnz - A_batch.nnz)
                    val_loss = (pos_val + val_ratio * neg_val) / (1 + val_ratio)

                    # logger.info('*' * 100)
                    logger.info("#s of existing edges (total) = {}".format(int(A.nnz // 2)))
                    logger.info("#s of existing edges  (training) = {}".format(int(A_batch.nnz // 2)))
                    logger.info("#s of non-existing edges (total) = {}".format(int((A.shape[0] * A.shape[0] - A.shape[0] - A.nnz) // 2)))
                    logger.info("#s of non-existing edges (training) = {}".format(int((A_batch.shape[0] * A_batch.shape[0] - A_batch.shape[0] - A_batch.nnz) // 2)))
                    logger.info(f'Epoch {epoch:4d}, loss.train = {batch_loss:.4f}, loss.val = {val_loss:.4f}, loss.full = {full_loss:.4f}')
                    logger.info("pos_batch = {}, neg_batch = {}".format(pos_batch, neg_batch))
                    logger.info("pos_val = {}, neg_val = {}".format(pos_val, neg_val))
                    logger.info("pos_full = {}, neg_full = {}".format(pos_full, neg_full))


                    # Check if it's time for early stopping / to save the model
                    early_stopping.next_step()
                    if early_stopping.should_save():
                        logger.info('======= Write Loss to Output! ======')
                        f_out.write(
                            f'Epoch {epoch:4d}, loss.train = {batch_loss:.4f}, loss.val = {val_loss:.4f}, loss.full = {full_loss:.4f}')
                        f_out.write('\n')
                        model_saver.save()
                    if early_stopping.should_stop():
                        logger.info(f'Breaking due to early stopping at epoch {epoch}')
                        break

        f_out.close()

        model_saver.restore(device)

        gnn.eval()
        Z = F.relu(gnn(feat, adj_norm))

        Z_cpu = Z.cpu().detach().numpy()

        _, _, full_loss = loss_cpu(Z_cpu, A)

        logger.info(f'loss.full = {full_loss:.4f}')

        thresh = math.sqrt(-math.log(1 - 1 / N))
        Z_pred = Z_cpu > thresh

        clust_results = cluster_infer(Z_pred, ID2NODE)

        with open(args.dirresult + save_file_name + '_cluster_results.pickle', 'wb') as handle:
            pickle.dump(clust_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

        # # remove .pth file
        # for out_file in os.listdir(args.dirresult):
        #     if out_file == model_out:
        #         os.remove(os.path.join(args.dirresult, out_file))

    return clust_results, save_file_name
