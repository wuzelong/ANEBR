import numpy as np
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch import nn
from entmax import entmax_bisect
from sklearn import metrics
import networkx as nx
import scipy as sp
from models.ANELR import ANELR, Discriminator


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--seed', default=20240409, type=int, help="random seed")
    parser.add_argument('--lr', default=0.001, type=float, help="the learning rate of optimizer")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--data', default='adjnoun')
    parser.add_argument('--nhid0', default=256, type=int)
    parser.add_argument('--nhid1', default=128, type=int)
    parser.add_argument('--activation_function', default='Leaky_Relu')
    parser.add_argument('--radio', default='005')

    parser.add_argument('--eta1', default=1, type=float)
    parser.add_argument('--eta2', default=1, type=float)
    parser.add_argument('--gamma', default=2, type=float)
    parser.add_argument('--tau', default=0.99, type=float)
    parser.add_argument('--beta', default=0.01, type=float)

    args = parser.parse_args()

    return args


def load_data(dataset):
    dataset = dataset.lower()
    if dataset in ['blogcatalog', 'ppi', '20-newsgroups', 'wikipedia']:
        data = sp.io.loadmat('data/' + dataset + '.mat')
        A = data['network'].toarray()
        A = torch.from_numpy(A).to(torch.float32)
        y = data['group'].toarray().astype(int)
        if dataset == '20-newsgroups':
            y = np.ravel(y)
        elif dataset == 'wikipedia':
            A[A > 0] = 1
    elif dataset in ['adjnoun']:
        graph = nx.read_gml('data/' + dataset + '.gml')
        A = nx.to_numpy_array(graph)
        A = torch.from_numpy(A)
        A = A.to(torch.float32)
        y = np.array([graph.nodes[node]['value'] for node in graph.nodes])
        y = np.ravel(y)
    else:
        raise ValueError("Wrong Dataset.")

    rows, cols = A.size()
    diag_indices = torch.arange(min(rows, cols))
    A[diag_indices, diag_indices] = 1
    return A, y


def getKatz(adj, beta):
    # (I-beta*A)^-1 -I
    n = len(adj)
    I = torch.eye(n)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    I = I.to(device)
    adj = adj.to(device)
    A = beta * adj
    IA = I - A
    try:
        s_matrix = torch.inverse(IA) - I
    except torch.linalg.LinAlgError:
        s_matrix = torch.pinverse(IA) - I
    return s_matrix


def laplacian_norm(A):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    D = torch.diag(A.sum(dim=1)).to(device)
    try:
        D_inv = torch.inverse(D)
    except torch.linalg.LinAlgError:
        D_inv = torch.pinverse(D)
    D = torch.sqrt(D_inv)
    return D @ A @ D


def train_ANELR(Data, A, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ANELR(A.shape[0], args.nhid0, args.nhid1, args.activation_function)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model = model.to(device)
    A = A.to(device)
    model.train()
    model_opt = model
    first = True
    loss_opt = -1

    L = laplacian_norm(A)
    S = getKatz(L, args.beta)
    W = entmax_bisect(S, alpha=2)
    A_tilde = W.clone()
    A_tilde[A_tilde != 0] = 1
    A_tilde = A_tilde.to(device)
    C = torch.where(A == 0, 1, args.gamma + W)

    disc = Discriminator(in_features=args.nhid1).to(device)
    opt_dis = torch.optim.Adam(disc.parameters(), lr=args.lr)
    criterion = nn.BCELoss(reduction="mean")
    for epoch in range(args.epochs):
        loss_sum = 0
        for index in Data:
            L_emb, L_rec, H_real, H_fake = model(A[index], A_tilde[index], C[index], W[index][:, index])
            # update discriminator
            disc_real, disc_fake = disc(H_real), disc(H_fake.detach())
            L_dis = criterion(disc_real, torch.ones_like(disc_real)) + criterion(disc_fake, torch.zeros_like(disc_fake))
            opt_dis.zero_grad()
            L_dis.backward()
            opt_dis.step()
            # update generator
            disc_fake_updated = disc(H_fake)
            L_gen = criterion(disc_fake_updated, torch.ones_like(disc_fake_updated))
            L_NSNE = args.eta1 * L_emb + args.eta2 * L_gen + L_rec
            opt.zero_grad()
            L_NSNE.backward()
            opt.step()
            model.update_target_network(args.tau)
            loss_sum += L_NSNE.item()
        if loss_opt > loss_sum or first:
            loss_opt = loss_sum
            model_opt = model
            first = False

    model_opt.eval()
    encode, decode = model_opt.savector(A)
    return encode, decode


def get_rankings_2D(scores, pos, neg):
    pos_rankings = scores[pos[:, 0], pos[:, 1]]
    pos_labels = np.ones_like(pos_rankings)

    neg_rankings = scores[neg[:, 0], neg[:, 1]]
    neg_labels = np.zeros_like(neg_rankings)

    rankings = np.concatenate([pos_rankings, neg_rankings])
    labels = np.concatenate([pos_labels, neg_labels])

    sorted_indices = np.argsort(rankings)[::-1]
    rankings = rankings[sorted_indices]
    labels = labels[sorted_indices]
    return rankings, labels


def get_AUC(rankings, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, rankings)
    auc = metrics.auc(fpr, tpr)
    return round(auc, 4)


def get_Recall(labels, total, k):
    cnt = 0
    for i in range(k):
        if labels[i] == 1:
            cnt += 1
    recall = cnt / total
    return round(recall, 4)


def get_Precision(labels, k):
    cnt = 0
    for i in range(k):
        if labels[i] == 1:
            cnt += 1
    recall = cnt / k
    return round(recall, 4)
