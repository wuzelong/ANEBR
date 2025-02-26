import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, node_size, nhid0, nhid1, activation_function):
        super(Encoder, self).__init__()
        self.encode0 = nn.Linear(node_size, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.af = activation_function

    def forward(self, A):
        if self.af == 'Leaky_Relu':
            encode = F.leaky_relu(self.encode0(A))
            encode = F.leaky_relu(self.encode1(encode))
        else:
            encode = F.sigmoid(self.encode0(A))
            encode = F.sigmoid(self.encode1(encode))
        return encode


class Decoder(nn.Module):
    def __init__(self, node_size, nhid0, nhid1, activation_function):
        super(Decoder, self).__init__()
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size)
        self.af = activation_function

    def forward(self, encode):
        if self.af == 'Leaky_Relu':
            decode = F.leaky_relu(self.decode0(encode))
            decode = F.leaky_relu(self.decode1(decode))
        else:
            decode = F.sigmoid(self.decode0(encode))
            decode = F.sigmoid(self.decode1(decode))
        return decode


class ANELR(nn.Module):
    def __init__(self, node_size, nhid0, nhid1, activation_function):
        super(ANELR, self).__init__()
        self.online_encoder = Encoder(node_size, nhid0, nhid1, activation_function)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.online_decoder = Decoder(node_size, nhid0, nhid1, activation_function)

    def cosine_similarity(self, X1, X2):
        X1 = F.normalize(X1, p=2, dim=1)
        X2 = F.normalize(X2, p=2, dim=1)
        X1[torch.isnan(X1)] = 0
        X2[torch.isnan(X2)] = 0
        cos_sim = X1 @ X2.t()
        return cos_sim

    def forward(self, A, A_tilde, C, S):
        H = self.online_encoder(A)
        decode = self.online_decoder(H)
        with torch.no_grad():
            H_tilde = self.target_encoder(A_tilde)

        L_emb = 1 - self.cosine_similarity(H, H)
        L_emb = torch.sum(S * L_emb)

        L_rec = torch.linalg.norm((decode - A) * C, ord='nuc')
        return L_emb, L_rec, H_tilde.detach(), H

    @torch.no_grad()
    def update_target_network(self, tau):
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(tau).add_(param_q.data, alpha=1. - tau)

    def savector(self, A):
        H = self.online_encoder(A)
        decode = self.online_decoder(H)
        return H, decode


class Discriminator(nn.Module):
    def __init__(self, in_features=128):
        super().__init__()
        self.disc = nn.Sequential(nn.Linear(in_features, 32),
                                  nn.LeakyReLU(0.1),
                                  nn.Linear(32, 1),
                                  nn.Sigmoid()
                                  )

    def forward(self, data):
        return self.disc(data)
