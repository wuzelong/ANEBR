import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from torch.utils.data.dataloader import DataLoader
from torch_geometric import seed_everything
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from matplotlib.colors import ListedColormap
from data import dataset
import utils

if __name__ == '__main__':
    # init
    args = utils.parse_args()
    seed_everything(args.seed)

    # dataset
    A, y = utils.load_data(args.data)
    Data = dataset.Dataload(A, A.shape[0])
    Data = DataLoader(Data, batch_size=args.bs, shuffle=True)

    # train
    encode, decode = utils.train_ANELR(Data, A, args)
    if torch.is_tensor(encode):
        encode = encode.detach().cpu().numpy()

    # eval
    tsne = TSNE(n_components=2, metric='cosine', perplexity=30, random_state=args.seed)
    X_tsne = tsne.fit_transform(encode)

    sc = round(silhouette_score(X_tsne, y), 4)
    chs = round(calinski_harabasz_score(X_tsne, y), 4)

    kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=args.seed)
    y_pred = kmeans.fit_predict(X_tsne)
    nmi = round(normalized_mutual_info_score(y, y_pred), 4)
    ari = round(adjusted_rand_score(y, y_pred), 4)
    print('SC:', sc, ' CHS:', chs, ' NMI:', nmi, ' ARI:', ari)

    if args.data == '20-Newsgroups':
        colors = ['#3682be', '#93c555', '#f05326']
        custom_cmap = ListedColormap(colors)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap=custom_cmap, c=y, s=3)
        plt.axis('off')
        plt.savefig("visualization.pdf", bbox_inches='tight')
        plt.show()
