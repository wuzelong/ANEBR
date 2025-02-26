import torch
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch_geometric import seed_everything
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
    if torch.is_tensor(decode):
        decode = decode.detach().cpu().numpy()

    # eval
    pos = torch.nonzero(A)
    neg = torch.nonzero(A == 0)
    rankings, labels = utils.get_rankings_2D(decode, pos, neg)
    auc = utils.get_AUC(rankings, labels)
    precision = utils.get_Precision(labels, int(len(pos) * 0.2))
    print('AUC', auc, 'Precision', precision)

    if args.data == 'adjnoun':
        plt.imshow(decode, vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.axis('off')
        plt.savefig('reconstruction.pdf', bbox_inches='tight')
        plt.show()
