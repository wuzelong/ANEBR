import random
import torch
from torch_geometric import seed_everything
from torch.utils.data.dataloader import DataLoader
import utils
from data import dataset


def sample_hide(matrix, dataset):
    upper_triangular = torch.triu(matrix, diagonal=1)
    idx_pos = torch.nonzero(upper_triangular)
    idx_neg = torch.nonzero(upper_triangular == 0)
    total = idx_pos.size(0)
    radio = 45
    range_pos = idx_pos.size(0)
    range_neg = idx_neg.size(0)
    sample_cnt = int(total * radio / 100.0)

    for i in range(5):
        sample_pos = random.sample(range(range_pos), sample_cnt)
        pos = torch.index_select(idx_pos, 0, torch.tensor(sample_pos).long())
        undirect_pos = torch.zeros_like(pos)
        undirect_pos[:, 1] = pos[:, 0]
        undirect_pos[:, 0] = pos[:, 1]
        torch.save(torch.cat((pos, undirect_pos), dim=0), 'data/pos' + '/' + dataset + '_pos_{:0>3}'.format(radio) + '.pth')

        sample_neg = random.sample(range(range_neg), sample_cnt)
        neg = torch.index_select(idx_neg, 0, torch.tensor(sample_neg).long())
        undirect_neg = torch.zeros_like(neg)
        undirect_neg[:, 1] = neg[:, 0]
        undirect_neg[:, 0] = neg[:, 1]
        torch.save(torch.cat((neg, undirect_neg), dim=0), 'data/neg' + '/' + dataset + '_neg_{:0>3}'.format(radio) + '.pth')

        idx_pos = pos
        idx_neg = neg
        range_pos = idx_pos.size(0)
        range_neg = idx_neg.size(0)
        radio -= 10
        sample_cnt = int(total * radio / 100.0)


if __name__ == '__main__':
    # init
    args = utils.parse_args()
    seed_everything(args.seed)

    # dataset
    A, y = utils.load_data(args.data)
    Data = dataset.Dataload(A, A.shape[0])
    Data = DataLoader(Data, batch_size=args.bs, shuffle=True)

    # sample_hide(A, args.data)  # create hidden links
    pos = torch.load('data/pos/' + args.data + '_pos_' + args.radio + '.pth')
    neg = torch.load('data/neg/' + args.data + '_neg_' + args.radio + '.pth')
    A[pos[:, 0], pos[:, 1]] = 0  # hide links

    # train
    encode, decode = utils.train_ANELR(Data, A, args)

    # eval
    if torch.is_tensor(decode):
        decode = decode.detach().cpu().numpy()
    rankings, labels = utils.get_rankings_2D(decode, pos, neg)
    auc = utils.get_AUC(rankings, labels)
    recall = utils.get_Recall(labels, len(pos), len(pos))
    print(auc, recall)
