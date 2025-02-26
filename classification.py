import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch_geometric import seed_everything
from data import dataset
import utils
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier


def ten_fold_cross_validation(x, y, seed, multi=False):
    MiF1 = []
    MaF1 = []
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    if multi:
        LR = MultiOutputClassifier(
            LogisticRegression(max_iter=1000, penalty='l2', solver='liblinear', random_state=seed))
    else:
        LR = LogisticRegression(penalty='l2', solver='liblinear', random_state=seed)
    for train_index, test_index in kfold.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if torch.any(torch.sum(torch.from_numpy(y_train), dim=0) == 0):
            continue
        LR.fit(x_train, y_train)
        if multi:
            LR_pred = LR.predict_proba(x_test)
            LR_pred = predict_with_n(y_test, LR_pred)
        else:
            LR_pred = LR.predict(x_test)
        MiF1.append(f1_score(y_test, LR_pred, average='micro', zero_division=1))
        MaF1.append(f1_score(y_test, LR_pred, average='macro', zero_division=1))
    avgMi = sum(MiF1) / len(MiF1)
    avgMa = sum(MaF1) / len(MaF1)
    print(round(avgMi, 4), round(avgMa, 4))


def predict_with_n(y_test, y_pred_pro):
    y_pred_pro = np.array(y_pred_pro)
    y_pred_pro = y_pred_pro[:, :, 1].reshape(len(y_pred_pro), len(y_pred_pro[0]))
    y_pred_pro = np.transpose(y_pred_pro)
    y_pred = np.zeros(y_pred_pro.shape)
    for i in range(y_test.shape[0]):
        n = sum(y_test[i])
        top_n = y_pred_pro[i, :].argsort()[-n:]
        y_pred[i, top_n] = 1
    return y_pred


def node_classification_radio(x, y, seed, multi=False):
    MiF1 = [0] * 10
    MaF1 = [0] * 10
    for times in range(3):
        idx = []
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed + times)
        for train_index, test_index in kfold.split(x):
            idx.append(test_index)
        if multi:
            labels = torch.sum(torch.from_numpy(y[idx[0]]), dim=0)
            zero_indices = [index for index, value in enumerate(labels) if value == 0]
            for index in zero_indices:
                column = y[:, index]
                sample_index = np.where(column == 1)[0]
                idx[0] = np.append(idx[0], sample_index[0])
        for i in range(9):
            test_index = []
            train_index = []

            for j in range(i + 1):
                train_index.append(idx[j])
            for j in range(i + 1, 10):
                test_index.append(idx[j])
            test_index = np.concatenate(test_index).ravel()
            train_index = np.concatenate(train_index).ravel()

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if multi:
                LR = MultiOutputClassifier(
                    LogisticRegression(max_iter=1000, penalty='l2', solver='liblinear', random_state=seed + times))
                LR.fit(x_train, y_train)
                LR_pred = LR.predict_proba(x_test)
                LR_pred = predict_with_n(y_test, LR_pred)
            else:
                LR = LogisticRegression(penalty='l2', solver='liblinear', random_state=seed + times)
                LR.fit(x_train, y_train)
                LR_pred = LR.predict(x_test)
            MiF1[i] += f1_score(y_test, LR_pred, average='micro', zero_division=1)
            MaF1[i] += f1_score(y_test, LR_pred, average='macro', zero_division=1)
    for i in range(9):
        print(round(MiF1[i] / 3 * 100, 2), round(MaF1[i] / 3 * 100, 2))


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
        x = encode.detach().cpu().numpy()
    else:
        x = encode

    # eval
    dataset = args.data.lower()
    multi = False
    if dataset in ['ppi', 'wikipedia', 'blogcatalog']:
        multi = True
    print('ten-fold:')
    ten_fold_cross_validation(x, y, args.seed, multi)
    print('train radio from 0.1 to 0.9:')
    node_classification_radio(x, y, args.seed, multi)
