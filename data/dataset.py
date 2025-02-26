from torch.utils import data


class Dataload(data.Dataset):

    def __init__(self, Adj, Node):
        self.Adj = Adj
        self.Node = Node

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.Node