import torch
from torch_geometric.data import InMemoryDataset
from utils.torch_data_reader import TorchDataReader

class Test8Nodes(InMemoryDataset):

    PATH = "/home/rogerio/git/wiener-impact-model/data/test8nodes/raw/"

    def __init__(self, root, name, transform=None, pre_transform=None):
        super(Test8Nodes, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.name = name

    @property
    def raw_file_names(self):
        return ["graphs.txt", "impacts.txt"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        reader = TorchDataReader(Test8Nodes.PATH)
        data_list = reader.torch_data

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list - [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.name)