from torch.utils.data import Dataset
import torch
import json
class ELI5(Dataset):

    def __init__(self, json_file):
        self.json_file = json_file
        self.json_list = []
        for i in range(len(json_file)):
            self.json_list += [json_file[i]]

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, index):
        data = self.json_list[index]
        data_load = json.loads(data)
        
        return data_load