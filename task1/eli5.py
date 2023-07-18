from torch.utils.data import Dataset
import torch
import json
class ELI5(Dataset):

    def __init__(self, json_file_name):
        self.json_file_name = json_file_name
        self.json_list = []
        with open(json_file_name, 'r') as json_file:
            json_list = list(json_file)

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, index):
        json = self.json_list[index]
        json_load = json.loads(json)
        
        return json_load