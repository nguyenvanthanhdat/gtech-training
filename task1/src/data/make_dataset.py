from torch.utils.data import Dataset
import torch
import json
class ELI5(Dataset):

    def __init__(self, type_file, json_file, max_length=None, model=None):
        self.type_file = type_file
        self.json_file = json_file
        self.json_list = []
        self.max_length = max_length
        self.model = model
        for i in range(len(json_file)):
            self.json_list += [json_file[i]]

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, index):
        data = self.json_list[index]
        data_load = json.loads(data)
        question_id = data_load['question_id']
        question = data_load['question']
        answers = data_load['answers']
        ctxs = data_load['ctxs']
        if self.type_file == 'val':
            ctxs = ctxs[0]
        return question_id, question, answers, ctxs