from torch.utils.data import Dataset
import torch
import json
import sys
my_dir = sys.path[0]
from preprocessing import reranking
class ELI5(Dataset):

    def __init__(self, json_file, type_file=None, max_length=None, model=None):
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
        answers = answers[0]
        rerank_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        sorted_ctxs = reranking(ctxs=ctxs, answer=answers,model_name=rerank_model)
        sorted_ctxs = " ".join(sorted_ctxs)
        input_sen = "[CLS] " + question\
            + " [SEP] " + ctxs + " [SEP]"
        
        input_token = tokening(input_sen)
        if self.type_file == None:
            ans_token = tokening(answers)

        return input_token.to('cuda'), ans_token.to('cuda')