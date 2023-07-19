from tqdm import tqdm
import sys, os, json, argparse
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from eli5 import *

def main(args):
    my_dir = sys.path[0]
    sys.path.append(os.path.join(my_dir, '..'))
    train_path = 'docs/ELI5.jsonl'
    valid_path = 'docs/ELI5_val.jsonl'

    with open(train_path, 'r') as json_file:
        json_list_train = list(json_file)
    with open(valid_path, 'r') as json_file:
        json_list_valid = list(json_file)

    train_dataset = ELI5(json_list_train)
    valid_dataset = ELI5(json_list_valid)

    bs = args.batch_size
    print(bs)
    train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', metavar='int', required=True,
                        help='batch to train model')
    args = parser.parse_args()
    main(args=args)