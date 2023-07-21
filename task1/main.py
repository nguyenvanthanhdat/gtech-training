from tqdm import tqdm
import sys, os, json, argparse, random
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
my_dir = sys.path[0]
sys.path.append(os.path.join(my_dir, 'src/data'))
sys.path.append(os.path.join(my_dir, 'src/models'))
from make_dataset import *
from train_model import *

def set_rand():
    SEED = 1234
    random.seed(SEED)
    # np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.cuda.is_available()

def main(args):
    set_rand()
    train_path = 'docs/ELI5.jsonl'
    valid_path = 'docs/ELI5_val.jsonl'

    with open(train_path, 'r') as json_file:
        json_list_train = list(json_file)
    with open(valid_path, 'r') as json_file:
        json_list_valid = list(json_file)

    train_dataset = ELI5(json_list_train)
    valid_dataset = ELI5(json_list_valid, 'val')

    bs = args.batch_size
    train_data_loader = DataLoader(train_dataset, batch_size=bs,\
        shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=bs,\
        shuffle=False)

    learning_rate = 1e-3
    steps = 500000
    train(model, steps, learning_rate, train_data_loader, valid_data_loader)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', metavar='int',\
        type=int, required=True,\
        help='batch to train model')
    args = parser.parse_args()
    main(args=args)