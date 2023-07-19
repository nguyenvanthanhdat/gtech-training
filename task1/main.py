from tqdm import tqdm
import sys, os, json, argparse, random
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
my_dir = sys.path[0]
sys.path.append(os.path.join(my_dir, 'src\data'))
from make_dataset import *

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
    valid_dataset = ELI5(json_list_valid)

    bs = args.batch_size
    _, question, _, ctxs = train_dataset[0]
    question_tokens = tokenizer.encode(question,\
        return_tensors='pt')
    for doc in ctxs:
        doc_tokens = tokenizer.encode(doc, return_tensors='pt')
        combined_tokens = torch.cat((question_tokens, doc_tokens), dim=1)
        segments = [combined_tokens[:, i:i+max_seq_length]\
            for i in range(0, combined_tokens.size(1), max_seq_length)]
        for segment in segments:
            output = model.generate(
                segment,
                max_length=1000,  # Set the maximum length for generated answers
                num_beams=5,
                num_return_sequences=1,
                early_stopping=True
            )
            answers.append(tokenizer.decode(output[0]))

    # train_data_loader = DataLoader(train_dataset, batch_size=bs,\
    #     shuffle=True)

    valid_data_loader = DataLoader(valid_dataset, batch_size=bs,\
        shuffle=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', metavar='int',\
        type=int, required=True,\
        help='batch to train model')
    args = parser.parse_args()
    main(args=args)