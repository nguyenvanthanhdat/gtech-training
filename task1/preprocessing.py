from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
    LongT5ForConditionalGeneration
import numpy as np
import torch

def reranking(ctxs, answer, model_name):
    sorted_ctxs = []

    # load model 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to('cuda')


    # calculate sts
    answers = [answer] * len(ctxs)
    ctxs    = ctxs
    features = tokenizer(
        answers,
        ctxs,
        padding=True,
        truncation=True,
        return_tensors="pt").to('cuda')
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits.to('cpu')
    
    # sort by descending
    tensor_array = np.array(scores)
    sorted_indices = np.argsort(tensor_array, axis=0)[::-1]
    sorted_ctxs = [ctxs[i[0]] for i in sorted_indices]
    
    return sorted_ctxs

def tokening(senteces, length):
    tokenizer = AutoTokenizer.from_pretrained(
        "t5-small")
    input_token = tokenizer(
        senteces,
        max_length=length,
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt" )
    return input_token