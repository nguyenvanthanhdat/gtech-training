from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch

def reranking(ctxs, answer, model_name):
    sorted_ctxs = []

    # load model 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # calculate sts
    answers = [answer] * len(ctxs)
    ctxs    = ctxs
    features = tokenizer(
        answers,
        ctxs,
        padding=True,
        truncation=True,
        return_tensors="pt")
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
    
    tensor_array = np.array(scores)
    sorted_indices = np.argsort(tensor_array, axis=0)[::-1]
    sorted_ctxs = [ctxs[i[0]] for i in sorted_indices]
    # sort by descending
    return sorted_ctxs