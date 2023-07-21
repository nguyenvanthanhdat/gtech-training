def reranking(ctxs, answer, model_name):
    sorted_ctxs = []

    # load model 
    model = AutoModelForSequenceClassification.from_pretrained('model_name')
    tokenizer = AutoTokenizer.from_pretrained('model_name')

    # calculate sts
    answers = [answer] * len(ctxs)
    ctxs    = ctxs
    features = tokenizer(
        answer,
        ctxs,
        padding=True,
        truncation=True,
        return_tensor="pt")
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
    print(sorted_ctxs)
    # sort by descending
    return sorted_ctxs