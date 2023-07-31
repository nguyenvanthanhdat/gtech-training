# Prepare data: https://huggingface.co/datasets/wikipedia
# get text and id
# chunk text and embedding each text
# database has 3 column: id, chunk, embedding
# input: sentence
# embedding sentence
# postgres retrieval the documents
# [CLS] + question + [SEP] + support chunk (top) + [SEP]
# generate the answer  




# TODO 2: get similarity with embedding
# TODO 3: tokenize the special token
# TODO 4: generate answer
# TODO 5: implement in cmd