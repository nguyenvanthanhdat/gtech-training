# def 

tokenizer = AutoTokenizer.from_pretrained(
    "t5-small")
config = LongT5Config("t5-small")
model = LongT5ForConditionalGeneration(
    "model.pt", config=config)


