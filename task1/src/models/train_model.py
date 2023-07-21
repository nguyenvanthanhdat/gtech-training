import evaluate
def train(model, steps, learning_rate, train_data_loader, valid_data_loader):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    max_result_roguel = 0
    step = 0
    while(step < steps):
        for batch in train_data_loader:
            if step > steps:
                break

            input_token = batch[0]
            ans_token = batch[1]
            outputs = model.generate(input_token, labels=ans_token)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            if step % 1000 == 0:
                result_roguel = eval(model, valid_data_loader)
                if result_roguel > max_result_roguel:
                    torch.save(model,'model.pt')
                    max_result_roguel = result_roguel
            
            step += 1

def eval(model, valid_data_loader):
    outputs = []
    answers = []
    for batch in valid_data_loader:
        input_token = batch[0]
        ans_token = batch[1]

        output = model.generate(input_ids)
        outputs.append(tokenizer.decode(output[0], skip_special_tokens=True))
        answers.append(ans_token)
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=outputs,references=answers)
    return results['rougeL']