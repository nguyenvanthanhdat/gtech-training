from tqdm import tqdm
import os, time, evaluate, torch
from torch.utils.tensorboard import SummaryWriter

def eval(model, tokenizer, valid_data_loader):
    outputs = []
    answers = []
    model.eval()
    for batch in tqdm(valid_data_loader):
        input_token = batch[0].input_ids.squeeze().to('cuda')
        input_mask = batch[0].attention_mask.squeeze().to('cuda')
        ans_token = batch[1]

        output = model.module.generate(
            input_ids = input_token,
            attention_mask = input_mask,
            max_length=128,
            early_stopping=True)
        outputs.append(tokenizer.decode(output[0], skip_special_tokens=True))
        answers.append(ans_token)
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=outputs,references=answers)
    return results['rougeL']


def train(model, tokenizer, steps, learning_rate, train_data_loader, valid_data_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    max_result_roguel = 0
    step = 1
    epochs = 1
    writer = SummaryWriter('/kaggle/working/runs/task1')
    for epoch in range(epochs):
        train_loss = 0
        for batch in train_data_loader:
            print(f'step: {step}')

            input_token = batch[0].input_ids.squeeze().to('cuda')
            input_mask = batch[0].attention_mask.squeeze().to('cuda')
            ans_token = batch[1].input_ids.squeeze().to('cuda')
            loss = model(
                input_ids = input_token,
                attention_mask = input_mask,
                labels=ans_token).loss
            train_loss += loss
            loss.mean().backward()

            optimizer.step()
            optimizer.zero_grad()
            
            if step % 50 == 0 and step != 0:
                result_roguel = eval(model, tokenizer, valid_data_loader)
                writer.add_scalar('RougeL/Valid', result_roguel, step)
                print(f'step: {step} - RougeL: {result_roguel}')
                if result_roguel > max_result_roguel:
                    torch.save(model,'/kaggle/working/model.pt')
                    print("Model saved")
                    max_result_roguel = result_roguel
            step += 1
            if step == 474:
                break
        writer.add_scalar('Loss/Train', train_loss, epoch)
    result_roguel = eval(model, tokenizer, valid_data_loader)
    writer.add_scalar('RougeL/Valid', result_roguel, step)
    if result_roguel > max_result_roguel:
        torch.save(model,'/kaggle/working/model.pt')
        print("Model saved")
        max_result_roguel = result_roguel
    os.system("zip /kaggle/working/model.zip /kaggle/working/model.pt") # compress model

    writer.close()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs