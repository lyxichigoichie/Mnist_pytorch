import torch.nn as nn
import torch
#定义超参数，采用SGD作为优化器
# learning_rate = 0.001
# batch_size = 256
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# loss_fn = nn.CrossEntropyLoss()
# model.to(device)
# loss_list = []
# acc_list = []
# epoch_num = []

def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

#定义训练循环和测试循环
def train(train_dataloader, test_dataloader, model, device, loss_fn, optimizer, epoch, ac_func):
    epoch_num = []
    loss_list = []
    acc_list = []
    size = len(train_dataloader.dataset)
    for t in range(epoch):
        print(f"Epoch {t+1}\n-------------------------------")
        running_loss = 0    
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X,ac_func)
            # pred_flat = pred.clone().argmax(dim=1)
            # pred_flat = torch.tensor(pred_flat, requires_grad=True)
            # print(pred_flat)
            # y = torch.tensor(y,dtype = torch.float32)
            loss = loss_fn(pred, y)
            # print(loss)
            running_loss += loss
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 50 == 49:
                # writer.add_scalar('training loss',
                #                 running_loss / 50,
                #                 epoch * len(dataloader)+batch+1)
                
                loss, current = loss.item(), (batch+1) * len(X)
                loss_list.append(loss), epoch_num.append(t+current/size)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                running_loss = 0
        
        correct = test(test_dataloader, model, device, loss_fn, ac_func)
        acc_list.append(correct)
    return acc_list, loss_list

def test(test_dataloader, model, device, loss_fn, ac_func):
    acc_list = []
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X,ac_func)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    # acc_list.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return correct