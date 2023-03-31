import torch.nn as nn
import torch
from AlexNet import AlexNet
from dataloader import LoadMnist
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import Trainer
from torch.nn import functional as F


net_config = {\
    'batch_size': 256,
    'device': 'cuda',
    'lr': 0.001,
    'optimizer': 'SGD',
    'loss_fn': 'CrossEntropyLoss',
    # 'loss_fn': 'MSELoss',
    'epoch_num': 30,
    }


if __name__ == '__main__':
    # Load dataset
    train_data, test_data = LoadMnist(root = './Dataset')
    # Dataloader
    train_dataloader = DataLoader(dataset = train_data,
                                  batch_size = net_config['batch_size'],
                                  shuffle = True,
                                  num_workers = 2)
    test_dataloader = DataLoader(dataset = test_data,
                                  batch_size = net_config['batch_size'],
                                  shuffle = True,
                                  num_workers = 2)
    
    # load network
    model = AlexNet()

    # define optimizer and loss function
    if net_config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = net_config['lr'])

    if net_config['loss_fn'] == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
        model_path = './Trained/epoch30_CrossEntropyLoss.pkl'
    elif net_config['loss_fn'] == 'MSELoss':
        loss_fn = nn.MSELoss()
        model_path = './Trained/epoch30_MSELoss.pkl'

    model_path = './Trained/epoch_CrossEntropyLoss_sigmoid.pkl'

    model.to(device=net_config['device'])

    # init weights
    model.apply(Trainer.init_weights)
    # train
    acc_list, loss_list = Trainer.train(train_dataloader = train_dataloader,
                                  test_dataloader = test_dataloader,
                                  model = model,
                                  device = net_config['device'],
                                  loss_fn = loss_fn,
                                  optimizer = optimizer,
                                  epoch = net_config['epoch_num'],
                                  ac_func = torch.sigmoid
                                  )
    # save model
    torch.save(model.state_dict(), model_path)

    # change acc_list, loss_list from list into numpy
    acc_list = np.array(acc_list)
    loss_list = np.array(loss_list)
    # save acc_list and loss_list to print picture
    np.save('./data/MSE_loss',loss_list)
    np.save('./data/MSE_acc',acc_list)

    #评估模型
    model.eval()
    model.to('cpu')
    pred_tensor = torch.tensor([])
    y_list = torch.tensor([])
    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X, torch.sigmoid)
            y_list = torch.cat([y_list,y])
            pred_tensor = torch.cat([pred_tensor, pred])
    pred_list = pred_tensor.clone().argmax(dim=1)
    # save y_list and pred_list in test_dataset
    y_list = y_list.numpy()
    pred_list = pred_list.numpy()
    np.save('./data/CrossEntropyLoss_sigmoid_y_list',y_list)
    np.save('./data/CrossEntropyLoss_sigmoid_pred_list',pred_list)
