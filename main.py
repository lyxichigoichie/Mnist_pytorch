import torch.nn as nn
import torch
from AlexNet import AlexNet
from dataloader import LoadMnist
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import Trainer


net_config = {\
    'batch_size': 256,
    'device': 'cuda',
    'lr': 0.001,
    'optimizer': 'SGD',
    'loss_fn': 'CrossEntropyLoss',
    'epoch_num': 30,
    'model_path': './Trained/MnistOnAlexNet_epoch30.pkl'
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
    elif net_config['loss_fn'] == 'MSE':
        optimizer = nn.MSELoss()

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
                                  epoch = net_config['epoch_num']
                                  )
    # save model
    torch.save(model.state_dict(), net_config['model_path'])

    # change acc_list, loss_list from list into numpy
    acc_list = np.array(acc_list)
    loss_list = np.array(loss_list)
    # save acc_list and loss_list
    np.save('./loss_acc/loss',loss_list)
    np.save('./loss_acc/acc',acc_list)