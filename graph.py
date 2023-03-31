import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from AlexNet import AlexNet
import torch
import numpy as np


def print_confusion_matrix(y_list, pred_list, save_path):
    sns.set()
    fig,ax = plt.subplots()
    C = confusion_matrix(y_list,pred_list,labels=[0,1,2,3,4,5,6,7,8,9])

    sns.heatmap(C,annot=True,ax=ax) #画热力图

    ax.set_title('confusion matrix') #标题
    ax.set_xlabel('predict') #x 轴
    ax.set_ylabel('true') #y 轴
    fig.set_size_inches(14,14)
    
    plt.savefig(save_path)
    # plt.show()

def print_loss_acc_curve(loss_list, acc_list, save_path):
    # loss_list = np.load('./loss_acc/loss.npy')
    # acc_list = np.load('./loss_acc/acc.npy')

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    fig.set_size_inches(5,5)
    # loss_list_index = np.array(range(len(loss_list))
    ax.plot(np.arange(len(loss_list))/4,loss_list) 
    ax.plot(np.arange(len(acc_list)),acc_list)

    plt.savefig(save_path)
    # plt.show()