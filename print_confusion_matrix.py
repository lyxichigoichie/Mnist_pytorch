import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from AlexNet import AlexNet
import torch

model = AlexNet()
model.load_state_dict(torch.load('./Trained/MnistOnAlexNet_epoch30.pkl'))
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
#评估模型
model.eval()
model.to('cpu')
pred_list = torch.tensor([])
y_list = torch.tensor([])
with torch.no_grad():
    for X, y in test_dataloader:
        pred = model(X)
        y_list = torch.cat([y_list,y])
        pred_list = torch.cat([pred_list, pred])

pred111 = pred_list.clone().argmax(dim=1)
sns.set()
fig,ax = plt.subplots()
y_true = [0,0,1,2,1,2,0,2,2,0,1,1]
y_pred = [1,0,1,2,1,0,0,2,2,0,1,1]
C2 = confusion_matrix(y_list,pred111,labels=[0,1,2,3,4,5,6,7,8,9])
#打印 C2
print(C2)
sns.heatmap(C2,annot=True,ax=ax) #画热力图

ax.set_title('confusion matrix') #标题
ax.set_xlabel('predict') #x 轴
ax.set_ylabel('true') #y 轴
fig.set_size_inches(14,14)