from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

#获取数据集
trans = [transforms.ToTensor()]
trans.insert(0, transforms.Resize(224))
trans = transforms.Compose(trans)
batch_size = 256

def LoadMnist(root):
    train_data = datasets.MNIST(
        root=root,
        train=True,
        download=True,                                                                                                                                                                                              
        transform=trans
    )

    test_data = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=trans
    )
    return train_data,test_data



# train_iter = data.DataLoader(training_data, batch_size, shuffle=True,
#                         num_workers=2)
# test_iter = data.DataLoader(test_data, batch_size, shuffle=False,
#                         num_workers=2)

# train_features, train_labels = next(iter(train_iter))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# print(f"batch size:{len(iter(train_iter))}")