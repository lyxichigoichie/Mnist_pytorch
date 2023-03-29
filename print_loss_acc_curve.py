import matplotlib.pyplot as plt
import numpy as np

loss_list = np.load('./loss_acc/loss.npy')
acc_list = np.load('./loss_acc/acc.npy')

fig, ax = plt.subplots()  # Create a figure containing a single axes.
fig.set_size_inches(5,5)
# loss_list_index = np.array(range(len(loss_list))
ax.plot(np.arange(len(loss_list))/4,loss_list) 
ax.plot(np.arange(len(acc_list)),acc_list)

plt.savefig('./figures/loss_acc_curve.png')
plt.show()