import pickle
import numpy as np
import matplotlib.pyplot as plt

fat_dir = "epoch50lr0.0001/"
EPOCH = 50

file = open(fat_dir+'ResNet_Sydney.pickle', 'rb')
loss= pickle.load(file)
file.close()
loss = np.array(loss)
loss_ave_table = []
for i in range(EPOCH):
    loss_ave = np.average(loss[i])
    loss_ave_table.append(loss_ave)
epoch = np.linspace(1,EPOCH,EPOCH)
l1, = plt.plot(epoch,loss_ave_table,linestyle='--',color='r')
plt.xlabel(r'Epoch')
plt.ylabel(r'Loss')
plt.grid(color='black', linestyle='--', linewidth=1)
# plt.title("Loss "+fat_dir)
plt.title("Loss")

file = open(fat_dir+'VGG_Sydney.pickle', 'rb')
loss= pickle.load(file)
file.close()
loss = np.array(loss)
loss_ave_table = []
for i in range(EPOCH):
    loss_ave = np.average(loss[i])
    loss_ave_table.append(loss_ave)
l2, = plt.plot(epoch,loss_ave_table,linestyle=':',color='b')

file = open(fat_dir+'ResNet_UCM.pickle', 'rb')
loss= pickle.load(file)
file.close()
loss = np.array(loss)
loss_ave_table = []
for i in range(EPOCH):
    loss_ave = np.average(loss[i])
    loss_ave_table.append(loss_ave)
l3, = plt.plot(epoch,loss_ave_table,linestyle='-.',color='black')

file = open(fat_dir+'VGG_UCM.pickle', 'rb')
loss= pickle.load(file)
file.close()
loss = np.array(loss)
loss_ave_table = []
for i in range(EPOCH):
    loss_ave = np.average(loss[i])
    loss_ave_table.append(loss_ave)
l4, = plt.plot(epoch,loss_ave_table,color='g')

plt.legend(handles=[l1, l2, l3, l4], labels=['ResNet_Sydney', 'VGG_Sydney','ResNet_UCM', 'VGG_UCM'],  loc='best')
# plt.legend(handles=[l1, l2], labels=['ResNet_Sydney', 'VGG_Sydney'],  loc='best')
plt.show()