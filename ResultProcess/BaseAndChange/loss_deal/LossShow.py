import pickle
import numpy as np
import matplotlib.pyplot as plt

tool = "bn"
curve1_dir = "./sydney/"+tool+"/vgg/Loss.pickle"
curve2_dir = "./sydney/"+tool+"/resnet/Loss.pickle"
curve3_dir = "./ucm/"+tool+"/vgg/Loss.pickle"
curve4_dir = "./ucm/"+tool+"/resnet/Loss.pickle"
EPOCH = 100

file = open(curve1_dir, 'rb')
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

file = open(curve2_dir, 'rb')
loss= pickle.load(file)
file.close()
loss = np.array(loss)
loss_ave_table = []
for i in range(EPOCH):
    loss_ave = np.average(loss[i])
    loss_ave_table.append(loss_ave)
l2, = plt.plot(epoch,loss_ave_table,linestyle=':',color='b')

file = open(curve3_dir, 'rb')
loss= pickle.load(file)
file.close()
loss = np.array(loss)
loss_ave_table = []
for i in range(EPOCH):
    loss_ave = np.average(loss[i])
    loss_ave_table.append(loss_ave)
l3, = plt.plot(epoch,loss_ave_table,linestyle='-.',color='black')

file = open(curve4_dir, 'rb')
loss= pickle.load(file)
file.close()
loss = np.array(loss)
loss_ave_table = []
for i in range(EPOCH):
    loss_ave = np.average(loss[i])
    loss_ave_table.append(loss_ave)
l4, = plt.plot(epoch,loss_ave_table,color='g')

plt.legend(handles=[l1, l2, l3, l4], labels=['VGG_Sydney', 'ResNet_Sydney','VGG_UCM', 'ResNet_UCM'],  loc='best')
# plt.legend(handles=[l1, l2], labels=['ResNet_Sydney', 'VGG_Sydney'],  loc='best')
plt.show()