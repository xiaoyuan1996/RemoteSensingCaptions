import numpy as np
import matplotlib.pyplot as plt

EPOCH = 100

"Extarct the data"
dir = ["./Resnet_Sydney","./VGG_Sydney","./Resnet_UCM","./VGG_UCM"]

data = []
for dir_name in dir:
    i = 1
    f = open(dir_name+'/{}'.format(i), 'rt')
    ls = f.readlines()
    A = []

    for i in range(1, EPOCH+1):
        f = open(dir_name+'/{}'.format(i), 'r')

        ls = f.readlines()

        a = [ls[0].replace('\n', ''),
             ls[34].replace('\n', ''),
             ls[32].replace('\n', ''),
             ls[30].replace('\n', ''),
             ls[29].replace('\n', ''),
             ls[28].replace('\n', ''),
             ls[27].replace('\n', '')]

        line = []
        line.append(int(a[0].replace('.npy', '')))
        line.append( float(a[1].replace('CIDEr: ', '')))
        line.append(float(a[2].replace('ROUGE_L: ', '')))
        line.append(float(a[3].replace('Bleu_4: ', '')))
        line.append(float(a[4].replace('Bleu_3: ', '')))
        line.append(float(a[5].replace('Bleu_2: ', '')))
        line.append(float(a[6].replace('Bleu_1: ', '')))

        A.append(line)
    A = np.array(A)
    A = A[np.lexsort(A[:, ::-1].T)]
    data.append(A)
data = np.array(data)

# """plot CIDEr"""
# x= np.linspace(1,EPOCH,EPOCH)
# CIDEr = []
#
# l1, = plt.plot(x,data[0,:,1],linestyle='--',color='r')
# l2, = plt.plot(x,data[1,:,1],linestyle=':',color='b')
# l3, = plt.plot(x,data[2,:,1],linestyle='-.',color='black')
# l4, = plt.plot(x,data[3,:,1],color='g')
# plt.grid(color='black', linestyle='--', linewidth=1)
# plt.title("CIDEr")
# plt.xlabel("Epoch")
# plt.legend(handles=[l1, l2, l3, l4], labels=['ResNet_Sydney','VGG_Sydney','ResNet_UCM', 'VGG_UCM'],  loc='best')
#
# plt.show()

# """plot ROUGE_L"""
# x= np.linspace(1,EPOCH,EPOCH)
# ROUGE_L = []
#
# l1, = plt.plot(x,data[0,:,2],linestyle='--',color='r')
# l2, = plt.plot(x,data[1,:,2],linestyle=':',color='b')
# l3, = plt.plot(x,data[2,:,2],linestyle='-.',color='black')
# l4, = plt.plot(x,data[3,:,2],color='g')
# plt.grid(color='black', linestyle='--', linewidth=1)
# plt.title("ROUGE_L")
# plt.xlabel("Epoch")
# plt.legend(handles=[l1, l2, l3, l4], labels=['ResNet_Sydney','VGG_Sydney','ResNet_UCM', 'VGG_UCM'],  loc='best')
#
# plt.show()

"""plot BLEU"""
x= np.linspace(1,EPOCH,EPOCH)

plt.subplot(2,2,1)
BLEU_4 = []
l1, = plt.plot(x,data[0,:,3],linestyle='--',color='r')
l2, = plt.plot(x,data[1,:,3],linestyle=':',color='b')
l3, = plt.plot(x,data[2,:,3],linestyle='-.',color='black')
l4, = plt.plot(x,data[3,:,3],color='g')
plt.grid(color='black', linestyle='--', linewidth=1)
plt.title("BLEU_4")
plt.xlabel("Epoch")
plt.legend(handles=[l1, l2, l3, l4], labels=['ResNet_Sydney','VGG_Sydney','ResNet_UCM', 'VGG_UCM'],  loc='best')

plt.subplot(2,2,2)
BLEU_3 = []
l1, = plt.plot(x,data[0,:,4],linestyle='--',color='r')
l2, = plt.plot(x,data[1,:,4],linestyle=':',color='b')
l3, = plt.plot(x,data[2,:,4],linestyle='-.',color='black')
l4, = plt.plot(x,data[3,:,4],color='g')
plt.grid(color='black', linestyle='--', linewidth=1)
plt.title("BLEU_3")
plt.xlabel("Epoch")
plt.legend(handles=[l1, l2, l3, l4], labels=['ResNet_Sydney','VGG_Sydney','ResNet_UCM', 'VGG_UCM'],  loc='best')

plt.subplot(2,2,3)
BLEU_2 = []
l1, = plt.plot(x,data[0,:,4],linestyle='--',color='r')
l2, = plt.plot(x,data[1,:,4],linestyle=':',color='b')
l3, = plt.plot(x,data[2,:,4],linestyle='-.',color='black')
l4, = plt.plot(x,data[3,:,4],color='g')
plt.grid(color='black', linestyle='--', linewidth=1)
plt.title("BLEU_2")
plt.xlabel("Epoch")
plt.legend(handles=[l1, l2, l3, l4], labels=['ResNet_Sydney','VGG_Sydney','ResNet_UCM', 'VGG_UCM'],  loc='best')

plt.subplot(2,2,4)
BLEU_1 = []
l1, = plt.plot(x,data[0,:,5],linestyle='--',color='r')
l2, = plt.plot(x,data[1,:,5],linestyle=':',color='b')
l3, = plt.plot(x,data[2,:,5],linestyle='-.',color='black')
l4, = plt.plot(x,data[3,:,5],color='g')
plt.grid(color='black', linestyle='--', linewidth=1)
plt.title("BLEU_1")
plt.xlabel("Epoch")
plt.legend(handles=[l1, l2, l3, l4], labels=['ResNet_Sydney','VGG_Sydney','ResNet_UCM', 'VGG_UCM'],  loc='best')


plt.show()