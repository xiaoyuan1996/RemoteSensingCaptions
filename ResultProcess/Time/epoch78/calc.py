import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPOCH=78

Sydney_name = ["ResNet_Sydney.txt","VGG_Sydney.txt"]
UCM_name = ["ResNet_UCM.txt","VGG_UCM.txt"]
"all time calc"
time_table = []
for dir in Sydney_name:
    f = open(dir, 'rt')
    text = f.readlines()
    time = []
    for i in range(EPOCH):
        time.append(float(text[i*2+2].replace('\n', '')[69:75]))
    time = np.array(time)
    time_table.append(time)

for dir in UCM_name:
    f = open(dir, 'rt')
    text = f.readlines()
    time = []
    for i in range(EPOCH):
        time.append(float(text[i*2+2].replace('\n', '')[57:64]))
    time = np.array(time)
    time_table.append(time)
time_table = np.array(time_table)

"batch time calc"
batch_table = []
for dir in Sydney_name:
    f = open(dir, 'rt')
    text = f.readlines()
    batch = []
    for i in range(EPOCH):
        batch.append(float(text[i*2+1].replace('\n', '')[71:75]))
    batch = np.array(batch)
    batch_table.append(batch)
for dir in UCM_name:
    f = open(dir, 'rt')
    text = f.readlines()
    batch = []
    for i in range(EPOCH):
        batch.append(float(text[i*2+1].replace('\n', '')[59:64]))
    batch = np.array(batch)
    batch_table.append(batch)
batch_table = np.array(batch_table)

"Print time"
print("                all time/h       batch time/s")
print("ResNet_Sydney:  {:.3f}            {:.3f}".format(np.sum(time_table[0])/3600.0,np.average(batch_table[0])))
print("VGG_Sydney:     {:.3f}            {:.3f}".format(np.sum(time_table[1])/3600.0,np.average(batch_table[1])))
print("ResNet_UCM:     {:.3f}           {:.3f}".format(np.sum(time_table[2])/3600.0,np.average(batch_table[2])))
print("VGG_UCM:        {:.3f}           {:.3f}".format(np.sum(time_table[3])/3600.0,np.average(batch_table[3])))


"""Plot"""
epoch = np.linspace(1,EPOCH,EPOCH)
l1, = plt.plot(epoch,time_table[0])
l2, = plt.plot(epoch,time_table[1])
l3, = plt.plot(epoch,time_table[2])
l4, = plt.plot(epoch,time_table[3])


plt.xlabel(r'Epoch')
plt.ylabel(r'Time')
plt.grid(color='black', linestyle='--', linewidth=1)
plt.title("Time")
plt.legend(handles=[l1, l2, l3, l4], labels=['ResNet_Sydney', 'VGG_Sydney','ResNet_UCM', 'VGG_UCM'],  loc='best')
plt.show()



# A = np.array(time_table)
# A = pd.DataFrame(A)
# A.index = ["ResNet_Sydney.txt","VGG_Sydney.txt","ResNet_UCM.txt","VGG_UCM.txt"]
# writer = pd.ExcelWriter('Time_Excel.xlsx')
# A.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
# writer.save()
