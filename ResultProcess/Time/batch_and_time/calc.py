import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPOCH=50

dir_name = ["10.txt","15.txt","20.txt","25.txt","30.txt"]

"all time calc"
time_table = []
for dir in dir_name:
    f = open(dir, 'rt')
    text = f.readlines()
    time = []
    for i in range(50):
        time.append(float(text[i*2+2].replace('\n', '')[69:75]))
    time = np.array(time)
    time_table.append(time)
time_table = np.array(time_table)

"batch time calc"
batch_table = []
for dir in dir_name:
    f = open(dir, 'rt')
    text = f.readlines()
    batch = []
    for i in range(50):
        batch.append(float(text[i*2+1].replace('\n', '')[71:75]))
    batch = np.array(batch)
    batch_table.append(batch)
batch_table = np.array(batch_table)


"Print time"
print("                all time/h       batch time/s")
print("10:             {:.3f}            {:.3f}".format(np.sum(time_table[0])/3600.0,np.average(batch_table[0])))
print("15:             {:.3f}            {:.3f}".format(np.sum(time_table[1])/3600.0,np.average(batch_table[1])))
print("20:             {:.3f}            {:.3f}".format(np.sum(time_table[2])/3600.0,np.average(batch_table[2])))
print("25:             {:.3f}            {:.3f}".format(np.sum(time_table[3])/3600.0,np.average(batch_table[3])))
print("30:             {:.3f}            {:.3f}".format(np.sum(time_table[4])/3600.0,np.average(batch_table[4])))

"""Plot"""
epoch = np.linspace(1,50,50)
l1, = plt.plot(epoch,time_table[0])
l2, = plt.plot(epoch,time_table[1])
l3, = plt.plot(epoch,time_table[2])
l4, = plt.plot(epoch,time_table[3])
l5, = plt.plot(epoch,time_table[4])

plt.xlabel(r'Epoch')
plt.ylabel(r'Time/s')
plt.grid(color='black', linestyle='--', linewidth=1)
plt.title("All Time")
plt.legend(handles=[l1, l2, l3, l4, l5], labels=['10', '15','20', '25', '30'],  loc='best')
plt.show()



# A = np.array(time_table)
# A = pd.DataFrame(A)
# A.index = ["ResNet_Sydney.txt","VGG_Sydney.txt","ResNet_UCM.txt","VGG_UCM.txt"]
# writer = pd.ExcelWriter('Time_Excel.xlsx')
# A.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
# writer.save()
