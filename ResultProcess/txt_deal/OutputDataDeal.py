import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPOCH = 50

"Extarct the data"
dir_name="./VGG_UCM"
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
A = pd.DataFrame(A)

A.columns = ['Name','CIDER','ROUGE_L','Bleu_4','Bleu_3','Bleu_2','Bleu_1']
writer = pd.ExcelWriter(dir_name+'/Save_excel.xlsx')
A.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
writer.save()