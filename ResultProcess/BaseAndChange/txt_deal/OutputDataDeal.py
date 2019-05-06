import numpy as np
import pandas as pd

dir_name = "./sydney/wt/vgg/"
# dir_name = "./test/"
sub_dir = "txt"
i = 1
f = open(dir_name+sub_dir+'/{}'.format(i), 'rt')

ls = f.readlines()
A = []

for i in range(1, 101):
    f = open(dir_name+sub_dir+'/{}'.format(i), 'r')

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

    cider_temp = float(a[1].replace('CIDEr: ', ''))
    rouge_l_temp = float(a[2].replace('ROUGE_L: ', ''))
    bleu4_temp=float(a[3].replace('Bleu_4: ', ''))
    line.append(0.25*cider_temp+0.5*rouge_l_temp+0.25*bleu4_temp)

    A.append(line)
A = np.array(A)

A = A[np.lexsort(-A.T)]
print("Name   	CIDER	 ROUGE_L	Bleu_4	 Bleu_3	Bleu_2	 Bleu_1	sum")
print(A[0])

A = pd.DataFrame(A)

A.columns = ['Name','CIDER','ROUGE_L','Bleu_4','Bleu_3','Bleu_2','Bleu_1','sum']
writer = pd.ExcelWriter('./all_excel/'+dir_name.replace('/','_').replace('.','')+'.xlsx')
A.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
writer.save()