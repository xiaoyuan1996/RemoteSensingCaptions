import pickle
import numpy as np
import cv2 as cv
import pandas as pd
import skimage.transform
import matplotlib.pyplot as plt




"""Load the alpha value"""
file = open('./last_output_all.pickle', 'rb')
last_output_all = pickle.load(file)
file.close()
last_output_all = np.array(last_output_all)

A=[]
for i in range(20):
    A.append(last_output_all[i, 0, :])
A = np.array(A)
print(A.shape)


"""Save to Excel"""
B = pd.DataFrame(A)
writer = pd.ExcelWriter('./Excel.xlsx')
B.to_excel(writer,'page_1',float_format='%.9f') # float_format 控制精度
writer.save()

