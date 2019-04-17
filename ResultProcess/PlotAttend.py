import pickle
import numpy as np
import cv2 as cv
import pandas as pd
import skimage.transform
import matplotlib.pyplot as plt

MAX_CAPTION = 20

"""Load origin image"""
# Load image value
img = cv.imread('D:/Python/tensorflow/image caption coding/test/images/1.jpg')
img = np.array(cv.resize(img,(224,224)))
print('image shape:',img.shape)


"""Load the alpha value"""
file = open('./alpha.pickle', 'rb')
alpha = pickle.load(file)
file.close()
alpha = np.array(alpha)
A=[]
for i in range(MAX_CAPTION):
    A.append(alpha[i, 0, :])

A = np.array(A)
print(A.shape)
"""Save to Excel"""
B = pd.DataFrame(A)
writer = pd.ExcelWriter('./attend/Excel.xlsx')
B.to_excel(writer,'page_1',float_format='%.9f') # float_format 控制精度
writer.save()

"""Visual Operation"""
for i in range(MAX_CAPTION):
    plt.imshow(img)
    alp_curr = A[i].reshape(14, 14)
    alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20, multichannel=False) #16 #20
    plt.imshow(alp_img, alpha=0.65)
    plt.savefig("./attend/"+str(i)+".jpg")
plt.show()