import os
# os.rename("./images/24.tif","./images/24.jpg")
files = os.listdir('./images')
i = 1
for file in files:
    os.rename("./images/"+file, "./images/"+str(i)+'.jpg')
    i += 1