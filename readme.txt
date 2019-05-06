This repository is used for remote sensing captions based on the "Encoder-Decoder" with attention, and here has some data analysis module. 

Envirment:
Python: 3.5
TensorFlow: 1.4.1
cuda: 8.0

Parameter:
CNN: VGGNet , ResNet
LSTM: with attention , no attention
trick: Batch Normalization, Fine Tune
Dataset: Sydney, UCM

Use Step:
1.download zip from link below, then unzip.
链接：https://pan.baidu.com/s/1H95ZBvhZToSeL35mii_3wg   提取码：gtua
2.put resnet_v1_50.ckpt and vgg16.npy in /.
3.use images dir /train replace ./data/sydney/images and use /UCM_Captions replace ./data/UCM_Captions .
4.run main.py

ps: you can change the cnn and dataset by change config.py  and main.py, and you can choose train, eval or test in main.py

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                            Coding Instruction
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
***************                 directory               **********************
data:               all data including sydney and ucm dataset, pay attention to json file, json file is produce by Datatransfer.py
models:             this file is used to save the modles after every epoch
ResultProcess:      this file is used to process and save the data produced by model
summary:            save the summary file for tensorboard
test:               a file for image test
utils:              a file used for captions evaluating
val:                a file to save eval result.json

***************                   file                  **********************
base_model.py       construct train, eval and test
config.py           something about the coding configuration
dataset.py          prepare the data into tf.data
DatasetTransfer.py  change format of the remote sensing data into format of coco data
main.py             main coding of this project
model.py            construct the network
resnet_v1_50.ckpt   this is a pretend model parameter of 50 layers resnet
vgg16.npy           this is a pretend model parameter of 16 layers VGGNet
vocabulary.csv      vocabulary saved by coding, if you want to change the dataset you should delete this at first and let codes produce another one