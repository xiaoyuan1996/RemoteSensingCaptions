# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:40:54 2019

@author: Dry
"""

 
import json       
import random
      
dataset = json.load(open(r'C:\Users\lenovo\Desktop\dataset.json','r'))
data1 = dataset.get('images')#我是将images字典中的元素列表取出来存放在data1里
data_train_t = []
data_test_t = []
data_val_t = []
#print(data1)
for i in range(len(data1)):  
    if data1[i]['split'] == 'train':       
        data_train_t.append(data1[i])   
    elif data1[i]['split'] == 'test':  
        data_test_t.append(data1[i])   
    else :       
        data_val_t.append(data1[i])        
#到这步为止就已经将sydney中的训练集，验证集，测试集分好了
data_train = {'images':data_train_t,'dataset':'Sydney'}
data_test = {'images':data_test_t,'dataset':'Sydney'}
data_val = {'images':data_val_t,'dataset':'Sydney'}

#f = open('data_train.json','w')
#json.dump(data_train,f)
#
#f = open('data_test.json','w')
#json.dump(data_test,f)
#
#f = open('data_val.json','w')
#json.dump(data_val,f)

#print(data_train['images'][0])



#将train数据集改为coco
data_train_coco = {}
data_train_coco['info'] = {'description':'This is a outstanding work we did.','auther':'Dry','date_created':'2019-3-14 16:06'}
data_train_coco['images'] = []
data_train_coco['licenses'] = [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'}, {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'}, {'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'}, {'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 'id': 6, 'name': 'Attribution-NoDerivs License'}, {'url': 'http://flickr.com/commons/usage/', 'id': 7, 'name': 'No known copyright restrictions'}, {'url': 'http://www.usa.gov/copyright.shtml', 'id': 8, 'name': 'United States Government Work'}]
data_train_coco['annotations'] = []

for i in data_train['images']:
    data_train_coco['images'].append(i)
   
for i in data_train_coco['images']:
    i['license'] = random.randint(1,5)
    i['file_name'] = i['filename']
    i['id'] = i['imgid']
    i['height'] = 500
    i['width'] = 500
    i['coco_url'] = 'None'
    i['date_captured'] = '2019-3-14 17:06:45'
    i['flickr_url'] = 'None'
       
    data_train_coco['annotations'] += i['sentences']

    del i['sentids']
    del i['imgid']
    del i['sentences']
    del i['split']
    del i['filename']    

for i in data_train_coco['annotations']:
    i['image_id'] = i['imgid']
    i['id'] = i['sentid']
    i['caption'] = i['raw']
    
    del i['tokens']
    del i['raw']
    del i['imgid']
    del i['sentid'] 
 
f = open('data_train_coco.json','w')
json.dump(data_train_coco,f)  





data_test_coco = {}
data_test_coco['info'] = {'description':'This is a outstanding work we did.','auther':'Dry','date_created':'2019-3-14 16:06'}
data_test_coco['images'] = []
data_test_coco['licenses'] = [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'}, {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'}, {'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'}, {'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 'id': 6, 'name': 'Attribution-NoDerivs License'}, {'url': 'http://flickr.com/commons/usage/', 'id': 7, 'name': 'No known copyright restrictions'}, {'url': 'http://www.usa.gov/copyright.shtml', 'id': 8, 'name': 'United States Government Work'}]
data_test_coco['annotations'] = []

for i in data_test['images']:
    data_test_coco['images'].append(i)
   
for i in data_test_coco['images']:
    i['license'] = random.randint(1,5)
    i['file_name'] = i['filename']
    i['id'] = i['imgid']
    i['height'] = 500
    i['width'] = 500
    i['coco_url'] = 'None'
    i['date_captured'] = '2019-3-14 17:06:45'
    i['flickr_url'] = 'None'
       
    data_test_coco['annotations'] += i['sentences']

    del i['sentids']
    del i['imgid']
    del i['sentences']
    del i['split']
    del i['filename']    

for i in data_test_coco['annotations']:
    i['image_id'] = i['imgid']
    i['id'] = i['sentid']
    i['caption'] = i['raw']
    
    del i['tokens']
    del i['raw']
    del i['imgid']
    del i['sentid'] 
 
f = open('data_test_coco.json','w')
json.dump(data_test_coco,f)  


data_val_coco = {}
data_val_coco['info'] = {'description':'This is a outstanding work we did.','auther':'Dry','date_created':'2019-3-14 16:06'}
data_val_coco['images'] = []
data_val_coco['licenses'] = [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'}, {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'}, {'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'}, {'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 'id': 6, 'name': 'Attribution-NoDerivs License'}, {'url': 'http://flickr.com/commons/usage/', 'id': 7, 'name': 'No known copyright restrictions'}, {'url': 'http://www.usa.gov/copyright.shtml', 'id': 8, 'name': 'United States Government Work'}]
data_val_coco['annotations'] = []

for i in data_val['images']:
    data_val_coco['images'].append(i)
   
for i in data_val_coco['images']:
    i['license'] = random.randint(1,5)
    i['file_name'] = i['filename']
    i['id'] = i['imgid']
    i['height'] = 500
    i['width'] = 500
    i['coco_url'] = 'None'
    i['date_captured'] = '2019-3-14 17:06:45'
    i['flickr_url'] = 'None'
       
    data_val_coco['annotations'] += i['sentences']

    del i['sentids']
    del i['imgid']
    del i['sentences']
    del i['split']
    del i['filename']    

for i in data_val_coco['annotations']:
    i['image_id'] = i['imgid']
    i['id'] = i['sentid']
    i['caption'] = i['raw']
    
    del i['tokens']
    del i['raw']
    del i['imgid']
    del i['sentid'] 
 
f = open('data_val_coco.json','w')
json.dump(data_val_coco,f)  