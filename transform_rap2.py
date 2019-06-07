import numpy as np
from scipy.io import loadmat
import pickle as pickle

dataset=dict()
dataset['image']=[]
dataset['att']=[]
dataset['att_name']=[]
data=loadmat('dataset/RAP_annotation/RAP_annotation.mat')
#for idx in range(152):
    #print(idx,data['RAP_annotation'][0][0][2][idx][0][0])
#[  0   1   2   3   4   7   8   9  11  12  13  14  15  16  17  21  22  23
#  24  25  26  27  28  29  30  45  47  48  50  51  52  67  68  69  70  72
#  73  88  89  90  92  93  94  95  97  98 100 101 105 106 107 108 109 110]
dataset['selected_attribute']=np.array([0 ,1 , 2 , 3 , 4 , 7 , 8 , 9 , 13 , 14 , 15 , 16 , 17 ,
                                        #21,22,23,24,25,26,27,28,29,30,
                                        23, #Vest
                                        31,32,33,34,35,36,37,38,39,40,41,42,#43, #ub
                                        #44,45,46,47,48,49,50,51,52,
                                        47, #skirt
                                        53,54,55,56,57,58,59,60,61,62,63,64 ,#65,  #lb
                                        #66,67,68,69,70,71,72,73,
                                        67, #shoes-Leather
                                        74,75,76,77,78,79,80,81,82,83,84,85,#86,  #shoes
                                        #87,
                                        88,89,90,91,92,93,94,95,
                                        #96,97
                                        ]).tolist()
for idx in range(152):
    dataset['att_name'].append(data['RAP_annotation'][0][0][2][idx][0][0])
for idx in range(84928):
    dataset['image'].append(data['RAP_annotation'][0][0][0][idx][0][0])
    dataset['att'].append(data['RAP_annotation'][0][0][1][idx,:].tolist())
with open('dataset/rap2_dataset.pkl','wb+') as f:
    pickle.dump(dataset,f)

partition=dict()
partition['train']=[]
partition['val']=[]
partition['test']=[]
partition['trainval']=[]

partition['weight_train']=[]
partition['weight_trainval']=[]
for idx in range(5):
    train=(data['RAP_annotation'][0][0][4][0,idx][0][0][0][0,:]-1).tolist()
    val = (data['RAP_annotation'][0][0][4][0, idx][0][0][1][0, :] - 1).tolist()
    test = (data['RAP_annotation'][0][0][4][0, idx][0][0][2][0, :] - 1).tolist()
    trainval=train+val
    partition['train'].append(train)
    partition['val'].append(val)
    partition['test'].append(test)
    partition['trainval'].append(trainval)

    weight_train=np.mean(data['RAP_annotation'][0][0][1][train,:].astype('float32')==1,axis=0).tolist()
    weight_trainval = np.mean(data['RAP_annotation'][0][0][1][trainval, :].astype('float32') == 1, axis=0).tolist()
    partition['weight_train'].append(weight_train)
    partition['weight_trainval'].append(weight_trainval)
with open('dataset/rap2_partition.pkl','wb+') as f:
    pickle.dump(partition,f)