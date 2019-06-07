import pickle as pickle
import numpy as np
from scipy.io import loadmat
import torch.nn as nn
import torch

data=loadmat('dataset/RAP_annotation/RAP_annotation.mat')
dataset=pickle.load(open('dataset/rap2_dataset.pkl','rb'))
partition=pickle.load(open('dataset/rap2_partition.pkl','rb'))
select=[]
for idx in partition['train'][0]:  # self.partition['train'][0]
    select=np.array(dataset['att'][idx])[dataset['selected_attribute']].tolist()

all=[]
for idx in range(152):
    all.append(data['RAP_annotation'][0][0][2][idx][0][0])

select_name=[]
for i in range(len(dataset['selected_attribute'])):
    select_name.append(all[dataset['selected_attribute'][i]])

word_to_ix = {j: i for i, j in enumerate(select_name)}
embeds=nn.Embedding(60,300)
word2vec=torch.tensor([])
for i in range(len(select)):
    lookup_tensor=torch.tensor([word_to_ix[select_name[i]]],dtype=torch.long)
    embed=embeds(lookup_tensor)
    word2vec=torch.cat((word2vec,embed),0)
#print(word2vec.size())
word2vec=word2vec.detach().numpy()
with open('dataset/glove.pkl','wb+') as f:
    pickle.dump(word2vec,f)