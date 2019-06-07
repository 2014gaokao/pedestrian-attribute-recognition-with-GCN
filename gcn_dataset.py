import pickle as pickle
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
class GCNDataset(Dataset):
    def __init__(self,root,set,transform=None,target_transform=None,inp_name=None,adj=None):
        self.root=root
        self.set=set
        self.transform=transform
        self.target_transform=target_transform

        self.dataset=pickle.load(open('dataset/rap2_dataset.pkl','rb'))
        self.partition=pickle.load(open('dataset/rap2_partition.pkl','rb'))
        self.att_name=[self.dataset['att_name'][i] for i in self.dataset['selected_attribute']]
        self.image=[]
        self.label=[]
        for idx in self.partition[self.set][0]: #self.partition['train'][0]
            self.image.append(self.dataset['image'][idx])
            self.label.append(np.array(self.dataset['att'][idx])[self.dataset['selected_attribute']])
        self.label=torch.tensor(self.label)#.tolist()) #标签换成自己想训练的标签

        if inp_name is not None:
            with open(inp_name,'rb') as f:
                self.inp=pickle.load(f)
                #self.inp.requires_grad=False
        else:
            self.inp=np.identity(60)
        self.inp_name=inp_name
    def __len__(self):
        return len(self.image)
    def __getitem__(self, item):
        image,label=self.image[item],self.label[item]
        img=Image.open('dataset/RAP_dataset/'+image)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return (img,image,self.inp),label
    def num_att(self):
        return len(self.dataset['selected_attribute'])