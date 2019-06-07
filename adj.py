import pickle as pickle
import numpy as np

dataset=pickle.load(open('dataset/rap2_dataset.pkl','rb'))
partition=pickle.load(open('dataset/rap2_partition.pkl','rb'))

item=[np.array(dataset['att'][idx])[dataset['selected_attribute']] for idx in partition['train'][0]]
sum=np.sum(item,axis=0)
#print(sum)
concur=np.zeros((len(dataset['selected_attribute']),len(dataset['selected_attribute'])))
#print(concur)

for idx in partition['train'][0]:
    t=np.array(dataset['att'][idx])[dataset['selected_attribute']]
    for i in range(len(np.array(dataset['att'][idx])[dataset['selected_attribute']])):
        if t[i]==1:
            for j in range(len(np.array(dataset['att'][idx])[dataset['selected_attribute']])):
                if t[j]==1 and j!=i:
                    concur[i][j]+=1
data={'nums':sum,'adj':concur}
with open('dataset/adj.pkl','wb+') as f:
    pickle.dump(data,f)





