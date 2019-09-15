# pedestrian-attribute-recognition-with-GCN

## Preparation

<font face="Times New Roman" size=4>
  
**Prerequisite: Python 3.6 and torch 1.1.0 and tqdm**

**Download and prepare RAP(v2) dataset and annotation then put in dataset directory**

</font>

## Train the model

<font face="Times New Roman" size=4>

   ```
   python transform_rap2.py     (transform data)
   python glove.py      (word2vec)
   python adj.py      (Adjacency matrix)
   python train.py      (weight file will locate in checkpoint directory)
   ``` 
</font>

## Demo

<font face="Times New Roman" size=4>
  
## checkpoint.pth.tar [Baidu Yun](https://pan.baidu.com/s/1m4Na3AFtZrl5i1jsEJD8qQ),password:5z1j
   ```
   python demo.py
   ``` 
</font>

## Methodology
![image](https://github.com/2014gaokao/pedestrian-attribute-recognition-with-GCN/blob/master/image/%E7%BB%98%E5%9B%BE1.jpg)
