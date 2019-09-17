# pedestrian-attribute-recognition-with-GCN

## Preparation

<font face="Times New Roman" size=4>
  
**Prerequisite: Python 3.6 and torch 1.1.0 and tqdm**

**Download RAP(v2) dataset and annotation then put in dataset directory**

</font>

## Train the model

<font face="Times New Roman" size=4>
  
  ( If you simply want to run the demo code without further modification, you might skip this step by downloading the weight file from
  [Baidu Yun](https://pan.baidu.com/s/1m4Na3AFtZrl5i1jsEJD8qQ) with password "5z1j" and put the model_best.pth.tar into directory         /checkpoint/ then run <br />
  python demo.py )

   ```
   python transform_rap2.py     (transform data)
   python glove.py      (word2vec)
   python adj.py      (Adjacency matrix)
   python train.py      (weight file will locate in checkpoint directory)
   ``` 
</font>

## Methodology
![image](https://github.com/2014gaokao/pedestrian-attribute-recognition-with-GCN/blob/master/image/%E7%BB%98%E5%9B%BE1.jpg)

## Superiority

| method | mA | accuracy | precision | recall | F1 |
|:-----:|---|---|---|---|---|
|ACN|69.66|62.61|80.12|72.26|75.98|
|DeepMar|73.79|	62.02|	74.92|	76.21	|75.56|
|HP-Net|76.12	|65.39	|77.33	|78.79	|78.05|
|JRL|77.81|	-|	78.11|	78.98|	78.58|
|VeSPa|77.70	|67.35	|79.51|	79.67	|79.59|
|Ours|75.70	|**68.73**	|**81.74**	|79.31	|**80.51**|
