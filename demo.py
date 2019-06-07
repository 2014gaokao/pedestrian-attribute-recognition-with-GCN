
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import pickle as pickle
from PIL import Image
import numpy as np
from models import *
import torch
from scipy.io import loadmat
from PIL import Image, ImageFont, ImageDraw

if __name__=='__main__':
    demo_image='image/demo_image.png'

    resize = (224, 224)
    test_transform=transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    with open('dataset/glove.pkl', 'rb') as f:
        inp = pickle.load(f)

    model = gcn_resnet101(num_classes=60, t=0.4, adj_file='dataset/adj.pkl')
    checkpoint=torch.load('checkpoint/model_best.pth.tar',map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    img=Image.open(demo_image)
    img_trans=test_transform(img)
    img_trans = torch.unsqueeze(img_trans, dim=0)
    inp=torch.from_numpy(inp)
    inp = torch.unsqueeze(inp, dim=0)
    score=model(img_trans,inp)
    score=score.detach().numpy()

    data = loadmat('dataset/RAP_annotation/RAP_annotation.mat')
    dataset = pickle.load(open('dataset/rap2_dataset.pkl', 'rb'))
    partition = pickle.load(open('dataset/rap2_partition.pkl', 'rb'))
    all = []
    for idx in range(152):
        all.append(data['RAP_annotation'][0][0][2][idx][0][0])
    select_name = []
    for i in range(len(dataset['selected_attribute'])):
        select_name.append(all[dataset['selected_attribute'][i]])
    for idx in range(len(select_name)):
        if(score[0,idx]>=0):
            print('%s: %.2f'%(select_name[idx],score[0,idx]))

    img = img.resize(size=(256, 512), resample=Image.BILINEAR)
    draw = ImageDraw.Draw(img)
    positive_cnt = 0
    for idx in range(len(select_name)):
        if score[0, idx] >= 0:
            txt = '%s: %.2f' % (select_name[idx], score[0, idx])
            draw.text((10, 10 + 20 * positive_cnt), txt, (255, 0, 0))
            positive_cnt += 1
    img.save('image/demo_image_result.png')
