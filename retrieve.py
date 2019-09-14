import pickle
import torchvision.transforms as transforms
from models import gcn_resnet101
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch

retrieve = ['Age17-30','BodyNormal','hs-BlackHair','ub-ColorWhite','lb-ColorBlue','shoes-ColorBlack','ub-ColorPurple']
#retrieve=['lb-ColorYellow','lb-Skirt']
#retrieve=['hs-BaldHead','attachment-ShoulderBag','lb-ColorBlack']

def retrieveContainAttribute(attribute_name):
    for idx in range(len(retrieve)):
        if retrieve[idx] == attribute_name:
            return True;
    return False;



if __name__ =='__main__':
    data = loadmat('dataset/RAP_annotation/RAP_annotation.mat')
    all = [] #所有的152个属性
    for idx in range(152):
        all.append(data['RAP_annotation'][0][0][2][idx][0][0])
    dataset = pickle.load(open('dataset/rap2_dataset.pkl', 'rb'))
    partition = pickle.load(open('dataset/rap2_partition.pkl', 'rb'))
    select_name = []
    for i in range(len(dataset['selected_attribute'])):
        select_name.append(all[dataset['selected_attribute'][i]])
    image = []
    label = []
    for idx in partition['test'][0]:
        image.append(dataset['image'][idx])
        label.append(np.array(dataset['att'][idx])[dataset['selected_attribute']])

    resize = (224, 224)
    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    model = gcn_resnet101(num_classes=60, t=0.9, adj_file='dataset/adj.pkl')
    checkpoint = torch.load('checkpoint/model_best_0.7624.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    for img_name in image:
        with open('dataset/glove.pkl', 'rb') as f:
            inp = pickle.load(f)
        img = Image.open('dataset/RAP_dataset/'+img_name)
        img_trans = test_transform(img)
        img_trans = torch.unsqueeze(img_trans, dim=0)
        inp = torch.from_numpy(inp)
        inp = torch.unsqueeze(inp, dim=0)
        score = model(img_trans, inp)
        score = score.detach().numpy()

        num=0;

        for idx in range(len(select_name)):
            if (score[0, idx] >= 0 and retrieveContainAttribute(select_name[idx])):
                num+=1;

        if num == len(retrieve):
            img.save('retrieve/'+img_name)
