import argparse
from engine import *
from models import *
from gcn_dataset import *
import pickle as pickle

parser = argparse.ArgumentParser(description='WILDCAT Training')
#parser.add_argument('data', metavar='DIR',
                    #help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 代码16/命令行32/256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='checkpoint/voc/voc_checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    args.data='data/gcn'
    args.evaluate=False
    args.resume=''

    use_gpu = torch.cuda.is_available()

    # define dataset
    #train_dataset = Voc2007Classification(args.data, 'trainval', inp_name='data/voc/voc_glove_word2vec.pkl')
    #val_dataset = Voc2007Classification(args.data, 'test', inp_name='data/voc/voc_glove_word2vec.pkl')
    train_dataset=GCNDataset(args.data,'trainval',inp_name='dataset/glove.pkl')
    val_dataset = GCNDataset(args.data, 'test', inp_name='dataset/glove.pkl')


    num_classes =train_dataset.num_att()

    # load model
    model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='dataset/adj.pkl')

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)



if __name__ == '__main__':
    main()

'''import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from Model import Model
import math
from att import *
if __name__=='__main__':
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resize = (224, 224)
    train_transform = transforms.Compose([transforms.Resize(resize),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),])
    dataset='dataset/rap2_dataset.pkl'
    partition='dataset/rap2_partition.pkl'

    train_dataset=AttDataset(dataset=dataset,partition=partition,split='train',partition_idx=0,
                             transform=train_transform,inp_name='dataset/glove.pkl')
    train_dataloader=DataLoader(dataset=train_dataset,batch_size=2,shuffle=True,num_workers=2)

    num_att=len(train_dataset.dataset['selected_attribute'])
    #kwargs=dict()
    #kwargs['num_att']=num_att

    model=Model(num_att=num_att,t=0.4,adj_file='dataset/adj.pkl')
    if torch.cuda.device_count()>1:
        model=nn.DataParallel(model)
    model.to(device)

    split='train'
    rate=np.array(train_dataset.partition['weight_'+split][0])
    rate=rate[train_dataset.dataset['selected_attribute']].tolist()
    if len(rate)!=num_att:
        print('选择的属性长度不匹配')
    #criterion = F.binary_cross_entropy_with_logits
    criterion=nn.MultiLabelSoftMarginLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0005)

    for epoch in range(1):
        for step,(img,label) in enumerate(train_dataloader):
            predict=model(img)
            label[label==-1]=0
            #loss=criterion(predict,label,weight=weights)*num_att
            loss = criterion(predict, label) * num_att
            optimizer.zero_grad()
            loss.backward()
            print(loss)
            optimizer.step()

    if (epoch+1) %50==0 or epoch+1==150:
        torch.save(model.state_dict(),'dataset/ckpt_epoch%d.pth'%(epoch+1))'''

