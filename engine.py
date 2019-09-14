import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
#import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from util import *
from evaluate import *
import numpy as np

tqdm.monitor_interval = 0
class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        self.state['pt_result']=[]
        self.state['gt_result']=[]
        # meters
        self.state['loss_meter']=AverageMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = False
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['pt_result']=[]
        self.state['gt_result']=[]
        self.state['loss_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss=self.state['loss_meter'].value()
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss))
            else:
                print('Test: \t Loss {loss:.4f}'.format(loss=loss))
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        self.state['loss_meter'].update(self.state['loss'].data.cpu().numpy())

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):

        input_var = torch.autograd.Variable(self.state['input'])
        target_var = torch.autograd.Variable(self.state['target'])

        if not training:
            input_var.volatile = True
            target_var.volatile = True

        # compute output
        self.state['output'] = model(input_var)
        self.state['loss'] = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    def init_learning(self, model, criterion):

        if self._state('train_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['train_transform'] = transforms.Compose([
                transforms.Resize((224,224)),#transforms.Resize((512, 512)),
                MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            print(self.state['train_transform'])

        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):

        self.init_learning(model, criterion)

        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['workers'])

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'],map_location='cpu')
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))


        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True


            model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()


            criterion = criterion.cuda()

        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            return

        # TODO define optimizer

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:{:.5f}'.format(lr))

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            prec1 = self.validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
            }, is_best)

            print(' *** best={best:.4f}'.format(best=self.state['best_score']))
        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch):

        # switch to train mode
        model.train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(async=True)

            self.on_forward(True, model, criterion, data_loader, optimizer)

            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)

        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):

        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(async=True)

            self.on_forward(False, model, criterion, data_loader)

            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

        score = self.on_end_epoch(False, model, criterion, data_loader)

        return score

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 30))
        decay = 0.1 ** (sum(self.state['epoch'] >= np.array(self.state['epoch_step'])))
        lr = self.state['lr'] * decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)

        self.state['pt_result']=[]
        self.state['gt_result']=[]
        self.state['loss_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['pt_result']=np.array(self.state['pt_result'])
        self.state['gt_result']=np.array(self.state['gt_result'])
        result=attribute_evaluate(self.state['pt_result'],self.state['gt_result'])
        ma=np.mean(result['label_acc'])
        acc=result['instance_acc']
        prec=result['instance_precision']
        rec=result['instance_recall']
        f1=result['instance_F1']
        loss=self.state['loss_meter'].value()
        if display:
            if training:
                with open('/data/zcc/gcn/log.txt','a') as f:
                      f.write('Epoch: [{0}]\t'
                              'Loss {loss:.4f}\t'
                              'mA {ma:.4f}'.format(self.state['epoch'],loss=loss,ma=ma))
                      f.write('\r\n')
                      f.write('Acc {acc:.4f}\t'
                              'Prec {prec:.4f}\t'
                              'Rec {rec:.4f}\t'
                              'F1 {f1:.4f}'.format(acc=acc,prec=prec,rec=rec,f1=f1))
                      f.write('\r\n\r\n')
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'mA {ma:.4f}'.format(self.state['epoch'], loss=loss, ma=ma))
            else:
                with open('/data/zcc/gcn/test.txt','a') as f:
                      f.write('Test: \t Epoch: [{0}]\t Loss {loss:.4f}\t mA {ma:.4f}'.format(self.state['epoch'],loss=loss,ma=ma))
                      f.write('\r\n')
                      f.write('Acc {acc:.4f}\t'
                              'Prec {prec:.4f}\t'
                              'Rec {rec:.4f}\t'
                              'F1 {f1:.4f}'.format(acc=acc,prec=prec,rec=rec,f1=f1))
                      f.write('\r\n\r\n')
                print('Test: \t Epoch: [{0}]\t Loss {loss:.4f}\t mA {ma:.4f}'.format(self.state['epoch'],loss=loss, ma=ma))

        return ma

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target']==0]=0
        self.state['target'][self.state['target']==1]=1

        input = self.state['input']
        self.state['input'] = input[0]
        self.state['name'] = input[1]

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)

        # measure mAP
        for i in range(self.state['output'].data.shape[0]):
            feat=self.state['output'].data[i,:].cpu().numpy()
            self.state['pt_result'].append(feat)
        for i in range(self.state['target_gt'].shape[0]):
            label=self.state['target_gt'][i,:].cpu().numpy()
            self.state['gt_result'].append(label)

class GCNMultiLabelMAPEngine(MultiLabelMAPEngine):
    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        feature_var = torch.autograd.Variable(self.state['feature']).float()
        target_var = torch.autograd.Variable(self.state['target']).float()
        inp_var = torch.autograd.Variable(self.state['input']).float().detach()  # one hot
        if not training:
            feature_var.volatile = True
            target_var.volatile = True
            inp_var.volatile = True

        # compute output
        self.state['output'] = model(feature_var, inp_var)
        self.state['loss'] = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)              ####userwarning最后加了个小横杠
            optimizer.step()


    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target']==1]=1
        self.state['target'][self.state['target']==0]=0

        input = self.state['input']
        self.state['feature'] = input[0]
        self.state['out'] = input[1]
        self.state['input'] = input[2]

