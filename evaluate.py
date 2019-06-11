import numpy as np

def attribute_evaluate(pt_result,gt_result):
    pt_result[pt_result >= 0] = 1
    pt_result[pt_result < 0] = 0
    result = {}
    gt_pos = np.sum((gt_result == 1).astype(float), axis=0)
    gt_neg = np.sum((gt_result == 0).astype(float), axis=0)
    pt_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=0)
    pt_neg = np.sum((gt_result == 0).astype(float) * (pt_result == 0).astype(float), axis=0)
    label_pos_acc = 1.0 * pt_pos / gt_pos
    label_neg_acc = 1.0 * pt_neg / gt_neg
    label_acc = (label_pos_acc + label_neg_acc) / 2
    result['label_pos_acc'] = label_pos_acc
    result['label_neg_acc'] = label_neg_acc
    result['label_acc'] = label_acc

    gt_pos=np.sum((gt_result==1).astype(float),axis=1)
    pt_pos=np.sum((pt_result==1).astype(float),axis=1)
    floatersect_pos=np.sum((gt_result==1).astype(float)*(pt_result==1).astype(float),axis=1)
    union_pos=np.sum(((gt_result==1)+(pt_result==1)).astype(float),axis=1)
    cnt_eff=float(gt_result.shape[0])
    for iter,key in enumerate(gt_pos):
        if key==0:
            union_pos[iter]=1
            pt_pos[iter]=1
            gt_pos[iter]=1
            cnt_eff=cnt_eff-1
            continue
        if pt_pos[iter]==0:
            pt_pos[iter]=1
    instance_acc=np.sum(floatersect_pos/union_pos)/cnt_eff
    instance_precision=np.sum(floatersect_pos/pt_pos)/cnt_eff
    instance_recall=np.sum(floatersect_pos/gt_pos)/cnt_eff
    floatance_F1=2*instance_precision*instance_recall/(instance_precision+instance_recall)
    result['instance_acc']=instance_acc
    result['instance_precision']=instance_precision
    result['instance_recall']=instance_recall
    result['instance_F1']=floatance_F1
    return result
class AverageMeter(object):
    def __init__(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=float(self.sum)/(self.count+1e-10)
    def value(self):
        return self.avg
