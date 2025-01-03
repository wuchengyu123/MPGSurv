import torch
from torch import nn
import numpy as np
class Regularization(object):
    def __init__(self, order, weight_decay):
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss += torch.norm(w, p=self.order)
        reg_loss *= self.weight_decay
        return reg_loss

class NegativeLogLikelihood(nn.Module):
    def __init__(self,l2_reg,device,smooth=1e-1):
        super(NegativeLogLikelihood, self).__init__()
        self.device = device
        self.l2_reg = Regularization(order=2, weight_decay=l2_reg)
        self.smooth = smooth

    def forward(self,predicted_event_times,event_times,event_observed,model):
        mask = torch.ones(event_times.shape[0],event_times.shape[0])  #创建mask
        mask[(event_times.T - event_times) > 0] = 0   # 将生存时间比自己短的人全部置为0  第i行第j列为1代表j的存活时长要长于i
        mask = mask.to(self.device)
        log_loss = torch.exp(predicted_event_times) * mask
        log_loss = torch.log(torch.sum(log_loss,dim=0) / torch.sum(mask,dim=0)).reshape(-1,1)
        neg_log_loss = -torch.sum((predicted_event_times - log_loss) * event_observed) / (torch.sum(event_observed)+self.smooth)
        l2_loss = self.l2_reg(model)
        return neg_log_loss+l2_loss

