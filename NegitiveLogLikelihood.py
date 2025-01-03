import torch
from torch import nn
import numpy as np

def make_surv_array(t,f,breaks=np.array([0,12.0,24.0,36.0,48.0,60.0,72.0,84.0,96.0,108.0,120.0,132.0])):
  """Transforms censored survival data into vector format that can be used in Keras.
    Arguments
        t: Array of failure/censoring times.
        f: Censoring indicator. 1 if failed, 0 if censored.
        breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
    Returns
        Two-dimensional array of survival data, dimensions are number of individuals X number of time intervals*2
  """
  t = np.array(t.detach().cpu())
  f = np.array(f.detach().cpu())
  n_samples=t.shape[0]
  n_intervals=len(breaks)-1
  timegap = breaks[1:] - breaks[:-1]
  breaks_midpoint = breaks[:-1] + 0.5*timegap
  y_train = np.zeros((n_samples,n_intervals*2))
  for i in range(n_samples):
    if f[i]: #if failed (not censored)
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks[:-1]) #give credit for surviving each time interval where failure time >= upper limit
      if t[i]<breaks[-1]: #if failure time is greater than end of last time interval, no time interval will have failure marked
        y_train[i,n_intervals+np.where(t[i]<breaks[:-1])[0][0]]=1 #mark failure at first bin where survival time < upper break-point
    else: #if censored
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks_midpoint) #if censored and lived more than half-way through interval, give credit for surviving the interval.
  return y_train

class NegativeLogLikelihood(nn.Module):
    def __init__(self,device):
        super(NegativeLogLikelihood, self).__init__()
        self.device = device

    def forward(self,y_pred,time,indicator):
        n_intervals = 11
        y_true = make_surv_array(time,indicator)
        y_true = torch.from_numpy(y_true).to(self.device)
        cens_uncens = 1. + y_true[:, 0:n_intervals] * (y_pred - 1.)  # component for all individuals
        uncens = 1. - y_true[:, n_intervals:2 * n_intervals] * y_pred  # component for only uncensored individuals
        loss = torch.sum(-torch.log(torch.clamp(torch.concat((cens_uncens, uncens), dim=-1), 1e-10, None)), axis=-1)
        loss = torch.mean(loss)
        return loss
if __name__ == '__main__':
    pass