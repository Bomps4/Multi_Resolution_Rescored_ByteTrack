from torch import nn


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion,self).__init__()
        self.log_once=0
    def forward(self,x):
        if self.log_once==0:
            print('empty criterion inserted for interface consistency')
            self.log_once+=1