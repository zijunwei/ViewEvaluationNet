import torch.nn as nn

class SiameseNet(nn.Module):
    def __init__(self, single_pass_net):
        super(SiameseNet, self).__init__()
        self.single_pass_net = single_pass_net

    def forward(self, x1, x2):
        x1 = self.single_pass_net(x1)
        x2 = self.single_pass_net(x2)
        return x1, x2