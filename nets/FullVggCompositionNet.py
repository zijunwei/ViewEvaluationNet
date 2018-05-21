import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models



class FullVggCompositionNet(nn.Module):
    def __init__(self, pretrained=True, isFreeze=False, LinearSize1=1024, LinearSize2=512):


        super(FullVggCompositionNet, self).__init__()

        model = models.vgg16(pretrained=pretrained)
        self.features = model.features

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, LinearSize1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(LinearSize1, LinearSize2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(LinearSize2, 1),
        )

        if isFreeze:
            for param in self.features.parameters():
                if isFreeze:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_name(self):
        return self.__class__.__name__


if __name__ == '__main__':
    test_input = torch.randn([1, 3, 224, 224])
    compNet = FullVggCompositionNet()
    test_input = Variable(test_input)
    output = compNet(test_input)
    print "DEBUG"




