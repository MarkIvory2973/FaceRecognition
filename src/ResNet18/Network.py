from . import *

class ResNet18(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.resnet18 = models.resnet18(num_classes=num_classes)
        
    def forward(self, x):
        y = self.resnet18(x)
        
        return y