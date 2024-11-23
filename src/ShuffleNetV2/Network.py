from . import *

class ShuffleNetV2(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.shufflenet_v2_x2_0 = models.shufflenet_v2_x2_0(num_classes=num_classes)
        
    def forward(self, x):
        y = self.shufflenet_v2_x2_0(x)
        
        return y