import torch.nn as nn
from torchvision.models.vgg import vgg16
import torch
import torch.nn.functional as F
def ranker_loss(model, img):
    # with torch.no_grad():
    score = model(img)['final_result']
    loss = torch.mean(F.sigmoid(-score))

    return loss

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        # vgg = vgg16(pretrained=True).cuda()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        h1 = self.to_relu_1_2(x1)
        h1 = self.to_relu_2_2(h1)
        h1 = self.to_relu_3_3(h1)
        h1 = self.to_relu_4_3(h1)

        h2 = self.to_relu_1_2(x2)
        h2 = self.to_relu_2_2(h2)
        h2 = self.to_relu_3_3(h2)
        h2 = self.to_relu_4_3(h2)

        return torch.mean(torch.abs(h1 - h2))