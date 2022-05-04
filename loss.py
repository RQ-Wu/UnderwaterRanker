import torch.nn as nn
from torchvision.models.vgg import vgg16, vgg19
from torchvision import transforms
import torch
import torch.nn.functional as F
import utils

def rank_loss(x1, x2):
    rank_loss = nn.MarginRankingLoss(margin=0.5).cuda()
    x1 = torch.clamp(x1, min=-5, max=5)
    x2 = torch.clamp(x2, min=-5, max=5)
    L_rank = rank_loss(x1, x2, torch.zeros_like(x1).cuda()+1.0)
    
    return L_rank

def ranker_loss(model, img):
    # with torch.no_grad():
    pre_input = utils.preprocessing(img)
    score = model(**pre_input)['final_result']
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

class perception_loss_norm_vgg19(nn.Module):
    def __init__(self):
        super(perception_loss_norm_vgg19, self).__init__()
        # vgg = vgg16(pretrained=True).cuda()
        features = vgg19(pretrained=True).features
        self.to_relu_5_4 = features[:-1]
        self.requires_grad_(False)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    def forward(self, x1, x2):
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        h1 = self.to_relu_5_4(x1)
        h2 = self.to_relu_5_4(x2)

        return torch.mean(torch.abs(h1 - h2))

class perception_loss_norm(perception_loss):
    def __init__(self):
        super().__init__()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    def forward(self, x1, x2):
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        return super().forward(x1, x2)

def make_perception_loss(args):
    if args is None:
        return perception_loss()
    class_dict = {
        (True, 16): perception_loss_norm,
        (True, 19): perception_loss_norm_vgg19,
        (False, 16): perception_loss,
    }
    has_norm = args.get('norm', False)
    layers = args.get('layers', 16)
    return class_dict[(has_norm, layers)]()
