import torch
import torch.nn as nn
from torchvision import models
from utility import ssim


class L_VGG(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """

    def __init__(self, _pretrained_=True):
        super(L_VGG, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None:
            layers = {'30': 'conv5_2'}  # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer] - pred_f[layer]) ** 2)


def get_loss():
    MSE_Loss = nn.MSELoss()
    SSIM_Loss = ssim.SSIM()
    # VGG_Loss = L_VGG()

    # 将损失函数移至GPU
    MSE_Loss.cuda()
    SSIM_Loss.cuda()
    # VGG_Loss.cuda()
    return MSE_Loss, SSIM_Loss
