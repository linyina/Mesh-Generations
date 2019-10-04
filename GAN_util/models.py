
import torch
import torch.nn as nn
import math

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        if self.opt.LeakyReLu:
            ReLu = nn.LeakyReLU(self.opt.leak_value)
        else:
            ReLu = nn.ReLU(True)

        # Conv3d
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1, bias=self.opt.bias),
            nn.BatchNorm3d(64),
            ReLu
        )
        # Conv3d
        self.layer2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=self.opt.bias),
            nn.BatchNorm3d(128),
            ReLu
        )
        # Conv3d
        self.layer3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1, bias=self.opt.bias),
            nn.BatchNorm3d(256),
            ReLu
        )
        # Conv3d
        self.layer4 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=self.opt.bias),
            nn.BatchNorm3d(256),
            ReLu
        )

        # Dilated conv3d
        self.layer5 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4, bias=self.opt.bias),
            nn.BatchNorm3d(256),
            ReLu
        )

        # Dilated conv3d
        self.layer6 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8, bias=self.opt.bias),
            nn.BatchNorm3d(256),
            ReLu

        )

        # Conv3d
        self.layer7 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=self.opt.bias),
            nn.BatchNorm3d(256),
            ReLu
        )

        # Deconv3d
        self.layer8 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=self.opt.bias),
            nn.BatchNorm3d(128),
            ReLu
        )

        # Deconv3d
        self.layer9 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=self.opt.bias),
            nn.BatchNorm3d(64),
            ReLu
        )

        # Deconv3d
        self.layer10 = nn.Sequential(
            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1, bias=self.opt.bias),
            self.opt.G_output_activation
        )
    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        out = self.layer10(x)

        return out

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.opt = opt

        if self.opt.LeakyReLu:
            ReLu = nn.LeakyReLU(self.opt.leak_value)
        else:
            ReLu = nn.ReLU(True)

        # Conv3d
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1, bias=self.opt.bias),
            nn.BatchNorm3d(64),
            ReLu
        )

        # Conv3d
        self.layer2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=self.opt.bias),
            nn.BatchNorm3d(128),
            ReLu
        )

        # Conv3d
        self.layer3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1, bias=self.opt.bias),
            nn.BatchNorm3d(256),
            ReLu
        )

        # Conv3d
        self.layer4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1, bias=self.opt.bias),
            nn.BatchNorm3d(512),
            ReLu
        )

        # Classifier
        self.layer5 = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        out = self.layer5(x)

        return out.view(-1, 1).squeeze(1)

def weights_init(m):
    for m in m.modules():
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
        elif isinstance(m, nn.ConvTranspose3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()