import torch
import torch.nn as nn


class _netG(nn.Module):
    def __init__(self, opt):
        super(_netG, self).__init__()
        # self.ngpu = opt.ngpu
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv1_stride = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_stride = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_stride = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.conv4_stride = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.conv5_stride = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.conv6_stride = nn.Conv2d(128, 128, 3, stride=2)
        self.deconv5_fs = nn.ConvTranspose2d(128, 128, 3, stride=2)
        self.deconv4_fs = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        self.deconv3_fs = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        self.deconv2_fs = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1)
        self.deconv1_fs = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1)
        self.recon = nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1)
        self.elu = nn.ELU()
    def forward(self, x):
        conv1_1 = self.elu(self.conv1_1(x))
        conv1_2 = self.elu(self.conv1_2(conv1_1))
        conv1_stride = self.conv1_stride(conv1_2)
        conv2_1 = self.elu(self.conv2_1(conv1_stride))
        conv2_2 = self.elu(self.conv2_2(conv2_1))
        conv2_stride = self.conv2_stride(conv2_2)
        conv3_1 = self.elu(self.conv3_1(conv2_stride))
        conv3_2 = self.elu(self.conv3_2(conv3_1))
        conv3_3 = self.elu(self.conv3_3(conv3_2))
        conv3_4 = self.elu(self.conv3_4(conv3_3))
        conv3_stride = self.conv3_stride(conv3_4)
        conv4_stride = self.elu(self.conv4_stride(conv3_stride))
        conv5_stride = self.elu(self.conv5_stride(conv4_stride))
        conv6_stride = self.elu(self.conv6_stride(conv5_stride))
        debn5_fs = self.elu(self.deconv5_fs(conv6_stride, output_size=conv5_stride.size()))
        skip5 = torch.cat((debn5_fs, conv5_stride), 1)
        debn4_fs = self.elu(self.deconv4_fs(skip5, output_size=conv4_stride.size()))
        skip4 = torch.cat((debn4_fs, conv4_stride), 1)
        debn3_fs = self.elu(self.deconv3_fs(skip4, output_size=conv3_stride.size()))
        skip3 = torch.cat((debn3_fs, conv3_stride), 1)
        debn2_fs = self.elu(self.deconv2_fs(skip3, output_size=conv2_stride.size()))
        skip2 = torch.cat((debn2_fs, conv2_stride), 1)
        debn1_fs = self.elu(self.deconv1_fs(skip2, output_size=conv1_stride.size()))
        skip1 = torch.cat((debn1_fs, conv1_stride), 1)
        recon = self.recon(skip1, output_size=x.size())
        return recon
        


class _netlocalD(nn.Module):
    def __init__(self):
        super(_netlocalD, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, input):
        output = self.layer1(input)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output.view(-1)
