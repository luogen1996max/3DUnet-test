"""
This code is referenced from https://github.com/assassint2017/MICCAI-LITS2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class ResUNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2 ,training=True):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(in_channel, 2, 3, 1, padding=1),
            nn.PReLU(2),

            # nn.Conv3d(16, 16, 3, 1, padding=1),
            # nn.PReLU(16),
        )

        # self.encoder_stage2 = nn.Sequential(
        #     # nn.Conv3d(32, 32, 3, 1, padding=1),
        #     # nn.PReLU(32),
        #     #
        #     # nn.Conv3d(32, 32, 3, 1, padding=1),
        #     # nn.PReLU(32),
        #
        #     nn.Conv3d(32, 32, 3, 1, padding=1),
        #     nn.PReLU(32),
        # )
        #
        # self.encoder_stage3 = nn.Sequential(
        #     # nn.Conv3d(64, 64, 3, 1, padding=1),
        #     # nn.PReLU(64),
        #     #
        #     # nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
        #     # nn.PReLU(64),
        #
        #     nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
        #     nn.PReLU(64),
        # )
        #
        # self.encoder_stage4 = nn.Sequential(
        #     # nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
        #     # nn.PReLU(128),
        #     #
        #     # nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
        #     # nn.PReLU(128),
        #
        #     nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
        #     nn.PReLU(128),
        # )

        # self.decoder_stage1 = nn.Sequential(
        #     nn.Conv3d(128, 256, 3, 1, padding=1),
        #     nn.PReLU(256),
        #
        #     # nn.Conv3d(256, 256, 3, 1, padding=1),
        #     # nn.PReLU(256),
        #     #
        #     # nn.Conv3d(256, 256, 3, 1, padding=1),
        #     # nn.PReLU(256),
        # )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(2 + 8, 12, 3, 1, padding=1),
            nn.PReLU(12),

            # nn.Conv3d(128, 128, 3, 1, padding=1),
            # nn.PReLU(128),
            #
            # nn.Conv3d(128, 128, 3, 1, padding=1),
            # nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(4 + 4, 12, 3, 1, padding=1),
            nn.PReLU(12),

            # nn.Conv3d(64, 64, 3, 1, padding=1),
            # nn.PReLU(64),
            #
            # nn.Conv3d(64, 64, 3, 1, padding=1),
            # nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(4 + 2, 12, 3, 1, padding=1),
            nn.PReLU(12),

            # nn.Conv3d(32, 32, 3, 1, padding=1),
            # nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(2, 4, 2, 2),
            nn.PReLU(4)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(4, 6, 2, 2),
            nn.PReLU(6)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(6, 8, 2, 2),
            nn.PReLU(8)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(8, 10, 3, 1, padding=1),
            nn.PReLU(10)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(10, 4, 2, 2),
            nn.PReLU(4)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(12, 4, 2, 2),
            nn.PReLU(4)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(12, 4, 2, 2),
            nn.PReLU(4)
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(12, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(12, out_channel, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(12, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False),

            nn.Softmax(dim=1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(10, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        # inputs        1, 1, 48, 256, 256
        long_range1 = self.encoder_stage1(inputs) + inputs      ## 1, 2, 48, 256, 256

        # short_range1
        short_range1 = self.down_conv1(long_range1)     ## 1, 4, 24, 128, 128

        short_range2 = self.down_conv2(short_range1)        ## 1, 6, 12, 64, 64

        short_range3 = self.down_conv3(short_range2)        ## 1, 8, 12, 64, 64

        outputs = self.down_conv4(short_range3)     ## 1, 10, 6, 32, 32
        output1 = self.map1(outputs)



        ## outputs          1, 256, 6, 32, 32
        short_range6 = self.up_conv2(outputs)       ## 1, 4, 12, 64, 64

        outputs = self.decoder_stage2(torch.cat([short_range6, short_range2], dim=1))##1, 12, 12, 64, 64  # + short_range6
        outputs = F.dropout(outputs, self.dorp_rate, self.training)


        output2 = self.map2(outputs)        ## 1, 2, 48, 256, 256

        short_range7 = self.up_conv3(outputs)   ## 1, 4, 24, 128, 128

        outputs = self.decoder_stage3(torch.cat([short_range7, short_range1], dim=1))##1, 12, 24, 128, 128    # + short_range7
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        output3 = self.map3(outputs)        ## 1, 2, 48, 256, 256

        short_range8 = self.up_conv4(outputs)   ## 1, 4, 48, 256, 256

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1))##1, 12, 48, 256, 256 # + short_range8

        output4 = self.map4(outputs)        ## 1, 2, 48, 256, 256
        # print('+++++++++++++++++++++++++++++++++++++')
        # print(output4.shape)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@22')

        if self.training is True:
            # print('############################3')
            return output1, output2, output3, output4
        else:
            return output4
