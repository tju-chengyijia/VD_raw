"""
This code is based on the TTSR
https://github.com/researchmm/TTSR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .swin_layer import ConvModBlock


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,block_num, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        
        self.block_num = block_num

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


class SFE(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(3, n_feats)

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                     res_scale=res_scale))

        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class CSFI2(nn.Module):
    def __init__(self, n_feats):
        super(CSFI2, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv21 = conv3x3(n_feats, n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats * 2, n_feats)
        self.conv_merge2 = conv3x3(n_feats * 2, n_feats)

    def forward(self, x1, x2):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x21 = F.relu(self.conv21(x2))

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12), dim=1)))

        return x1, x2


class CSFI3(nn.Module):
    def __init__(self, n_feats):
        super(CSFI3, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv13 = conv1x1(n_feats, n_feats)

        self.conv21 = conv3x3(n_feats, n_feats, 2)
        self.conv23 = conv1x1(n_feats, n_feats)

        self.conv31_1 = conv3x3(n_feats, n_feats, 2)
        self.conv31_2 = conv3x3(n_feats, n_feats, 2)
        self.conv32 = conv3x3(n_feats, n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats * 3, n_feats)
        self.conv_merge2 = conv3x3(n_feats * 3, n_feats)
        self.conv_merge3 = conv3x3(n_feats * 3, n_feats)

    def forward(self, x1, x2, x3):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))

        x21 = F.relu(self.conv21(x2))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x31 = F.relu(self.conv31_1(x3))
        x31 = F.relu(self.conv31_2(x31))
        x32 = F.relu(self.conv32(x3))

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21, x31), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12, x32), dim=1)))
        x3 = F.relu(self.conv_merge3(torch.cat((x3, x13, x23), dim=1)))

        return x1, x2, x3


class MergeTail3(nn.Module):
    def __init__(self, n_feats, out_channel=3):
        super(MergeTail3, self).__init__()
        self.conv13 = conv1x1(n_feats, n_feats)
        self.conv23 = conv1x1(n_feats, n_feats)
        self.conv_merge = conv3x3(n_feats * 3, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats // 2)
        self.conv_tail2 = conv1x1(n_feats // 2, out_channel)

    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge(torch.cat((x3, x13, x23), dim=1)))
        x = self.conv_tail1(x)
        x = self.conv_tail2(x)
        # x = torch.clamp(x, 0, 1)

        return x


class MergeTail2(nn.Module):
    def __init__(self, n_feats, out_channel=3):
        super(MergeTail2, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv_merge = conv3x3(n_feats * 2, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats // 2)
        self.conv_tail2 = conv1x1(n_feats // 2, out_channel)

    def forward(self, x1, x2):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))

        x = F.relu(self.conv_merge(torch.cat((x2, x12), dim=1)))
        x = self.conv_tail1(x)
        x = self.conv_tail2(x)
        # x = torch.clamp(x, 0, 1)

        return x


class MergeTail1(nn.Module):
    def __init__(self, n_feats, out_channel=3):
        super(MergeTail1, self).__init__()
        self.conv_tail1 = conv3x3(n_feats, n_feats // 2)
        self.conv_tail2 = conv1x1(n_feats // 2, out_channel)

    def forward(self, x1):
        x = x1
        x = self.conv_tail1(x)
        x = self.conv_tail2(x)
        # x = torch.clamp(x, 0, 1)

        return x


class MainNet_V1(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale, use_shuffle=False):
        super(MainNet_V1, self).__init__()
        self.num_res_blocks = num_res_blocks  # a list containing number of res-blocks of different stages
        self.n_feats = n_feats
        self.use_shuffle = use_shuffle

        ### stage11
        self.RB11 = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):#16
            self.RB11.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale,block_num=i))
        self.conv11_tail = conv3x3(n_feats, n_feats)

        # ### subpixel 1 -> 2
        # self.conv12 = conv3x3(n_feats, n_feats*4)
        # self.ps12 = nn.PixelShuffle(2)

        ### stage21, 22
        self.ex12 = CSFI2(n_feats)

        self.RB21 = nn.ModuleList()
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):#8
            self.RB21.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale,block_num=i))
            self.RB22.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale,block_num=i))

        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)

        # ### subpixel 2 -> 3
        # self.conv23 = conv3x3(n_feats, n_feats*4)
        # self.ps23 = nn.PixelShuffle(2)

        ### stage31, 32, 33
        self.ex123 = CSFI3(n_feats)

        self.RB31 = nn.ModuleList()
        self.RB32 = nn.ModuleList()
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):#4
            self.RB31.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale,block_num=i))
            self.RB32.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale,block_num=i))
            self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale,block_num=i))

        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)

        if use_shuffle:
            out_channel = 12
        else:
            out_channel = 3

        self.merge_tail1 = MergeTail1(n_feats, out_channel)
        self.merge_tail2 = MergeTail2(n_feats, out_channel)
        self.merge_tail3 = MergeTail3(n_feats, out_channel)

    def forward(self, merge_lv3=None, merge_lv2=None, merge_lv1=None):

        ### stage11
        x11 = merge_lv3
        x11_res = x11

        for i in range(self.num_res_blocks[1]):#16
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res

        ### stage21, 22
        x21 = x11
        x21_res = x21
        # x22 = self.conv12(x11)
        # x22 = F.relu(self.ps12(x22))
        x22 = merge_lv2
        x22_res = x22

        x21_res, x22_res = self.ex12(x21_res, x22_res)

        for i in range(self.num_res_blocks[2]):
            x21_res = self.RB21[i](x21_res)
            x22_res = self.RB22[i](x22_res)

        x21_res = self.conv21_tail(x21_res)
        x22_res = self.conv22_tail(x22_res)
        x21 = x21 + x21_res
        x22 = x22 + x22_res

        ### stage31, 32, 33
        x31 = x21
        x31_res = x31
        x32 = x22
        x32_res = x32
        # x33 = self.conv23(x22)
        # x33 = F.relu(self.ps23(x33))
        x33 = merge_lv1
        x33_res = x33

        x31_res, x32_res, x33_res = self.ex123(x31_res, x32_res, x33_res)

        for i in range(self.num_res_blocks[3]):
            x31_res = self.RB31[i](x31_res)
            x32_res = self.RB32[i](x32_res)
            x33_res = self.RB33[i](x33_res)

        x31_res = self.conv31_tail(x31_res)
        x32_res = self.conv32_tail(x32_res)
        x33_res = self.conv33_tail(x33_res)
        x31 = x31 + x31_res
        x32 = x32 + x32_res
        x33 = x33 + x33_res

        x3 = self.merge_tail3(x31, x32, x33)
        x2 = self.merge_tail2(x31, x32)
        x1 = self.merge_tail1(x31)

        if not self.use_shuffle:
            # import pdb; pdb.set_trace()
            return x1, x2, x3, x1, x2, x3

        z3 = nn.functional.pixel_shuffle(x3, 2)
        z2 = nn.functional.pixel_shuffle(x2, 2)
        z1 = nn.functional.pixel_shuffle(x1, 2)
        return z1, z2, z3, x31, x32, x33


# depth fusion
class ResBlock_depth(nn.Module):
    def __init__(self, in_channels, out_channels,block_num, stride=1, groups=4, downsample=None, res_scale=1):
        super(ResBlock_depth, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1, groups=groups, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=stride, padding=1, groups=groups, bias=True)
        
        self.block_num = block_num

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out
    
class Branch_depth(nn.Module):
    def __init__(self, num_res_blocks, n_feats, groups, res_scale):
        super(Branch_depth, self).__init__()
        self.num_res_blocks = num_res_blocks  # a list containing number of res-blocks of different stages
        self.n_feats = n_feats
        ### stage1
        self.RB1 = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):#16
            self.RB1.append(ResBlock_depth(in_channels=n_feats, out_channels=n_feats, groups=groups,
                                      res_scale=res_scale,block_num=i))
        self.weight1 = nn.Parameter(torch.randn(1, n_feats, 1, 1))
        self.up1 = conv1x1(n_feats, n_feats)
        ### stage2
        self.RB2 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):#8
            self.RB2.append(ResBlock_depth(in_channels=n_feats, out_channels=n_feats, groups=groups,
                                      res_scale=res_scale,block_num=i))
        self.weight2 = nn.Parameter(torch.randn(1, n_feats, 1, 1))
        self.up2 = conv1x1(n_feats, n_feats)
        ### stage3
        self.RB3 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):#4
            self.RB3.append(ResBlock_depth(in_channels=n_feats, out_channels=n_feats, groups=groups,
                                      res_scale=res_scale,block_num=i))
        self.weight3 = nn.Parameter(torch.randn(1, n_feats, 1, 1))
    
    def forward(self, merge_rc=None):
        ### stage1
        x1_cur = merge_rc
        for i in range(self.num_res_blocks[1]):#16
            x1_cur = self.RB1[i](x1_cur)
        
        ### stage2
        x2_cur = self.up1(F.interpolate(x1_cur, scale_factor=2, mode='bilinear'))
        for i in range(self.num_res_blocks[2]):
            x2_cur = self.RB2[i](x2_cur)
            
        ### stage3
        x3_cur = self.up2(F.interpolate(x2_cur, scale_factor=2, mode='bilinear'))
        for i in range(self.num_res_blocks[3]):
            x3_cur = self.RB3[i](x3_cur)
            
        save_dict = {
                'feat1': x1_cur,
                'weight1': self.weight1,
                'feat1_multi_weight1': x1_cur*self.weight1,
                'feat2': x2_cur,
                'weight2': self.weight2,
                'feat2_multi_weight2': x2_cur*self.weight2,
                'feat3': x3_cur,
                'weight3': self.weight3,
                'feat3_multi_weight3': x3_cur*self.weight3,
                }

        return x1_cur*self.weight1, x2_cur*self.weight2, x3_cur*self.weight3, save_dict
    
class MainNet_depth(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale, use_shuffle=False):
        super(MainNet_depth, self).__init__()
        self.num_res_blocks = num_res_blocks  # a list containing number of res-blocks of different stages
        self.n_feats = n_feats
        self.use_shuffle = use_shuffle

        ### branch
        self.branch_depth = Branch_depth(self.num_res_blocks, self.n_feats, 4, res_scale)
        self.convmod1 = ConvModBlock(self.n_feats)
        self.convmod2 = ConvModBlock(self.n_feats)
        self.convmod3 = ConvModBlock(self.n_feats)
        
        ### stage11
        self.RB11 = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):#16
            self.RB11.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale,block_num=i))
        self.conv11_tail = conv3x3(n_feats, n_feats)

        # ### subpixel 1 -> 2
        # self.conv12 = conv3x3(n_feats, n_feats*4)
        # self.ps12 = nn.PixelShuffle(2)

        ### stage21, 22
        self.ex12 = CSFI2(n_feats)

        self.RB21 = nn.ModuleList()
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):#8
            self.RB21.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale,block_num=i))
            self.RB22.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale,block_num=i))

        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)

        # ### subpixel 2 -> 3
        # self.conv23 = conv3x3(n_feats, n_feats*4)
        # self.ps23 = nn.PixelShuffle(2)

        ### stage31, 32, 33
        self.ex123 = CSFI3(n_feats)

        self.RB31 = nn.ModuleList()
        self.RB32 = nn.ModuleList()
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):#4
            self.RB31.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale,block_num=i))
            self.RB32.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale,block_num=i))
            self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale,block_num=i))

        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)

        if use_shuffle:
            out_channel = 12
        else:
            out_channel = 3

        self.merge_tail1 = MergeTail1(n_feats, out_channel)
        self.merge_tail2 = MergeTail2(n_feats, out_channel)
        self.merge_tail3 = MergeTail3(n_feats, out_channel)

    def forward(self, merge_lv3=None, merge_lv2=None, merge_lv1=None, merge_rc=None):

        ### branch
        branch_1, branch_2, branch_3, save_dict = self.branch_depth(merge_rc)
        
        ### stage11
        x11 = merge_lv3
        x11_res = x11

        for i in range(self.num_res_blocks[1]):#16
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res
        
        ### branch modulate
        x11 = self.convmod1(x11+branch_1)

        ### stage21, 22
        x21 = x11
        x21_res = x21
        # x22 = self.conv12(x11)
        # x22 = F.relu(self.ps12(x22))
        x22 = merge_lv2
        x22_res = x22

        x21_res, x22_res = self.ex12(x21_res, x22_res)

        for i in range(self.num_res_blocks[2]):
            x21_res = self.RB21[i](x21_res)
            x22_res = self.RB22[i](x22_res)

        x21_res = self.conv21_tail(x21_res)
        x22_res = self.conv22_tail(x22_res)
        x21 = x21 + x21_res
        x22 = x22 + x22_res
        
        ### branch modulate
        x22 = self.convmod2(x22+branch_2)

        ### stage31, 32, 33
        x31 = x21
        x31_res = x31
        x32 = x22
        x32_res = x32
        # x33 = self.conv23(x22)
        # x33 = F.relu(self.ps23(x33))
        x33 = merge_lv1
        x33_res = x33

        x31_res, x32_res, x33_res = self.ex123(x31_res, x32_res, x33_res)

        for i in range(self.num_res_blocks[3]):
            x31_res = self.RB31[i](x31_res)
            x32_res = self.RB32[i](x32_res)
            x33_res = self.RB33[i](x33_res)

        x31_res = self.conv31_tail(x31_res)
        x32_res = self.conv32_tail(x32_res)
        x33_res = self.conv33_tail(x33_res)
        x31 = x31 + x31_res
        x32 = x32 + x32_res
        x33 = x33 + x33_res
        
        ### branch modulate
        x33 = self.convmod3(x33+branch_3)

        x3 = self.merge_tail3(x31, x32, x33)
        x2 = self.merge_tail2(x31, x32)
        x1 = self.merge_tail1(x31)

        if not self.use_shuffle:
            # import pdb; pdb.set_trace()
            return x1, x2, x3, x1, x2, x3

        z3 = nn.functional.pixel_shuffle(x3, 2)
        z2 = nn.functional.pixel_shuffle(x2, 2)
        z1 = nn.functional.pixel_shuffle(x1, 2)
        return z1, z2, z3, x31, x32, x33, save_dict
    
