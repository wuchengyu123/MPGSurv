import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import model.Linformer as LF


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class CBAMConv(nn.Module):
    def __init__(self,channel,out_channel,kernel_size,stride,reduction=16,spatial_kernel=7):
        super().__init__()

        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channel,channel // reduction,1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction,channel,1,bias=False)
        )
        self.spatital = nn.Conv3d(2,1,kernel_size=spatial_kernel,padding=spatial_kernel // 2,bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Sequential(nn.Conv3d(channel,out_channel,kernel_size,padding=kernel_size // 2,stride=stride))

    def forward(self,x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out+avg_out)

        x = channel_out * x

        max_out,_ = torch.max(x,dim=1,keepdim=True)
        avg_out = torch.mean(x,dim=1,keepdim=True)
        spatial_out = self.sigmoid(self.spatital(torch.cat([max_out,avg_out],dim=1)))

        x = spatial_out * x

        return self.conv(x)

        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)#CBAMConv(in_planes,planes,3,stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = CBAMConv(planes,planes,3,stride)#conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Linformer_Layer(nn.Module):
    """
    A wrapper function to accept LM tasks, inspired by https://github.com/lucidrains/sinkhorn-transformer
    """
    def __init__(self, num_tokens, input_size, channels,
                       dim_k=64, dim_ff=1024, dim_d=None,
                       dropout_ff=0.1, dropout_tokens=0.1, nhead=4, depth=1, ff_intermediate=None,
                       dropout=0.05, activation="gelu", checkpoint_level="C0",
                       parameter_sharing="layerwise", k_reduce_by_layer=0, full_attention=False,
                       include_ff=True, w_o_intermediate_dim=None, emb_dim=None,
                       return_emb=False, decoder_mode=False, causal=False, method="learnable"):
        super(Linformer_Layer, self).__init__()
        emb_dim = channels if emb_dim is None else emb_dim

        self.input_size = input_size

        self.to_token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = LF.PositionalEmbedding(emb_dim)
        self.linformer = LF.Linformer(input_size, channels, dim_k=dim_k,
                                   dim_ff=dim_ff, dim_d=dim_d, dropout_ff=dropout_ff,
                                   nhead=nhead, depth=depth, dropout=dropout, ff_intermediate=ff_intermediate,
                                   activation=activation, checkpoint_level=checkpoint_level, parameter_sharing=parameter_sharing,
                                   k_reduce_by_layer=k_reduce_by_layer, full_attention=full_attention, include_ff=include_ff,
                                   w_o_intermediate_dim=w_o_intermediate_dim, decoder_mode=decoder_mode, causal=causal, method=method)

        if emb_dim != channels:
            self.linformer = LF.ProjectInOut(self.linformer, emb_dim, channels)

        self.dropout_tokens = nn.Dropout(dropout_tokens)

    def forward(self, tensor, **kwargs):
        """
        Input is (batch_size, seq_len), and all items are ints from [0, num_tokens-1]
        """
        # tensor = self.to_token_emb(tensor)
        tensor = self.pos_emb(tensor).type(tensor.type()) + tensor
        tensor = self.dropout_tokens(tensor)
        tensor = self.linformer(tensor, **kwargs)
        return tensor

class ResNet_Linformer(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 num_classes=256):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        print(self.layer1)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=(2, 2, 2))
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=(2, 2, 2))

        #self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(161792, num_classes)
        self.lf1 = Linformer_Layer(num_tokens=1, input_size=21*32*32+11*23*23+8*17*17+4*12*12, channels=256)
        self.lf2 = Linformer_Layer(num_tokens=1, input_size=11*16*16+6*12*12+4*9*9+2*6*6, channels=512)
        self.lf3 = Linformer_Layer(num_tokens=1, input_size=6*8*8+3*6*6+2*5*5+1*3*3, channels=1024)
        self.lf4 = Linformer_Layer(num_tokens=1, input_size=3*4*4+2*3*3+1*3*3+1*2*2, channels=2048)

    def linformer_layer(self, x1, x2, x3, x4, lf):
        b1, c1, d1, h1, w1 = x1.size()
        b2, c2, d2, h2, w2 = x2.size()
        b3, c3, d3, h3, w3 = x3.size()
        b4, c4, d4, h4, w4 = x4.size()

        x1 = x1.view(x1.size(0), x1.size(1), -1).transpose(1, 2) # 21504
        x2 = x2.view(x2.size(0), x2.size(1), -1).transpose(1, 2) # 5819
        x3 = x3.view(x3.size(0), x3.size(1), -1).transpose(1, 2) # 2312
        x4 = x4.view(x4.size(0), x4.size(1), -1).transpose(1, 2) # 576
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = lf(x).transpose(1, 2)
        x1 = x[:, :, 0:d1 * h1 * w1].view(b1, c1, d1, h1, w1)
        x2 = x[:, :, d1 * h1 * w1:d1 * h1 * w1 + d2 * h2 * w2].view(b2, c2, d2, h2, w2)
        x3 = x[:, :, d1 * h1 * w1 + d2 * h2 * w2:d1 * h1 * w1 + d2 * h2 * w2 + d3 * h3 * w3].view(b3, c3, d3, h3, w3)
        x4 = x[:, :, d1 * h1 * w1 + d2 * h2 * w2 + d3 * h3 * w3:d1 * h1 * w1 + d2 * h2 * w2 + d3 * h3 * w3 + d4 * h4 * w4].view(b4, c4, d4, h4, w4)
        return x1, x2, x3, x4, x


    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))
        print(downsample)
        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3, x4):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        if not self.no_max_pool:
            x1 = self.maxpool(x1)  # [3,64,32,32,32]
        #print('x1 shape',x1.shape)
        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        if not self.no_max_pool:
            x2 = self.maxpool(x2)
        #print('x2 shape', x2.shape)
        x3 = self.conv1(x3)
        x3 = self.bn1(x3)
        x3 = self.relu(x3)
        if not self.no_max_pool:
            x3 = self.maxpool(x3)
        #print('x3 shape', x1.shape)
        x4 = self.conv1(x4)
        x4 = self.bn1(x4)
        x4 = self.relu(x4)
        if not self.no_max_pool:
            x4 = self.maxpool(x4)
        #print('x4 shape', x1.shape)
        x1 = self.layer1(x1) # [3,64,32,32,32]
        #print('x1 lf1:',x1.shape)
        x2 = self.layer1(x2)
        #print('x2 lf1:', x2.shape)
        x3 = self.layer1(x3)
        #print('x3 lf1:', x3.shape)
        x4 = self.layer1(x4)
        #print('x4 lf1:', x4.shape)

        _x1, _x2, _x3, _x4, _ = self.linformer_layer(x1, x2, x3, x4, self.lf1)
        #print(_x1.shape)
        x1, x2, x3, x4 = x1 + _x1, x2 + _x2, x3 + _x3, x4 + _x4

        x1 = self.layer2(x1)
        #print('x1 lf2:',x1.shape)
        x2 = self.layer2(x2)
        #print('x2 lf2:', x2.shape)
        x3 = self.layer2(x3)
        #print('x3 lf2:', x3.shape)
        x4 = self.layer2(x4)
        #print('x4 lf2:', x4.shape)

        #print('after layer2 shape:',x1.shape) [3, 128, 16, 16, 16]
        _x1, _x2, _x3, _x4, _ = self.linformer_layer(x1, x2, x3, x4, self.lf2)
        x1, x2, x3, x4 = x1 + _x1, x2 + _x2, x3 + _x3, x4 + _x4

        x1 = self.layer3(x1)
        #print('x1 lf3',x1.shape)
        x2 = self.layer3(x2)
        #print('x2 lf3', x2.shape)
        x3 = self.layer3(x3)
        #print('x3 lf3', x3.shape)
        x4 = self.layer3(x4)
        #print('x4 lf3', x4.shape)

        _x1, _x2, _x3, _x4, _ = self.linformer_layer(x1, x2, x3, x4, self.lf3)
        x1, x2, x3, x4 = x1 + _x1, x2 + _x2, x3 + _x3, x4 + _x4

        x1 = self.layer4(x1)
        #print('x1 lf4',x1.shape)
        x2 = self.layer4(x2)
        #print('x2 lf4', x2.shape)
        x3 = self.layer4(x3)
        #print('x3 lf4', x3.shape)
        x4 = self.layer4(x4)
        #print('x4 lf4', x4.shape)

        # cls_x1 = self.fc_x1(torch.cat((self.dropout(x1.mean(dim=(2, 3, 4))), info), dim=1))
        # cls_x2 = self.fc_x2(torch.cat((self.dropout(x2.mean(dim=(2, 3, 4))), info), dim=1))
        # cls_x3 = self.fc_x3(torch.cat((self.dropout(x3.mean(dim=(2, 3, 4))), info), dim=1))
        # cls_x4 = self.fc_x4(torch.cat((self.dropout(x4.mean(dim=(2, 3, 4))), info), dim=1))

        x1, x2, x3, x4, _ = self.linformer_layer(x1, x2, x3, x4, self.lf4)

        x1 = x1.reshape(x1.size(0),-1)
        x2 = x2.reshape(x2.size(0), -1)
        x3 = x3.reshape(x3.size(0), -1)
        x4 = x4.reshape(x4.size(0), -1)

        x = torch.concat((x1,x2,x3,x4),dim=1)
        x = self.dropout(self.fc(x))

        return x

def generate_model_BraTS_LF(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    model = []
    if model_depth == 10:
        model = ResNet_Linformer(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet_Linformer(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet_Linformer(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet_Linformer(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet_Linformer(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet_Linformer(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet_Linformer(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

if __name__ == '__main__':
    x1 = torch.rand((3,1,42,128,128))
    x2 = torch.rand((3,1,22,90,90))
    x3 = torch.rand((3,1,16,68,68))
    x4 = torch.rand((3,1,8,48,48))

    info = torch.zeros((1,6))
    resnet_transformer = generate_model_BraTS_LF(model_depth=50)
    output = resnet_transformer(x1,x2,x3,x4)

    #print(resnet_transformer)
    output = torch.mean(output,dim=1,keepdim=True)
    #print(output)







