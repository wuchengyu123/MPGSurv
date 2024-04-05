import torch
from torch import nn
from model.ResNet3D_BraTS import generate_model_BraTS_LF
from model.DeepSurv import DeepSurv_1dcnn
import torch.nn.functional as F

class ResNet_Linformer(nn.Module):
    def __init__(self):
        super(ResNet_Linformer,self).__init__()

        self.resnet_linformer = generate_model_BraTS_LF(model_depth=50)
        self.conv_deepsurv = DeepSurv_1dcnn()
        self.linear_projection = nn.Linear(759,1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, t, n_patch0, n_patch1, n_patch2,tabular_data):
        res_linformer_output = self.resnet_linformer(t,n_patch0,n_patch1,n_patch2)
        conv_deepsurv_output = self.conv_deepsurv(tabular_data)

        #one_d_conv_output = self.oned_conv(tabular_data)
        output = torch.concat((res_linformer_output,conv_deepsurv_output),dim=1)
        output = F.tanh(self.linear_projection(output))

        return output
