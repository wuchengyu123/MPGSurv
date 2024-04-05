import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class DeepSurv_1dcnn(nn.Module):
    def __init__(self):
        super(DeepSurv_1dcnn, self).__init__()
        self.deepsurv = nn.Sequential(
            nn.Linear(54,256),
            nn.BatchNorm1d(256),
            nn.ELU(alpha=0.5,inplace=False),
            nn.Dropout(p=0.22),
            nn.Linear(256,16),
            nn.BatchNorm1d(16),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.22),
            nn.Linear(16,8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=0.22),
            nn.Linear(8,1)
        )

        self.conv_batch = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1,padding=1),
        nn.BatchNorm1d(16),
        nn.AvgPool1d(2),
        nn.SELU(),
        nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2,padding=1),
        nn.BatchNorm1d(32),
        nn.AvgPool1d(2),
        nn.SELU(),
        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=3,padding=1),
        nn.SELU(),
        nn.AvgPool1d(2),
        nn.Flatten()
    )

        self.dilation_conv_batch = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=16, dilation=1, kernel_size=3, stride=1,padding=0),
        nn.BatchNorm1d(16),
        nn.LeakyReLU(),
        nn.Conv1d(in_channels=16, out_channels=32, dilation=2, kernel_size=3, stride=2,padding=0),
        nn.BatchNorm1d(32),
        nn.LeakyReLU(),
        nn.Conv1d(in_channels=32, out_channels=64, dilation=3, kernel_size=3, stride=3,padding=0),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(),
        nn.Flatten()
    )
        for m in self.modules():
            if isinstance(m,nn.Linear):
                init.kaiming_uniform_(m.weight,mode='fan_out',nonlinearity='relu')  #best c-index:0.8492702821126428 in epcho 781

    def forward(self,clinical_features):
        out4 = clinical_features
        out3 = self.deepsurv(clinical_features)
        all_features = clinical_features#torch.concat((clinical_features,radiomics_features),dim=1)
        #out3 = all_features
        #out3 = self.ds(all_features)
        features_size_2 = all_features.size(1)
        all_features = all_features.reshape(-1,1,features_size_2)

        out1 = self.conv_batch(all_features)
        out2 = self.dilation_conv_batch(all_features)
        output = torch.concat((out1,out2,out3,out4),dim=1)
        return output


if __name__ == '__main__':
    x1 = torch.rand((256,45))
    model = DeepSurv_1dcnn()
    output = model(x1)
    print(output.shape)