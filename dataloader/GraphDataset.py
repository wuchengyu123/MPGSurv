import os.path
import pandas as pd
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import torchio as tio

class CustomDataSet(Dataset):
    def __init__(self,
                 csv_data,  # 临床csv特征文件
                 csv_data_col_names, # 使用特征列名
                 ):
        super(CustomDataSet, self).__init__()

        self.csv_data_col_names = csv_data_col_names
        self.csv_data = csv_data
        self.patient_clinical_data = self.csv_data[[*csv_data_col_names]].values

        self.csv_IBEX_CT_NAME = np.squeeze(self.csv_data.loc[:,['IBEX_CT_NAME']].values)


    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index):
        clinical_data = np.asarray(self.patient_clinical_data[index],dtype=np.float32)
        OSstatue = np.asarray(self.csv_data.loc[index,'OSstatue'],dtype=np.float32)
        OStime = np.asarray(self.csv_data.loc[index,'OStime'],dtype=np.float32)
        PatientID = np.asarray(self.csv_data.loc[index, 'PatientID'], dtype=np.int32)


        OStime = np.expand_dims(OStime,axis=-1)
        OSstatue = np.expand_dims(OSstatue,axis=-1)
        PatientID = np.expand_dims(PatientID, axis=-1)

        return {'OSstatue': OSstatue, 'clinical_data': clinical_data,'PatientID':PatientID, 'OStime': OStime}
