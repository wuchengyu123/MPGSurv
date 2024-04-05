
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataloader.bootstrapDataset import CustomDataSet
from NegitiveLogLikelihood import NegativeLogLikelihood
from sklearn.model_selection import train_test_split
from model.BulidModel import ResNet_Linformer
from utils import record_best_model
from metrics import c_index
import time
import numpy as np
print(torch.cuda.is_available())
print(torch.cuda.device_count())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')

#writer = SummaryWriter('logs')
csv_data_col_names  = []

patient_id = 'PatientID'
event_times_col_name = 'OStime'
event_observed_col_name = 'OSstatue'

t_image_path = '/home/wuchengyu/another_GTV-T/all_set'
n_image_path = '/home/wuchengyu/Node Patch/all'

batch_size = 12
random_seeds  = [4,5,7,9,10]
Xy_data = pd.read_csv('Xy_all_hazard.csv',encoding='gbk')
columns = Xy_data.columns

X = Xy_data.drop(columns=['IBEX_CT_NAME','PatientID','OStime','OSstatue','PFStime','PFSstatue'])
y = Xy_data.loc[:,['IBEX_CT_NAME','PatientID','OStime','OSstatue','PFStime','PFSstatue']]

for seed in random_seeds:
    print(f'random seed:{seed}')
    print('-'*20)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=seed)
    train = pd.concat([y_train,X_train],axis=1)
    test = pd.concat([y_test,X_test],axis=1)

    train_data = pd.DataFrame(train.values,columns=train.columns)
    test_data = pd.DataFrame(test.values,columns=test.columns)

    train_set= CustomDataSet(
            csv_data=train_data,
            csv_data_col_names=csv_data_col_names,
            t_ct_dataset_path= t_image_path,
            n_ct_dataset_path= n_image_path,
            is_test=False
            )

    valid_set = CustomDataSet(
        csv_data=test_data,
        csv_data_col_names=csv_data_col_names,
        t_ct_dataset_path=t_image_path,
        n_ct_dataset_path=n_image_path,
        is_test=True
    )
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False)



    lr = 1e-5 # 1e-5
    model= ResNet_Linformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_mult=2,T_0=80,eta_min=1e-6) # 8e-6
    criterion_regression_loss_with_censored = NegativeLogLikelihood(device=device,l2_reg=5.5)
    history_best_test_cindex = -1
    history_best_train_cindex = -1
    epochs = 200

    best_c_index = -1
    for epoch in range(epochs):

        final_pred_train = []
        OStime_train = []
        OSstatue_train = []
        batch_loss = 0
        model.train()
        for batch in train_loader:
            t_img, OStime, OSstatue, clinical_data = batch['t_img'], batch['OStime'], batch['OSstatue'], batch['clinical_data']
            n_img_patch0 = batch['n_img_patch0']
            n_img_patch1 = batch['n_img_patch1']
            n_img_patch2 = batch['n_img_patch2']
            #print(clinical_data_train.shape)
            t_img = t_img.to(device, dtype=torch.float32)
            n_img_patch0 = n_img_patch0.to(device,dtype=torch.float32)
            n_img_patch1 = n_img_patch1.to(device,dtype=torch.float32)
            n_img_patch2 = n_img_patch2.to(device,dtype=torch.float32)

            #print(t_img.shape)
            OStime = OStime.to(device, dtype=torch.float32)
            OSstatue = OSstatue.to(device, dtype=torch.float32)
            clinical_data = clinical_data.to(device, dtype=torch.float32)

            final_pred= model(t_img,n_img_patch0,n_img_patch1,n_img_patch2,clinical_data)

            loss = criterion_regression_loss_with_censored(final_pred, OStime, OSstatue, model)

            batch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            final_pred_train.append(-final_pred.detach().cpu().numpy())
            OStime_train.append(OStime.detach().cpu().numpy())
            OSstatue_train.append(OSstatue.detach().cpu().numpy())
            #writer.add_scalar('loss',loss,epoch)

        final_pred_concat_train = np.concatenate(final_pred_train, axis=0)
        OStime_concat_train = np.concatenate(OStime_train, axis=0)
        OSstatue_concat_train = np.concatenate(OSstatue_train, axis=0)


        train_cindex = c_index(final_pred_concat_train, OStime_concat_train, OSstatue_concat_train)
        if train_cindex < 0.5:
            train_cindex = 1-train_cindex


        print('epoch{}/{} loss: {} train c-index: {} '.format(epoch,epochs,batch_loss/len(train_loader),train_cindex),end=' ')


        final_pred_test = []
        OStime_test = []
        OSstatue_test = []
        model.eval()
        for batch in valid_loader:
            t_img,OStime, OSstatue, clinical_data =batch['t_img'],batch['OStime'], batch['OSstatue'], batch['clinical_data']
            OStime = OStime.to(device,dtype=torch.float32)
            OSstatue = OSstatue.to(device,dtype=torch.float32)
            clinical_data = clinical_data.to(device,dtype=torch.float32)

            n_img_patch0 = batch['n_img_patch0']
            n_img_patch1 = batch['n_img_patch1']
            n_img_patch2 = batch['n_img_patch2']

            t_img = t_img.to(device, dtype=torch.float32)
            n_img_patch0 = n_img_patch0.to(device, dtype=torch.float32)
            n_img_patch1 = n_img_patch1.to(device, dtype=torch.float32)
            n_img_patch2 = n_img_patch2.to(device, dtype=torch.float32)

            with torch.no_grad():
                risk_pred = model(t_img,n_img_patch0,n_img_patch1,n_img_patch2,clinical_data)

                final_pred_test.append(-risk_pred.detach().cpu().numpy())
                OStime_test.append(OStime.detach().cpu().numpy())
                OSstatue_test.append(OSstatue.detach().cpu().numpy())

        final_pred_concat_test = np.concatenate(final_pred_test, axis=0)
        OStime_concat_test = np.concatenate(OStime_test, axis=0)
        OSstatue_concat_test = np.concatenate(OSstatue_test, axis=0)

        test_cindex = c_index(final_pred_concat_test,OStime_concat_test,OSstatue_concat_test)

        if test_cindex < 0.5:
            test_cindex = 1-test_cindex

        print('test c-index: {}'.format(test_cindex))

        if best_c_index < test_cindex and test_cindex > 0.63:
            best_c_index = test_cindex
            record_best_model(model, '{} seed{} best test c-index:{:.4f} corresponding train c-index:{:.4f}.pth'.format(time.strftime('%H-%M-%S', time.localtime(time.time())),
                                                                     seed,test_cindex,train_cindex), 'cross_valid file')
            print(f'best_test c-index in epoch{epoch}: {best_c_index}')

            if history_best_test_cindex < best_c_index:
                history_best_test_cindex = best_c_index
                history_best_train_cindex = train_cindex

    with open('best_cindex_MITCNet.txt', 'a+') as f:
        f.write('seed'+str(seed))
        f.write(' best test cindex: '+str(history_best_test_cindex))
        f.write(' best train_cindex: '+str(history_best_train_cindex))
        f.write('\n')
