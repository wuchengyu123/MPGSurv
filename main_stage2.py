import time

import numpy as np
import torch
import random
import math
from models import GraphSage
from metirc import c_index
from csv_gen import hazard_file
from NegitiveLogLikelihood import NegativeLogLikelihood
from models import UnsupervisedLoss
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import record_best_model
from utils import get_train_test_nodes
from dataloader.GraphDataset import CustomDataSet
from dataloader.DataCenter import DataCenter
from torch.utils.data import DataLoader

csv_data_col_names  = []

batch_size = 12
random_seeds  = [4,5,7,9,10]
Xy_data = pd.read_csv('Xy_all_hazard.csv',encoding='gbk')
columns = Xy_data.columns

X = Xy_data.drop(columns=['IBEX_CT_NAME','PatientID','OStime','OSstatue','PFStime','PFSstatue'])
y = Xy_data.loc[:,['IBEX_CT_NAME','PatientID','OStime','OSstatue','PFStime','PFSstatue']]

for seed in random_seeds:

    print(f'random seed:{seed}')
    l1 = 0.5

    train_set = CustomDataSet(
        csv_data=train_data,
        csv_data_col_names=csv_data_col_names,
    )

    valid_set = CustomDataSet(
        csv_data=valid_data,
        csv_data_col_names=csv_data_col_names,
    )
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)

    file_paths = {'feature':'./CBAM feature_vector/feature_seed{}.txt'.format(seed),'cite':'./CBAM feature_vector/cite_all_reID.txt'}
    datacenter = DataCenter(file_paths)
    node_map = datacenter.load_Dataset()
    feature_data = torch.FloatTensor(getattr(datacenter, 'features'))
    adj_lists = getattr(datacenter,'adj_lists')
    train_nodes,test_nodes = get_train_test_nodes(node_map,train_loader,test_loader)

    eopchs = 200
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphSage = GraphSage(2, feature_data.size(1), 128, feature_data, adj_lists, device, gcn=False, agg_func='MEAN')

    optimizer = torch.optim.Adam(graphSage.parameters(),lr=lr,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=80,eta_min=1e-6)

    criterion = NegativeLogLikelihood(l2_reg=0.2,device=device)
    unsupervised_loss = UnsupervisedLoss(adj_lists, train_nodes, device)

    test_best_cindex = 0
    train_best_cindex = 0
    best_epoch = 0
    for eopch in range(eopchs):
        final_pred_train = []
        OStime_train = []
        OSstatue_train = []
        graphloss  = 0
        for batch in train_loader:
            train_nodes_batch = []
            PatientID = batch['PatientID']
            OStime = batch['OStime']
            OSstatue = batch['OSstatue']
            tabular_data = batch['clinical_data']

            OStime = OStime.to(device)
            OSstatue = OSstatue.to(device)

            for i in PatientID:
                train_nodes_batch.append(node_map[int(i)])
            train_nodes_batch = np.asarray(train_nodes_batch)
            train_nodes_ = np.asarray(list(unsupervised_loss.extend_nodes(train_nodes_batch, num_neg=3))) # 3

            node_embedding,_ = graphSage(train_nodes_)
            loss_net = unsupervised_loss.get_loss_sage(node_embedding, train_nodes_)

            _,final_pred = graphSage(train_nodes_batch)
            final_pred_train.append(-final_pred.detach().cpu().numpy())
            OStime_train.append(OStime.detach().cpu().numpy())
            OSstatue_train.append(OSstatue.detach().cpu().numpy())
            pred_hazard = final_pred.to(device)

            loss_survival = criterion(pred_hazard, OStime, OSstatue, graphSage)
            lambda1 = 0.5
            loss = lambda1*loss_survival  + (1-lambda1)*loss_net

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        final_pred_concat_train = np.concatenate(final_pred_train, axis=0)
        OStime_concat_train = np.concatenate(OStime_train, axis=0)
        OSstatue_concat_train = np.concatenate(OSstatue_train, axis=0)

        train_cindex = c_index(OStime_concat_train, final_pred_concat_train, OSstatue_concat_train)
        train_cindex = train_cindex if train_cindex>0.5 else 1-train_cindex
        print('graph loss: {}'.format(graphloss/len(train_loader)),end=' ')


        final_pred_test = []
        OStime_test = []
        OSstatue_test = []
        graphSage.eval()
        for batch in test_loader:
            test_nodes_batch = []
            PatientID = batch['PatientID']
            OStime = batch['OStime']
            OSstatue = batch['OSstatue']
            tabular_data = batch['clinical_data']

            event_times = OStime.to(device)
            event_observed = OSstatue.to(device)

            for i in PatientID:
                test_nodes_batch.append(node_map[int(i)])
            test_nodes_batch = np.asarray(test_nodes_batch)

            _,risk_pred = graphSage(test_nodes_batch)
            final_pred_test.append(-risk_pred.detach().cpu().numpy())
            OStime_test.append(OStime.detach().cpu().numpy())
            OSstatue_test.append(OSstatue.detach().cpu().numpy())

        final_pred_concat_test = np.concatenate(final_pred_test, axis=0)
        OStime_concat_test = np.concatenate(OStime_test, axis=0)
        OSstatue_concat_test = np.concatenate(OSstatue_test, axis=0)



        test_cindex = c_index(OStime_concat_test,final_pred_concat_test,  OSstatue_concat_test)

        print('eopch{}/{} test c-index: {}'.format(eopch,eopchs,test_cindex))

        if test_best_cindex < test_cindex and test_cindex > 0.60:
            test_best_cindex = test_cindex
            train_best_cindex = train_cindex
            best_epoch = eopch
            record_best_model(graphSage, '{} seed{} test best cindex {:.4f}.pth'.format(time.strftime('%H-%M-%S', time.localtime(time.time())),seed,test_best_cindex))

    with open('best_cindex.txt', 'a+') as f:
        f.write('seed' + str(seed))
        f.write(' best test cindex: ' + str(test_best_cindex))
        f.write('\n')
    print('epoch:{} train best cindex:{} test best cindex:{}'.format(best_epoch,train_best_cindex,test_best_cindex))

