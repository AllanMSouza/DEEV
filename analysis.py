import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = 'MotionSense'

paths = {'FedAvg'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/FedAvg-All/DNN/evaluate_client.csv',
         'POC-0.5'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/POC-POC-0.5/DNN/evaluate_client.csv',
         'POC-0.1'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/POC-POC-0.1/DNN/evaluate_client.csv',
         'DEEV-0.01'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/DEEV-DEEV-0.01/DNN/evaluate_client.csv',
         'DEEV-0.1'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/DEEV-DEEV-0.1/DNN/evaluate_client.csv',
         'DEEV-0.5'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/DEEV-DEEV-0.5/DNN/evaluate_client.csv',
         'teste'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/teste-DEEV-0.5/DNN/evaluate_client.csv'
         }

solutions = paths.keys()
for sol in solutions:
    acc =  pd.read_csv(paths[sol] , names=['rounds', 'client', 'comm','loss', 'acc'])
    sns.lineplot(acc.groupby('rounds').mean().reset_index(), y = 'acc', x = 'rounds', legend='brief', label=sol)
plt.show()



# acc =  pd.read_csv('/home/gabrieltalasso/DEEV/logs/MotionSense/FedAvg2-All/DNN/train_client.csv' ,
#                     names=['rounds', 'client', 'comm','loss', 'acc_train', 'acc_test'])

# sns.lineplot(acc, y = 'acc_train', x = 'rounds', legend='brief', label='Local_train')
# sns.lineplot(acc, y = 'acc_test', x = 'rounds', legend='brief', label='Local_test')

# acc =  pd.read_csv('/home/gabrieltalasso/DEEV/logs/MotionSense/FedAvg2-All/DNN/evaluate_client.csv' ,
#                     names=['rounds', 'client', '_', '--', 'comm','loss', 'acc', 'acc_train', 'acc_test'])

# sns.lineplot(acc, y = 'acc_train', x = 'rounds', legend='brief', label='Global_train')
# sns.lineplot(acc, y = 'acc_test', x = 'rounds', legend='brief', label='Global_test')

# plt.show()

