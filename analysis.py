import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = 'MotionSense'

#acuracia das soluções por round
paths = {'FedAvg'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/FedAvg-All/DNN/evaluate_client.csv',
         'POC-0.5'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/POC-POC-0.5/DNN/evaluate_client.csv',
         'POC-0.1'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/POC-POC-0.1/DNN/evaluate_client.csv',
         'DEEV-0.01'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/DEEV-DEEV-0.01/DNN/evaluate_client.csv',
         'DEEV-0.1'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/DEEV-DEEV-0.1/DNN/evaluate_client.csv',
         'DEEV-0.5'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/DEEV-DEEV-0.5/DNN/evaluate_client.csv',
         'teste2_RAWCS':'/home/gabrieltalasso/DEEV/logs/MotionSense/teste2-Rawcs/DNN/evaluate_client.csv',
         'teste3_RAWCS':'/home/gabrieltalasso/DEEV/logs/MotionSense/teste3-Rawcs/DNN/evaluate_client.csv'
         }

solutions = paths.keys()
for sol in solutions:
    acc =  pd.read_csv(paths[sol] , names=['rounds', 'client', 'comm','loss', 'acc'])
    sns.lineplot(acc.groupby('rounds').mean().reset_index(), y = 'acc', x = 'rounds', legend='brief', label=sol)
plt.show()


#quantidade de clientes selecionados por round
paths = {'FedAvg'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/FedAvg-All/DNN/train_client.csv',
         'POC-0.5'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/POC-POC-0.5/DNN/train_client.csv',
         'POC-0.1'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/POC-POC-0.1/DNN/train_client.csv',
         'DEEV-0.01'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/DEEV-DEEV-0.01/DNN/train_client.csv',
         'DEEV-0.1'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/DEEV-DEEV-0.1/DNN/train_client.csv',
         'DEEV-0.5'   :'/home/gabrieltalasso/DEEV/logs/MotionSense/DEEV-DEEV-0.5/DNN/train_client.csv',
         'teste2_RAWCS':'/home/gabrieltalasso/DEEV/logs/MotionSense/teste2-Rawcs/DNN/train_client.csv',
         'teste3_RAWCS':'/home/gabrieltalasso/DEEV/logs/MotionSense/teste3-Rawcs/DNN/train_client.csv'
         }
solutions = paths.keys()
for sol in solutions:
    acc =  pd.read_csv(paths[sol] , names=['rounds', 'client', '-', '--', '---', 'loss', 'acc'])
    sns.lineplot(acc.groupby('rounds').count().reset_index(), y = 'acc', x = 'rounds', legend='brief', label=sol)
    plt.ylabel('number of clients selected')
plt.show()
