import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = 'MotionSense'

#acuracia das soluções por round
paths = {'FedAvg'   :   'logs/MotionSense/FedAvg-All/DNN/evaluate_client.csv',
         'POC-0.5'   :  'logs/MotionSense/POC-POC-0.5/DNN/evaluate_client.csv',
         'POC-0.1'   :  'logs/MotionSense/POC-POC-0.1/DNN/evaluate_client.csv',
         'DEEV-0.01'   :'logs/MotionSense/DEEV-DEEV-0.01/DNN/evaluate_client.csv',
         'DEEV-0.1'   : 'logs/MotionSense/DEEV-DEEV-0.1/DNN/evaluate_client.csv',
         'DEEV-0.5'   : 'logs/MotionSense/DEEV-DEEV-0.5/DNN/evaluate_client.csv',
         'teste2_RAWCS':'logs/MotionSense/teste2-Rawcs/DNN/evaluate_client.csv',
         'teste3_RAWCS':'logs/MotionSense/teste3-Rawcs/DNN/evaluate_client.csv',
         'teste4_RAWCS':'logs/MotionSense/teste4-Rawcs/DNN/evaluate_client.csv',
         }

solutions = paths.keys()
for sol in solutions:
    acc =  pd.read_csv(paths[sol] , names=['rounds', 'client', 'comm','loss', 'acc'])
    sns.lineplot(acc.groupby('rounds').mean().reset_index(), y = 'acc', x = 'rounds', legend='brief', label=sol)
plt.show()


#quantidade de clientes selecionados por round
paths = {'FedAvg'   :   'logs/MotionSense/FedAvg-All/DNN/train_client.csv',
         'POC-0.5'   :  'logs/MotionSense/POC-POC-0.5/DNN/train_client.csv',
         'POC-0.1'   :  'logs/MotionSense/POC-POC-0.1/DNN/train_client.csv',
         'DEEV-0.01'  : 'logs/MotionSense/DEEV-DEEV-0.01/DNN/train_client.csv',
         'DEEV-0.1'   : 'logs/MotionSense/DEEV-DEEV-0.1/DNN/train_client.csv',
         'DEEV-0.5'   : 'logs/MotionSense/DEEV-DEEV-0.5/DNN/train_client.csv',
         'teste2_RAWCS':'logs/MotionSense/teste2-Rawcs/DNN/train_client.csv',
         'teste3_RAWCS':'logs/MotionSense/teste3-Rawcs/DNN/train_client.csv',
         'teste4_RAWCS':'logs/MotionSense/teste4-Rawcs/DNN/train_client.csv',
         }

solutions = paths.keys()
for sol in solutions:
    acc =  pd.read_csv(paths[sol] , names=['rounds', 'client', '-', '--', '---', 'loss', 'acc'])
    sns.lineplot(acc.groupby('rounds').count().reset_index(), y = 'acc', x = 'rounds', legend='brief', label=sol)
    plt.ylabel('number of clients selected')
plt.show()




paths = {'FedAvg'   :'logs/ExtraSensory/FedAvg-All/DNN/evaluate_client.csv',
         'DEEV-0.1'   :'logs/ExtraSensory/deev-DEEV-0.1/DNN/evaluate_client.csv',
         'POC-0.1'   :'logs/ExtraSensory/POC-POC-0.1/DNN/evaluate_client.csv',
         'RAWCS': 'logs/ExtraSensory/RAWCS-RAWCS/DNN/evaluate_client.csv'}

solutions = paths.keys()
for sol in solutions:
    acc =  pd.read_csv(paths[sol] , names=['rounds', 'client', 'comm','loss', 'acc'])
    sns.lineplot(acc.groupby('rounds').mean().reset_index(), y = 'acc', x = 'rounds', legend='brief', label=sol)
plt.show()