import pandas as pd
import pickle
import numpy as np

def load_motionsense_cli_data(cid, data_dir):
    with open(data_dir + f"/{cid}_train.pickle", 'rb') as train_file:
        train = pd.read_pickle(train_file)

    with open(data_dir + f"/{cid}_test.pickle", 'rb') as test_file:
        test = pd.read_pickle(test_file)

    y_train = train['activity'].values
    train.drop('activity', axis=1, inplace=True)
    train.drop('subject', axis=1, inplace=True)
    train.drop('trial', axis=1, inplace=True)

    x_train = train.values

    y_test = test['activity'].values
    test.drop('activity', axis=1, inplace=True)
    test.drop('subject', axis=1, inplace=True)
    test.drop('trial', axis=1, inplace=True)

    x_test = test.values

    return x_train, y_train, x_test, y_test


def load_ExtraSensory(cid):
    
    with open(f'Client/data/ExtraSensory/x_train_client_{cid}.pickle', 'rb') as x_train_file:
        x_train = pickle.load(x_train_file)

    with open(f'Client/data/ExtraSensory/x_test_client_{cid}.pickle', 'rb') as x_test_file:
        x_test = pickle.load(x_test_file)
    
    with open(f'Client/data/ExtraSensory/y_train_client_{cid}.pickle', 'rb') as y_train_file:
        y_train = pickle.load(y_train_file)

    with open(f'Client/data/ExtraSensory/y_test_client_{cid}.pickle', 'rb') as y_test_file:
        y_test = pickle.load(y_test_file)

    y_train = np.array(y_train) + 1
    #print('------------------------------', len(y_train), np.max(y_train))
    y_test  = np.array(y_test) + 1

    return x_train, y_train, x_test, y_test

def get_dataset(dataset_name, dir_path, cid):

    if dataset_name == "MotionSense":
        x_train, y_train, x_test, y_test = load_motionsense_cli_data(cid+1, dir_path)
    elif dataset_name == 'ExtraSensory':
        x_train, y_train, x_test, y_test = load_ExtraSensory(cid+1)    
    else:
        return None

    return x_train, y_train, x_test, y_test