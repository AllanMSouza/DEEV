import pandas as pd

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
def get_dataset(dataset_name, dir_path, cid):
    if dataset_name == "MotionSense":
        x_train, y_train, x_test, y_test = load_motionsense_cli_data(cid + 1, dir_path)
    else:
        return

    return x_train, y_train, x_test, y_test