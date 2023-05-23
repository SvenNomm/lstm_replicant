import pickle as pkl


def data_loader(path, fname_1, fname_2):
    with open(path + fname_1, 'rb') as f:
        data_1 = pkl.load(f)

    initial_data_train = data_1[0]
    initial_data_valid = data_1[1]
    target_data_train = data_1[2]
    target_data_valid = data_1[3]
    valid_index = data_1[4]

    with open(path + fname_2, 'rb') as f:
        data_2 = pkl.load(f)

    input_data_test = data_2[0]
    target_data_test = data_2[1]
    test_data_indexes = data_2[2]

    return initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index, input_data_test, \
        target_data_test, test_data_indexes
