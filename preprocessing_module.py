import pandas as pd
from sklearn.model_selection import train_test_split
from random import randrange
import torch
import torch.nn.functional as F
import numpy as np


def initial_formatting_old_data(initial_data_1, initial_data_2, target_data):
    print("Aloha! Performing initial formatting")

    # the following lines are very local
    del initial_data_1['time']
    del initial_data_2['time']
    del target_data['time']
    del initial_data_1['station_ind']
    del initial_data_2['station_ind']
    del target_data['station_ind']
    #del initial_data_1['Unnamed: 0']
    #del initial_data_2['Unnamed: 0']
    #del target_data['Unnamed: 0']

    # target_data.drop(target_data.index[0], inplace=True)
    # target_data = target_data.reset_index()
    # del target_data['index']

    print("Initial  formatting has been completed!")
    return initial_data_1, initial_data_2, target_data


def time_based_splitter(input_data_1, input_data_2, target_data):
    # get column list and create new data frames
    input_columns_1 = input_data_1.columns.values.tolist()
    input_frame_1 = pd.DataFrame(columns=input_columns_1)

    input_columns_2 = input_data_2.columns.values.tolist()
    input_frame_2 = pd.DataFrame(columns=input_columns_2)

    target_columns = target_data.columns.values.tolist()
    output_frame = pd.DataFrame(columns=target_columns)

    # create list of unique time instances and #split it between training and testing
    time_list = input_data_1['time'].drop_duplicates()
    time_list_train, time_list_test = train_test_split(time_list, train_size=0.7, test_size=0.3)

    # crfeate data frames containing data for training and testing
    input_data_1_train = input_data_1.loc[input_data_1['time'].isin(time_list_train)]
    input_data_2_train = input_data_2.loc[input_data_1['time'].isin(time_list_train)]
    target_data_train = target_data.loc[input_data_1['time'].isin(time_list_train)]

    input_data_1_test = input_data_1.loc[input_data_1['time'].isin(time_list_test)]
    input_data_2_test = input_data_2.loc[input_data_2['time'].isin(time_list_test)]
    target_data_test = target_data.loc[input_data_1['time'].isin(time_list_test)]

    for instance in time_list_test:
        # go through all the time instances in testing fdata set
        time_frame = input_data_1_test.loc[input_data_1_test['time'] == instance]
        time_frame_2 = input_data_2_test.loc[input_data_2_test['time'] == instance]
        # get the station list for each instance
        station_list = time_frame['station_ind'].drop_duplicates().tolist()
        station_list_2 = time_frame['station_ind'].drop_duplicates().tolist()

        for station in station_list:
            # go through each station
            station_frame = time_frame.loc[time_frame['station_ind'] == station]
            index_list = station_frame.index.values.tolist()
            # get only one record per station  should we use random here instead of index zero
            #input_frame_1 = pd.concat([input_frame_1, station_frame.loc[index_list[0], :].to_frame().T])

            # for using random selection of the area
            idx = np.random.randint(0, high=len(index_list), dtype=int)
            input_frame_1 = pd.concat([input_frame_1, station_frame.loc[index_list[idx], :].to_frame().T])
            # should we use random here instead of index zero?
            #idx = randrange(len(index_list))
            #input_frame_1 = pd.concat([input_frame_1, station_frame.loc[index_list[idx], :].to_frame().T])

    # get indexes of the data
    test_indexes = input_frame_1.index.values.tolist()

    # form the data frames
    input_data_2_test = input_data_2_test.loc[test_indexes]
    target_data_test = target_data_test.loc[test_indexes]
    input_data_1_test = input_frame_1

    # reset indexes
    input_data_1_train = input_data_1_train.reset_index()
    del input_data_1_train['index']

    input_data_2_train = input_data_2_train.reset_index()
    del input_data_2_train['index']

    target_data_train = target_data_train.reset_index()
    del target_data_train['index']

    input_data_1_test= input_data_1_test.reset_index()
    del input_data_1_test['index']

    input_data_2_test = input_data_2_test.reset_index()
    del input_data_2_test['index']

    target_data_test = target_data_test.reset_index()
    del target_data_test['index']

    return input_data_1_train, input_data_2_train, target_data_train, input_data_1_test, input_data_2_test, target_data_test, test_indexes


def create_dataset_ng(**kwargs):
    kwargs_keys = kwargs.keys()
    results = []

    for key in kwargs_keys:
        output = []
        arr = kwargs[key]
        n, m = kwargs[key].shape
        for i in range(0, n):
            output.append(np.transpose(arr[i, :]).reshape([m, 1]))

        output = np.asarray(output)
        output = torch.tensor(output).float()
        results.append(output)

    return results

def combine_polarisations_log_div_norm(tensor_1, tensor_2):
    tensor_1_log = torch.log(tensor_1)
    tensor_2_log = torch.log(tensor_2)
    tensor_log_div = torch.div(tensor_1_log, tensor_2_log)
    tensor_log_div_norm = F.normalize(tensor_log_div)
    return tensor_log_div_norm


def combine_polarisations_div_norm(tensor_1, tensor_2):
    tensor_div = torch.div(tensor_1, tensor_2)
    tensor_div_norm = F.normalize(tensor_div)
    return tensor_div_norm


def combine_polarisations_div_tensor(tensor_1, tensor_2, tensor_3, device):
    tensor_div = torch.div(tensor_1, tensor_2)
    input_data = tensor_div.detach().cpu().numpy()
    target_data = tensor_3.detach().cpu().numpy()

    input_data = pd.DataFrame(input_data[:, :, 0])
    target_data = pd.DataFrame(target_data[:, :, 0])

    input_data, target_data = delete_nan_rows(input_data, target_data)

    input_data = input_data.values
    target_data = target_data.values

    input_data, target_data = create_dataset_ng(input_data=input_data, target_data=target_data)
    #input_data = F.normalize(input_data)
    #target_data = F.normalize(target_data)

    return input_data.to(device), target_data.to(device)


def cut_tail(data, length):
    data_trunc = data[:, 0:length]
    return data_trunc


def delete_nan_rows(df1, df2):
    selected_rows = df1.loc[df1.isna().any(axis=1)].index.tolist()
    df1 = df1.drop(df1.index[selected_rows])
    df2 = df2.drop(df2.index[selected_rows])
    return df1, df2