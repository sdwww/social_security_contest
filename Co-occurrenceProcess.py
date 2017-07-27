import csv
import time

import matplotlib.pyplot as plt
import numpy as np

import DBOptions


def load_train_list():
    train_list = []
    with open('./origin_data/DF_ID_TRAIN.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            train_list.append(row)
    train_list = train_list[1:]
    train_id_dict = {}
    for i in range(len(train_list)):
        train_id_dict[i] = train_list[i][0]
    return train_id_dict

def load_test_list():
    test_list = []
    with open('./origin_data/DF_ID_TEST.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            test_list.append(row)
    test_list = test_list[1:]
    train_id_dict = {}
    for i in range(len(test_list)):
        train_id_dict[i] = test_list[i][0]
    return train_id_dict


def get_train_co_corrence(train_id_dict):
    co_occurrence_matrix = np.load('./data_npz/co-occurrence matrix.npz')['co_occurrence_matrix']
    co_occurrence_dict = {}
    for i in range(len(co_occurrence_matrix)):
        co_occurrence_matrix[i][i] = 0
    for i in range(len(co_occurrence_matrix)):
        co_occurrence_dict[train_id_dict[i]] = np.max(co_occurrence_matrix[i])
    print(co_occurrence_dict)
    return co_occurrence_dict

def get_test_co_corrence(train_id_dict):
    co_occurrence_matrix = np.load('./data_npz/co_occurrence_test.npz')['arr_0']
    co_occurrence_dict = {}
    for i in range(len(co_occurrence_matrix)):
        co_occurrence_matrix[i][i] = 0
    for i in range(len(co_occurrence_matrix)):
        co_occurrence_dict[train_id_dict[i]] = np.max(co_occurrence_matrix[i])
    print(co_occurrence_dict)
    return co_occurrence_dict


def update_train_co_occurrence(con, cursor, max_co_occurrence):
    for i, j in max_co_occurrence.items():
        sql = 'update DF_ID_TRAIN SET MAX_CO_OCCURRENCE=' + str(j) + " where GRBM='" + str(i) + "'"
        DBOptions.exeSQL(sql, cursor, con)

def update_test_co_occurrence(con, cursor, max_co_occurrence):
    for i, j in max_co_occurrence.items():
        sql = 'update DF_ID_TEST SET MAX_CO_OCCURRENCE=' + str(j) + " where GRBM='" + str(i) + "'"
        DBOptions.exeSQL(sql, cursor, con)


def plot_co_occurrence(cursor):
    sql = 'select is_fraud,max_co_occurrence from DF_ID_TRAIN order by max_co_occurrence desc'
    co_occurrence_data = DBOptions.getSQL(sql, cursor=cursor)
    fraud_data = {}
    regular_data = {}
    for i in range(len(co_occurrence_data)):
        if co_occurrence_data[i][0] == 1:
            fraud_data[i]=co_occurrence_data[i][1]
        else:
            regular_data[i]=co_occurrence_data[i][1]
    plt.bar(list(fraud_data.keys()), list(fraud_data.values()), color='black')
    plt.bar(list(regular_data.keys()), list(regular_data.values()), color='y')
    plt.savefig('./data_png/co_occurrence.png',dpi=1500)


if __name__ == '__main__':
    start = time.clock()
    connect = DBOptions.connect()
    db_cursor = connect.cursor()
    # test_id_dict = load_test_list()
    # co_occurrence_dict = get_test_co_corrence(test_id_dict)
    # update_test_co_occurrence(connect, db_cursor, co_occurrence_dict)
    plot_co_occurrence(cursor=db_cursor)
    print(time.clock() - start)
