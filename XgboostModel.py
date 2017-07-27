import csv
import time

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

import DBOptions


def load_mlp_data(cursor):
    sql = 'select count(SXH),sum(YPFFSJE),sum(YPFSBJE),sum(ZLFFSJE),sum(ZLFSBJE),' \
          'MAX_CO_OCCURRENCE,is_fraud from DF_TRAIN,DF_ID_TRAIN ' \
          'where DF_TRAIN.grbm=DF_ID_TRAIN.GRBM group by DF_TRAIN.grbm,IS_FRAUD,MAX_CO_OCCURRENCE'
    mlp_data = DBOptions.getSQL(sql, cursor=cursor)
    scale = MinMaxScaler()
    mlp_data = scale.fit_transform(mlp_data)
    train_data, valid_data = train_test_split(mlp_data, test_size=0.2, random_state=0)
    train_x = train_data[:, :-1]
    train_y = train_data[:, -1]
    valid_x = valid_data[:, :-1]
    valid_y = valid_data[:, -1]
    return train_x, valid_x, train_y, valid_y


def load_test_data(cursor):
    sql = 'select DF_TEST.GRBM,count(SXH),sum(YPFFSJE),sum(YPFSBJE),sum(ZLFFSJE),sum(ZLFSBJE),' \
          'MAX_CO_OCCURRENCE from DF_TEST,DF_ID_TEST ' \
          'where DF_TEST.grbm=DF_ID_TEST.GRBM group by DF_TEST.grbm,MAX_CO_OCCURRENCE'
    test_x = DBOptions.getSQL(sql, cursor=cursor)
    test_id = [i[0] for i in test_x]
    scale = MinMaxScaler()
    test_x = scale.fit_transform(test_x)
    return test_x[:, 1:], test_id


def load_test_list():
    train_list = []
    with open('./origin_data/DF_ID_TEST.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            train_list.append(row[0])
    train_list = train_list[1:]
    return train_list


def calculate_recall(true_vec, predict_vec):
    predict_vec = predict_vec // (0.65 + 1e-3)
    recall = recall_score(true_vec, predict_vec)
    return recall


def calculate_precision(true_vec, predict_vec):
    predict_vec = predict_vec // (0.65 + 1e-3)
    precision = precision_score(true_vec, predict_vec)
    return precision


def calculate_f1score(true_vec, predict_vec):
    predict_vec = predict_vec // (0.65 + 1e-3)
    f_score = f1_score(true_vec, predict_vec)
    return f_score


def xgboost_model(train_x, valid_x, train_y, valid_y, test_x, test_id, test_list):
    classifier = XGBClassifier(base_score=0.5)
    classifier.fit(train_x, train_y)
    predict_y = classifier.predict(valid_x)
    valid_precision = calculate_precision(valid_y, predict_y)
    valid_recall = calculate_recall(valid_y, predict_y)
    valid_f1score = calculate_f1score(valid_y, predict_y)
    buf = 'precision:%f, recall:%f, f1score:%f' % (valid_precision, valid_recall, valid_f1score)
    print(buf)
    predict_y = classifier.predict(test_x)
    test_dict = {}
    for i in range(len(test_id)):
        test_dict[test_id[i]] = int(predict_y[i])
    print(test_dict)
    count=0
    for i in test_list:
        print(i + ',' + str(test_dict[i]))
        count+=test_dict[i]
    print(count)



if __name__ == '__main__':
    start = time.clock()
    connect = DBOptions.connect()
    db_cursor = connect.cursor()
    train_x, valid_x, train_y, valid_y = load_mlp_data(db_cursor)
    test_x, test_id = load_test_data(cursor=db_cursor)
    test_list = load_test_list()
    xgboost_model(train_x, valid_x, train_y, valid_y, test_x, test_id, test_list)
    print(time.clock() - start)
