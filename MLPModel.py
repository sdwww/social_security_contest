import csv
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import DBOptions


def load_mlp_data(cursor):
    sql = 'select count(SXH),sum(YPFFSJE),sum(YPFSBJE),sum(ZLFFSJE),sum(ZLFSBJE),' \
          'MAX_CO_OCCURRENCE,is_fraud from DF_TRAIN,DF_ID_TRAIN ' \
          'where DF_TRAIN.grbm=DF_ID_TRAIN.GRBM group by DF_TRAIN.grbm,IS_FRAUD,MAX_CO_OCCURRENCE'
    mlp_data = DBOptions.getSQL(sql, cursor=cursor)
    scale = MinMaxScaler()
    mlp_data = scale.fit_transform(mlp_data)
    train_data, valid_data = train_test_split(mlp_data, test_size=0.2, random_state=0)
    new_train_data = []
    for i in train_data:
        if i[-1] == 0:
            if np.random.randint(1, 20) == 1:
                new_train_data.append(i)
        else:
            new_train_data.append(i)
    new_train_data = np.array(new_train_data)
    train_x = new_train_data[:, :-1]
    train_y = new_train_data[:, -1].reshape([len(new_train_data), 1])
    valid_x = valid_data[:, :-1]
    valid_y = valid_data[:, -1].reshape([len(valid_data), 1])
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
            train_list.append(row)
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


def mlp_model(train_x, valid_x, train_y, valid_y, test_x, test_id, test_list):
    x = tf.placeholder(tf.float32, [None, 6])
    y = tf.placeholder(tf.float32, [None, 1])

    w1 = tf.Variable(tf.random_normal([6, 50], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([50], stddev=0.01))

    h1 = tf.add(tf.matmul(x, w1), b1)

    w2 = tf.Variable(tf.random_normal([50, 1], stddev=0.01))
    b2 = tf.Variable(tf.random_normal([1], stddev=0.01))

    o = tf.nn.sigmoid(tf.add(tf.matmul(h1, w2), b2))

    loss = -tf.reduce_sum(y * tf.log(o) + (1 - y) * tf.log(1 - o))
    optimizer = tf.train.AdadeltaOptimizer(0.05)
    optimize = optimizer.minimize(loss=loss)

    n_epochs = 200
    batch_size = 100
    n_train_batches = int(np.ceil(float(train_x.shape[0])) / float(batch_size))
    n_valid_batches = int(np.ceil(float(valid_x.shape[0])) / float(batch_size))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(n_epochs):
            loss_vec = []
            idx = np.arange(train_x.shape[0])
            for i in range(n_train_batches):
                batch_idx = np.random.choice(idx, size=batch_size, replace=False)
                batch_x = train_x[batch_idx]
                batch_y = train_y[batch_idx]
                _, mlp_loss = sess.run([optimize, loss],
                                       feed_dict={x: batch_x, y: batch_y})
                loss_vec.append(mlp_loss)

            idx = np.arange(len(valid_x))
            valid_precision_vec = []
            valid_recall_vec = []
            valid_f1score_vec = []
            for i in range(n_valid_batches):
                batch_idx = np.random.choice(idx, size=batch_size, replace=False)
                batch_x = valid_x[batch_idx]
                batch_y = valid_y[batch_idx]
                predict_y = sess.run(o, feed_dict={x: batch_x})
                valid_precision = calculate_precision(batch_y, predict_y)
                valid_recall = calculate_recall(batch_y, predict_y)
                valid_f1score = calculate_f1score(batch_y, predict_y)
                valid_precision_vec.append(valid_precision)
                valid_recall_vec.append(valid_recall)
                valid_f1score_vec.append(valid_f1score)
            buf = 'Epoch:%d, loss:%f, precision:%f, recall:%f, f1score:%f' % (
                epoch, np.mean(loss_vec), np.mean(valid_precision_vec),
                np.mean(valid_recall_vec), np.mean(valid_f1score_vec))
            print(buf)
        predict_y = sess.run(o, feed_dict={x: test_x}) // (0.65 + 1e-3)
        test_dict = {}
        for i in range(len(test_id)):
            test_dict[test_id[i]] = int(predict_y[i][0])
        for i in test_list:
            print(i + ',' + str(test_dict[i]))


if __name__ == '__main__':
    start = time.clock()
    connect = DBOptions.connect()
    db_cursor = connect.cursor()
    train_x, valid_x, train_y, valid_y = load_mlp_data(db_cursor)
    test_x, test_id = load_test_data(cursor=db_cursor)
    test_list = load_test_list()
    mlp_model(train_x, valid_x, train_y, valid_y, test_x, test_id, test_list)
    print(time.clock() - start)
