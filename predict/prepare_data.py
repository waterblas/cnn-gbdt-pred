import numpy as np
import math
import cPickle as pickle
import sys


def pre_model_data(user_features, merchant_feature_f, label_f, user_merchant_feature_f, cnn_feature_f, merge_feature_f, predict_mode):

    with open(merchant_feature_f, 'r') as f:
        merchant_feature = pickle.load(f)
    m_num = len(merchant_feature['941'])

    data_u = set()
    with open(label_f, 'r') as f:
        f.readline()
        for line in f:
            uid, merchant_id, _ = line.rstrip().split(',')
            data_u.add(uid)

    num_data = len(data_u)
    data_l = list(data_u)
    print "total user num: %s" % (num_data)

    with open(cnn_feature_f, "r") as f:
        cnn_features, _, um = pickle.load(f)
        user_merchant_map = {um[i]:i for i in range(len(um))} 

    with open(user_merchant_feature_f, "r") as f1:
        _, sample_line = f1.readline().rstrip().split("#")
        um_num = len(sample_line.split(','))
        f1.seek(0)
        train_num = 0
        for line in f1:
            user_feature, action_feature = line.rstrip().split("#")
            user_id, merchant_id, label, age, gender = user_feature.split(',')
            if label == "-1":
                continue
            if user_id in data_u:
                train_num += 1
        f1.seek(0)

        vec_len = u_num+m_num+um_num+ cnn_features.shape[1]
        train_x = np.zeros((train_num, vec_len))
        train_y = np.zeros(train_num)
        train_tag = []
        train_idx = 0
        for idx, line in enumerate(f1):
            if idx % 20000 == 0:
                sys.stdout.write(">")
                sys.stdout.flush()
            user_feature, action_feature = line.rstrip().split("#")
            user_id, merchant_id, label, age, gender = user_feature.split(',')
            user_merchant_tag = "%s_m%s" % (user_id, merchant_id)
            um2id = user_merchant_map[user_merchant_tag]
            if label == '-1' :
                continue
            if age=='':
                age =0 
            if gender == '':
                gender = 2
            if label == '':
                label = 2
            features = map(float, action_feature.split(','))
            features = features + user_features[user_id] + merchant_feature[merchant_id]
            features = features + list(cnn_features[um2id])
            if user_id in data_u:
                train_x[train_idx] = features
                train_y[train_idx] = int(label)
                train_tag.append("%s,%s" % (user_id, merchant_id))
                train_idx += 1
    with open(merge_feature_f, 'w') as f:
        if predict_mode:
            pickle.dump([train_x, train_tag], f, 2)
        else:
            pickle.dump([train_x, train_y], f, 2)


user_feature_f = "../output/user_feature.txt"
user_features = {}
with open(user_feature_f, "r") as f0:
    _, sample_line = f0.readline().rstrip().split("\t")
    u_num = len(sample_line.split(','))
    for line in f0:
        user_id, features = line.rstrip().split("\t")
        user_features[user_id] = map(float, features.split(','))

label_f = "../data/data_format1/train_format1.csv"
merchant_feature_f = "../output/merchant_feature_offline.pkl"
user_merchant_feature_f = "../output/user_merchant_feature_offline.txt"
cnn_feature_f = "../output/cnn_features_offline.pkl"
merge_feature_f = "../output/features_offline.pkl"

pre_model_data(user_features, merchant_feature_f, label_f, user_merchant_feature_f, cnn_feature_f, merge_feature_f, False)

label_f = "../data/data_format1/test_format1.csv"
merchant_feature_f = "../output/merchant_feature.pkl"
user_merchant_feature_f = "../output/user_merchant_feature.txt"
cnn_feature_f = "../output/cnn_features.pkl"
merge_feature_f = "../output/features.pkl"
    
pre_model_data(user_features, merchant_feature_f, label_f, user_merchant_feature_f, cnn_feature_f, merge_feature_f, True)

