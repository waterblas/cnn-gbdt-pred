import numpy as np
import math
import cPickle as pickle

# test sample percent
alpha = 0.3

user_feature_f = "../output/user_feature.txt"
merchant_feature_f = "../output/merchant_feature_offline.pkl"
user_merchant_feature_f = "../output/user_merchant_feature_offline.txt"
cnn_feature_f = "../output/cnn_features_offline.pkl"
merge_feature_f = "../output/features.pkl"

# 260811
user_features = {}
with open(user_feature_f, "r") as f0:
    _, sample_line = f0.readline().rstrip().split("\t")
    u_num = len(sample_line.split(','))
    for line in f0:
        user_id, features = line.rstrip().split("\t")
        user_features[user_id] = map(float, features.split(','))

with open(merchant_feature_f, 'r') as f:
    merchant_feature = pickle.load(f)
m_num = len(merchant_feature['941'])

data_u = set()
with open("../data/data_format1/train_format1.csv", 'r') as f:
    f.readline()
    for line in f:
        uid, merchant_id, _ = line.rstrip().split(',')
        data_u.add(uid)

num_data = len(data_u)
data_l = list(data_u)

idxs = np.random.permutation(num_data)
train_len = int(num_data * (1-alpha))

train_u = set()
test_u = set()

for idx in idxs[: train_len]:
    train_u.add(data_l[idx])

for idx in idxs[train_len:]:
    test_u.add(data_l[idx])

print "total user num: %s, train num: %s, test num: %s" % (num_data, len(train_u), len(test_u))

with open(cnn_feature_f, "r") as f:
    cnn_features, _, um = pickle.load(f)
    user_merchant_map = {um[i]:i for i in range(len(um))} 

with open(user_merchant_feature_f, "r") as f1:
    _, sample_line = f1.readline().rstrip().split("#")
    um_num = len(sample_line.split(','))
    f1.seek(0)
    train_num = 0
    test_num = 0
    for line in f1:
        user_feature, action_feature = line.rstrip().split("#")
        user_id, merchant_id, label, age, gender = user_feature.split(',')
        if label == "-1":
            continue
        if user_id in train_u:
            train_num += 1
        elif user_id in test_u:
            test_num += 1
    f1.seek(0)

    vec_len = u_num+m_num+um_num+ cnn_features.shape[1]
    train_x = np.zeros((train_num, vec_len))
    train_y = np.zeros(train_num)
    test_x = np.zeros((test_num, vec_len))
    test_y = np.zeros(test_num)
    train_idx = 0
    test_idx = 0
    for line in f1:
        user_feature, action_feature = line.rstrip().split("#")
        user_id, merchant_id, label, age, gender = user_feature.split(',')
        user_merchant_tag = "%s_m%s" % (user_id, merchant_id)
        um2id = user_merchant_map[user_merchant_tag]
        if label not in ['0', '1']:
            continue
        if age=='':
            age =0 
        if gender == '':
            gender = 2
        features = map(float, action_feature.split(','))
        features = features + user_features[user_id] + merchant_feature[merchant_id]
        #features = [math.log(1+ele) for ele in features]
        #features = [int(age), int(gender)] + features + list(cnn_features[um2id])
        features = features + list(cnn_features[um2id])
        if user_id in train_u:
            train_x[train_idx] = features
            train_y[train_idx] = int(label)
            train_idx += 1
        elif user_id in test_u:
            test_x[test_idx] = features
            test_y[test_idx] = int(label)
            test_idx += 1

with open(merge_feature_f, 'w') as f:
    pickle.dump([train_x, train_y, test_x, test_y], f, 2)
    
