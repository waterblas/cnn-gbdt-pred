import cPickle as pickle
import numpy as np

all_u = set()
with open("../data/data_format1/train_format1.csv", 'r') as f:
    f.readline()
    for line in f:
        uid, merchant_id, _ = line.rstrip().split(',')
        all_u.add(uid)

num_data = len(all_u)
data_l = list(all_u)

idxs = np.random.permutation(num_data)
train_len = int(num_data * 0.7)

train_u = set()
test_u = set()

for idx in idxs[: train_len]:
    train_u.add(data_l[idx])

for idx in idxs[train_len:]:
    test_u.add(data_l[idx])

print "total:%s, train:%s, test:%s\n" % (num_data, len(train_u), len(test_u))

with open("../output/serial_feature.pkl", "r")  as f:
    X, y_label = pickle.load(f)

um_arr, ag, merchant, brand, purchase_brand, purchase_cat  = X
um_train = []
um_test = []
ids_train = []
ids_test = []

for idx, ele in enumerate(um_arr):
    uid, _ = ele.split('_')
    if uid in train_u:
        um_train.append(ele)
        ids_train.append(idx)
    elif uid in test_u:
        um_test.append(ele)
        ids_test.append(idx)

train_x = [um_train, ag[ids_train], merchant[ids_train], brand[ids_train], purchase_brand[ids_train], purchase_cat[ids_train]]
test_x = [um_test, ag[ids_test], merchant[ids_test], brand[ids_test], purchase_brand[ids_test], purchase_cat[ids_test]]
train_y = y_label[ids_train]
test_y = y_label[ids_test]
        
with open("../output/train.pkl", "w") as f:
    pickle.dump((train_x, train_y), f, 2)

with open("../output/test.pkl", "w") as f:
    pickle.dump((test_x, test_y), f, 2)


