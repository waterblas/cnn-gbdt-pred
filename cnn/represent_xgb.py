import xgboost as xgb
import numpy as np
import cPickle as pickle
import sys
from sklearn.cross_validation import train_test_split

if len(sys.argv) < 2:
    dtrain = xgb.DMatrix("train.buffer")
    dtest = xgb.DMatrix("test.buffer")
elif sys.argv[1] == 'r':
    with open("../output/cnn_features.pkl", "r") as f:
        X, Y, _ = pickle.load(f)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.33, random_state=42)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtrain.save_binary("train.buffer")
    dtest = xgb.DMatrix(test_x, label=test_y)
    dtest.save_binary("test.buffer")
evallist  = [(dtrain,'train'), (dtest, 'eval')]

param = {
    'max_depth':6,
    'eta':0.1,
    'silent':1,
    'objective':'binary:logistic' ,
    'eval_metric' : 'auc',
    'scale_pos_weight': 0.06 / 0.94
    }

num_round = 150
bst = xgb.train(param, dtrain, num_round, evallist)

