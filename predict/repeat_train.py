import xgboost as xgb
import numpy as np
import sys

train_x , train_y = np.load("../output/features_offline.pkl")
dtrain = xgb.DMatrix(train_x, label=train_y)
evallist  = [(dtrain,'train')]

param = {
    'max_depth':6,
    'eta':0.2,
    'silent':1,
    'objective':'binary:logistic' ,
    'eval_metric' : 'auc',
    'scale_pos_weight': 0.06 / 0.94
    }

num_round = 35
bst = xgb.train(param, dtrain, num_round, evallist)
bst.save_model('../model/repeat.model')

