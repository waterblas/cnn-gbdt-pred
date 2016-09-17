from __future__ import print_function
import numpy as np
import cPickle as pickle
import xgboost as xgb

modelfile = '../model/repeat.model'
outfile = 'submission_pre.csv'

with open("../output/features.pkl", "r") as f:
    dtest, name = pickle.load(f)
dtest = xgb.DMatrix(dtest)

print ('finish loading from pkl ')
bst = xgb.Booster(model_file = modelfile)
ypred = bst.predict(dtest)

with open(outfile, "w") as f:
    for i in range(len(ypred)):
        f.write("%s,%s\n" % (name[i], ypred[i]))

