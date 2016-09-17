from __future__ import print_function

from keras.models import Model, model_from_json
#from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import cPickle as pickle
from sklearn import metrics
import glob

with open('../model/mt.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

with open('../output/test.pkl', 'r') as f:
    x_test, y_test = pickle.load(f)
hdf5_files = glob.glob('../model/*.hdf5')
for weight_f in hdf5_files: 
    print('epoch:', weight_f.split('.')[-2])
    model.load_weights(weight_f)
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    prob = model.predict(x_test[1:], batch_size=50, verbose=1)
    prob = prob.flatten()

    fpr, tpr, thresholds = metrics.roc_curve(y_test, prob, pos_label=1)
    print(metrics.auc(fpr, tpr))

