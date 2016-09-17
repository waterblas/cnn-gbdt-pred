from __future__ import print_function
from keras.models import Model, model_from_json
#from keras.callbacks import ModelCheckpoint, EarlyStopping
#from keras import backend as K
import numpy as np
import cPickle as pickle

with open('../model/encoder.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights('../model/encoder.h5')
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
with open('../output/serial_feature.pkl', 'r') as f:
    x_train, y_train = pickle.load(f)
represent_unit = model.predict(x_train[1:])
with open("../output/cnn_features.pkl", "w") as f:
    pickle.dump([represent_unit, y_train, x_train[0]], f, 2)

