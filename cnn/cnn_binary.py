from __future__ import print_function
from keras.layers.embeddings import Embedding
from keras.layers import Merge,merge, Flatten,Input
from keras.layers import Convolution1D, MaxPooling1D,Dense,AveragePooling1D
from keras.models import Model, model_from_json
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import cPickle as pickle


AGE_GENDER_DIM = 50
MERCHANT_DIM = 200
BRAND_DIM = 200
PURCHASE_BRAND_DIM = 200
PURCHASE_CATEGORY_DIM = 150

BRAND_SIZE = 80
PURCHASE_BRAND_SIZE = 30
PURCHASE_CATEGORY_SIZE = 30

#ag:37, m:1993, b:7894, pb:6552, pc:1141
AGE_GENDER_RANGE = 37 + 1
MERCHANT_RANGE = 1993 + 1
BRAND_RANGE = 7894 + 1
PURCHASE_BRAND_RANGE = 6552 + 1
PURCHASE_CATEGORY_RANGE = 1141 + 1

HDIM = 250

age_gender_input = Input(shape=(1,), dtype='int32', name='age_gender_input') 
age_gender_emb = Embedding(input_dim=AGE_GENDER_RANGE, output_dim=AGE_GENDER_DIM, init='uniform', input_length=1)(age_gender_input)

merchant_input = Input(shape=(1,), dtype='int32', name='merchant_input')
merchant_emb = Embedding(input_dim=MERCHANT_RANGE, output_dim=MERCHANT_DIM, init='uniform')(merchant_input)

brand_input = Input(shape=(BRAND_SIZE,), dtype='int32', name='brand_input')
brand_emb = Embedding(input_dim=BRAND_RANGE, output_dim=BRAND_DIM, init='uniform')(brand_input)

purchase_brand_input = Input(shape=(PURCHASE_BRAND_SIZE,), dtype='int32', name='purchase_brand_input')
purchase_brand_emb = Embedding(input_dim=PURCHASE_BRAND_RANGE, output_dim=PURCHASE_BRAND_DIM, init='uniform')(purchase_brand_input)

purchase_category_input = Input(shape=(PURCHASE_CATEGORY_SIZE,), dtype='int32', name='purchase_category_input')
purchase_category_emb = Embedding(input_dim=PURCHASE_CATEGORY_RANGE, output_dim=PURCHASE_CATEGORY_DIM, init='uniform')(purchase_category_input)

w0 = Convolution1D(200, 1, input_shape=(BRAND_SIZE, BRAND_DIM), activation='relu', border_mode='same')(brand_emb)
w0 = MaxPooling1D(BRAND_SIZE)(w0) 
w0 = Flatten()(w0)
w0 = Dense(200)(w0)
w0 = Dropout(0.2)(w0) 
w_hidden = Activation('relu')(w0)

e0 = Convolution1D(200, 1, input_shape=(PURCHASE_BRAND_SIZE, PURCHASE_BRAND_DIM), activation='relu', border_mode='same')(purchase_brand_emb)
e0 = MaxPooling1D(PURCHASE_BRAND_SIZE)(e0) 
e0 = Flatten()(e0)
e0 = Dense(200)(e0)
e0 = Dropout(0.2)(e0) 
e_hidden = Activation('relu')(e0)

r0 = Convolution1D(200, 1, input_shape=(PURCHASE_CATEGORY_SIZE, PURCHASE_CATEGORY_DIM), activation='relu', border_mode='same')(purchase_category_emb)
r0 = MaxPooling1D(PURCHASE_CATEGORY_SIZE)(r0) 
r0 = Flatten()(r0)
r0 = Dense(200)(r0)
r0 = Dropout(0.2)(r0) 
r_hidden = Activation('relu')(r0)

q_hidden = Flatten()(merchant_emb)
t_hidden = Flatten()(age_gender_emb)
two_unit = merge([t_hidden, q_hidden, w_hidden, e_hidden, r_hidden], mode='concat', concat_axis=1)
l2_layer = Dense(HDIM, activation='tanh')(two_unit)
out = Dense(1,activation='sigmoid')(l2_layer)

model = Model([age_gender_input, merchant_input, brand_input, purchase_brand_input, purchase_category_input], out)
model.compile(loss='binary_crossentropy',
    #optimizer='adagrad',
    optimizer='adadelta',
    metrics=['accuracy'])
with open('../model/mt.json', 'w') as f:
    f.write(model.to_json())
# Train
#early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpointer = ModelCheckpoint(filepath="../model/mt.{epoch:02d}.hdf5", verbose=1, save_best_only=False, monitor='val_acc')
#with open('../output/train.pkl', 'r') as f:
with open('../output/serial_feature.pkl', 'r') as f:
    x_train, y_train = pickle.load(f)

model.fit(x_train[1:], y_train,
    nb_epoch=5,batch_size=50,
    shuffle=True,
    class_weight={0:0.06, 1:0.94},
    validation_split=0, callbacks=[checkpointer])

encoder = Model([age_gender_input, merchant_input, brand_input, purchase_brand_input, purchase_category_input], l2_layer)
with open('../model/encoder.json', 'w') as f:
    f.write(encoder.to_json())
encoder.save_weights('../model/encoder.h5')

