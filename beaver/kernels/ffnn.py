import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers

def gen_features(X):
    features = []

    X1 = X[:75000]
    X2 = X[75000:]

    for segment in (X1, X2):
        features.append(segment.min())
        features.append(np.quantile(segment, 0.001))

        for i in range(1,100):
            features.append(np.quantile(segment, i/100))

        features.append(np.quantile(segment, 0.999))
        features.append(segment.max())

        features.append(segment.mean())
        features.append(segment.std())
        features.append(segment.mad())
        features.append(segment.skew())
        features.append(segment.kurtosis())

        features.append(np.abs(segment).mean())
        features.append(np.abs(segment).std())
        features.append(np.sqrt(np.mean(segment**2)))

    return pd.Series(features)

train = pd.read_csv('../input/train.csv',
                    iterator=True,
                    chunksize=30_000,
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

X_train = pd.DataFrame()
y_train = pd.Series()

df = pd.DataFrame()
i = 0
for chunk in train:
    i += 1
    if not i%1000:
        print(i)

    df = df.append(chunk)
    if len(df) >= 150000:
        df = df[-150000:]
        ch = gen_features(df['acoustic_data'])
        X_train = X_train.append(ch, ignore_index=True)
        y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_tr = X_train[:18000]
X_val = X_train[18000:]
y_tr = y_train[:18000]
y_val = y_train[18000:]

scores = []
preds = []

for _ in range(5):
    NN_model = Sequential()

    NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1]))
    NN_model.add(BatchNormalization())
    NN_model.add(Activation('tanh'))
    NN_model.add(Dropout(0.25))

    NN_model.add(Dense(64, kernel_initializer='normal'))
    NN_model.add(BatchNormalization())
    NN_model.add(Activation('tanh'))
    NN_model.add(Dropout(0.25))

    NN_model.add(Dense(32, kernel_initializer='normal'))
    NN_model.add(Activation('tanh'))

    NN_model.add(Dense(1, kernel_initializer='normal'))
    NN_model.add(Activation('linear'))

    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    NN_model.fit(X_tr, y_tr, epochs=3, batch_size=32)

    cur_preds = NN_model.predict(X_val).clip(0,16.1)
    preds.append(cur_preds)

    cur_score = mean_absolute_error(y_val, cur_preds)
    scores.append(cur_score)
    print(cur_score)

print("========================")

print(np.mean(scores))

print("========================")

preds = np.hstack([p.reshape(-1,1) for p in preds])
preds = np.mean(preds, axis=1)
print(mean_absolute_error(y_val, preds))

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame()

for seg_id in submission.index:
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    ch = gen_features(seg['acoustic_data'])
    X_test = X_test.append(ch, ignore_index=True)

X_test = sc.transform(X_test)

preds = []

for _ in range(10):
    NN_model = Sequential()

    NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1]))
    NN_model.add(BatchNormalization())
    NN_model.add(Activation('tanh'))
    NN_model.add(Dropout(0.25))

    NN_model.add(Dense(64, kernel_initializer='normal'))
    NN_model.add(BatchNormalization())
    NN_model.add(Activation('tanh'))
    NN_model.add(Dropout(0.25))

    NN_model.add(Dense(32, kernel_initializer='normal'))
    NN_model.add(Activation('tanh'))

    NN_model.add(Dense(1, kernel_initializer='normal'))
    NN_model.add(Activation('linear'))

    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    NN_model.fit(X_train, y_train, epochs=3, batch_size=32)

    cur_preds = NN_model.predict(X_test).clip(0,16.1)
    preds.append(cur_preds)

preds = np.hstack([p.reshape(-1,1) for p in preds])
preds = np.mean(preds, axis=1)

submission['time_to_failure'] = preds
submission.to_csv('submission.csv')