# %%
import lightgbm
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import timedelta
sns.set()

# %%
df = pd.read_csv('../data/stock_days_fq/sh.600157.csv')
df=df[(df['volume'] != 0) & (df['close'] != 0)]    #去除没有交易的数据，volume为0 判断
df = df.reset_index(drop=True)
date_ori = pd.to_datetime(df.iloc[:, 1]).tolist()
df.head()

# %%
minmax = MinMaxScaler().fit(df.iloc[:, 5].values.reshape((-1,1)))
close_normalize = minmax.transform(df.iloc[:, 5].values.reshape((-1,1))).reshape((-1))

# %%
close_normalize.shape

# %%
class encoder:
    def __init__(self, input_, dimension = 2, learning_rate = 0.01, hidden_layer = 256, epoch = 20):
        input_size = input_.shape[1]
        self.X = tf.placeholder("float", [None, input_.shape[1]])
    
        weights = {
        'encoder_h1': tf.Variable(tf.random_normal([input_size, hidden_layer])),
        'encoder_h2': tf.Variable(tf.random_normal([hidden_layer, dimension])),
        'decoder_h1': tf.Variable(tf.random_normal([dimension, hidden_layer])),
        'decoder_h2': tf.Variable(tf.random_normal([hidden_layer, input_size])),
        }
    
        biases = {
        'encoder_b1': tf.Variable(tf.random_normal([hidden_layer])),
        'encoder_b2': tf.Variable(tf.random_normal([dimension])),
        'decoder_b1': tf.Variable(tf.random_normal([hidden_layer])),
        'decoder_b2': tf.Variable(tf.random_normal([input_size])),
        }
    
        first_layer_encoder = tf.nn.sigmoid(tf.add(tf.matmul(self.X, weights['encoder_h1']), biases['encoder_b1']))
        self.second_layer_encoder = tf.nn.sigmoid(tf.add(tf.matmul(first_layer_encoder, weights['encoder_h2']), biases['encoder_b2']))
        first_layer_decoder = tf.nn.sigmoid(tf.add(tf.matmul(self.second_layer_encoder, weights['decoder_h1']), biases['decoder_b1']))
        second_layer_decoder = tf.nn.sigmoid(tf.add(tf.matmul(first_layer_decoder, weights['decoder_h2']), biases['decoder_b2']))
        self.cost = tf.reduce_mean(tf.pow(self.X - second_layer_decoder, 2))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.cost)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        
        for i in range(epoch):
            last_time = time.time()
            _, loss = self.sess.run([self.optimizer, self.cost], feed_dict={self.X: input_})
            if (i + 1) % 10 == 0:
                print('epoch:', i + 1, 'loss:', loss, 'time:', time.time() - last_time)
    
    def encode(self, input_):
        return self.sess.run(self.second_layer_encoder, feed_dict={self.X: input_})

# %%
tf.reset_default_graph()
# from tensorflow.python.framework import ops
# ops.reset_default_graph()
Encoder=encoder(close_normalize.reshape((-1,1)), 32, 0.01, 128, 80)
thought_vector = Encoder.encode(close_normalize.reshape((-1,1)))
thought_vector.shape

# %%
from sklearn.ensemble import *
ada = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)
bagging = BaggingRegressor(n_estimators=500)
et = ExtraTreesRegressor(n_estimators=500)
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1)
rf = RandomForestRegressor(n_estimators=500)
param = {'num_leaves': 31,
        'min_data_in_leaf': 20,
        'objective': 'binary',
        'learning_rate': 0.06,
        "boosting": "gbdt",
#          "bagging_freq": 1,
#          "bagging_seed": 11,
        "metric": 'None',
        "verbosity": -1}
num_round =50


# %%
train_vector = thought_vector[:-252, :]
test_vector = thought_vector[-252:, :]

train_y = close_normalize[1:-251]
test_y = close_normalize[-252:]

ada.fit(train_vector, train_y)
bagging.fit(train_vector, train_y)
et.fit(train_vector, train_y)
gb.fit(train_vector, train_y)
rf.fit(train_vector, train_y)


train_data = lightgbm.Dataset(train_vector, train_y)

light = lightgbm.train(param, train_data, num_round, verbose_eval=30)
light_pred = light.predict(test_vector, num_iteration=light.best_iteration)

# %%
plt.bar(np.arange(32), ada.feature_importances_)
plt.title('ada boost important feature')
plt.show()
plt.savefig('ada.png')

# %%
plt.bar(np.arange(32), et.feature_importances_)
plt.title('et important feature')
plt.show()
plt.savefig('et.png')

# %%
plt.bar(np.arange(32), gb.feature_importances_)
plt.title('gb important feature')
plt.show()
plt.savefig('gb.png')

# %%
plt.bar(np.arange(32), rf.feature_importances_)
plt.title('rf important feature')
plt.show()
plt.savefig('rf.png')

# bagging.n_features_
# plt.bar(np.arange(32), bagging.feature_importances_)
# plt.title('bagging important feature')
# plt.show()
# plt.savefig('bagging.png')


# %%
ada_pred=ada.predict(train_vector)
bagging_pred=bagging.predict(train_vector)
et_pred=et.predict(train_vector)
gb_pred=gb.predict(train_vector)
rf_pred=rf.predict(train_vector)


# %%
ada_actual = np.hstack([close_normalize[0],ada_pred[:-1]])
bagging_actual = np.hstack([close_normalize[0],bagging_pred[:-1]])
et_actual = np.hstack([close_normalize[0],et_pred[:-1]])
gb_actual = np.hstack([close_normalize[0],gb_pred[:-1]])
rf_actual = np.hstack([close_normalize[0],rf_pred[:-1]])
stack_predict = np.vstack([ada_actual,bagging_actual,et_actual,gb_actual,rf_actual,close_normalize[:-252]]).T
corr_df = pd.DataFrame(stack_predict)

# %%
sns.heatmap(corr_df.corr(), annot=True)
plt.show()
plt.savefig('heatmap.png')

# %% [markdown]
# # Wow, I do not expect this heatmap. Totally a heat!

# %%
import xgboost as xgb
params_xgd = {
    'max_depth': 7,
    'objective': 'reg:logistic',
    'learning_rate': 0.05,
    'n_estimators': 10000
    }
# train_Y = close_normalize[1:]
clf = xgb.XGBRegressor(**params_xgd)
clf.fit(stack_predict[:-1,:],train_y[:-1], eval_set=[(stack_predict[:-1,:],train_y[:-1])], 
        eval_metric='rmse', early_stopping_rounds=20, verbose=False)

#计算所有的预测值
ada_pred=ada.predict(test_vector)
bagging_pred=bagging.predict(test_vector)
et_pred=et.predict(test_vector)
gb_pred=gb.predict(test_vector)
rf_pred=rf.predict(test_vector)


# %%
ada_actual = np.hstack([close_normalize[0],ada_pred[:-1]])
bagging_actual = np.hstack([close_normalize[0],bagging_pred[:-1]])
et_actual = np.hstack([close_normalize[0],et_pred[:-1]])
gb_actual = np.hstack([close_normalize[0],gb_pred[:-1]])
rf_actual = np.hstack([close_normalize[0],rf_pred[:-1]])
stack_predict = np.vstack([ada_actual,bagging_actual,et_actual,gb_actual,rf_actual,close_normalize[-252:]]).T
# %%
xgb_pred = clf.predict(stack_predict)
xgb_actual = np.hstack([close_normalize[0],xgb_pred[:-1]])
date_original=pd.Series(date_ori[-252:]).dt.strftime(date_format='%Y-%m-%d').tolist()

# %%
def reverse_close(array):
    return minmax.inverse_transform(array.reshape((-1,1))).reshape((-1))

# %%
plt.figure(figsize = (15,6))
x_range = np.arange(df['close'][-252:].shape[0])
plt.plot(x_range, df['close'][-252:], label = 'Real Close')
# plt.plot(x_range, reverse_close(light_pred), label = 'light Close')

# plt.plot(x_range, reverse_close(ada_pred), label = 'ada Close')
# plt.plot(x_range, reverse_close(bagging_pred), label = 'bagging Close')
# plt.plot(x_range, reverse_close(et_pred), label = 'et Close')
plt.plot(x_range, reverse_close(gb_pred), label = 'gb Close')
# plt.plot(x_range, reverse_close(rf_pred), label = 'rf Close')
plt.plot(x_range, reverse_close(xgb_pred), label = 'xgb stacked Close')
plt.legend()
plt.xticks(x_range[::50], date_original[::50])
plt.title('stacked')
plt.show()
plt.savefig('stacked_gb.png')

# %%
ada_list = ada_pred.tolist()
bagging_list = bagging_pred.tolist()
et_list = et_pred.tolist()
gb_list = gb_pred.tolist()
rf_list = rf_pred.tolist()
xgb_list = xgb_pred.tolist()
def predict(count, history = 5):
    for i in range(count):
        roll = np.array(xgb_list[-history+i:])
        thought_vector = Encoder.encode(roll.reshape((-1,1)))
        ada_pred=ada.predict(thought_vector)
        bagging_pred=bagging.predict(thought_vector)
        et_pred=et.predict(thought_vector)
        gb_pred=gb.predict(thought_vector)
        rf_pred=rf.predict(thought_vector)
        ada_list.append(ada_pred[-1])
        bagging_list.append(bagging_pred[-1])
        et_list.append(et_pred[-1])
        gb_list.append(gb_pred[-1])
        rf_list.append(rf_pred[-1])
        ada_actual = np.hstack([xgb_list[-history+i],ada_pred[:-1]])
        bagging_actual = np.hstack([xgb_list[-history+i],bagging_pred[:-1]])
        et_actual = np.hstack([xgb_list[-history+i],et_pred[:-1]])
        gb_actual = np.hstack([xgb_list[-history+i],gb_pred[:-1]])
        rf_actual = np.hstack([xgb_list[-history+i],rf_pred[:-1]])
        stack_predict = np.vstack([ada_actual,bagging_actual,et_actual,gb_actual,rf_actual,xgb_list[-history+i:]]).T
        xgb_pred = clf.predict(stack_predict)
        xgb_list.append(xgb_pred[-1])
        date_ori.append(date_ori[-1]+timedelta(days=1))

# %%
predict(6, history = 5)

# %%
plt.figure(figsize = (15,6))
x_range = np.arange(df['close'][-252:].shape[0])
x_range_future = np.arange(len(xgb_list))
plt.plot(x_range, df['close'][-252:], label = 'Real Close')
plt.plot(x_range_future, reverse_close(np.array(ada_list)), label = 'ada Close')
plt.plot(x_range_future, reverse_close(np.array(bagging_list)), label = 'bagging Close')
plt.plot(x_range_future, reverse_close(np.array(et_list)), label = 'et Close')
plt.plot(x_range_future, reverse_close(np.array(gb_list)), label = 'gb Close')
plt.plot(x_range_future, reverse_close(np.array(rf_list)), label = 'rf Close')
plt.plot(x_range_future, reverse_close(np.array(xgb_list)), label = 'xgb stacked Close')
plt.legend()
plt.xticks(x_range_future[::50], pd.Series(date_ori[-252:]).dt.strftime(date_format='%Y-%m-%d').tolist()[::50])
plt.title('stacked')
plt.show()
plt.savefig('stacked2.png')

# %%



