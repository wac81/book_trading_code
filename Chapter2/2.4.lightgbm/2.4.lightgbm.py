import pandas as pd 
import numpy as np


def features_create(df: pd.DataFrame):

    df = df[['date','open','high','low','close','volume','turn','code']]

    #新增特征
    df['co'] = df['close']/df['open'] - 1
    df['ol'] = df['open']/df['low'] - 1
    df['oh'] = df['high']/df['open'] - 1
    df['cl'] = df['close']/df['low'] - 1
    df['ch'] = df['high']/df['close'] - 1
    df['lh'] = df['high']/df['low'] - 1
    df['cc'] = df['close']/df['close'].shift(1) - 1
    #
    df['ma5'] = df['cc'].rolling(5).mean()
    df['ma10'] = df['cc'].rolling(10).mean()
    df['ma20'] = df['cc'].rolling(20).mean()
    df['ma30'] = df['cc'].rolling(30).mean()

    df['ewm5'] = df['cc'].ewm(5).mean()
    df['ewm10'] = df['cc'].ewm(10).mean()
    df['ewm20'] = df['cc'].ewm(20).mean()
    df['ewm30'] = df['cc'].ewm(30).mean()

    df.fillna(method='bfill', inplace=True)
    return df

def Y_create(df: pd.DataFrame):
    df['label'] = np.where(df['close'].shift(-1)/df['close'] > 1, 1,0)
    df.fillna(method='bfill', inplace=True)
    return df


def train_lightgbm(train, train_label, val, val_label, model_path=None):
    # 模型训练及评价
    import lightgbm as lgb

    param = {'num_leaves': 31,
            'min_data_in_leaf': 20,
            'objective': 'binary',
            'learning_rate': 0.06,
            "boosting": "gbdt",
    #          "bagging_freq": 1,
    #          "bagging_seed": 11,
            "metric": 'None',
            "verbosity": -1}
    train_data = lgb.Dataset(train, train_label)
    val_data = lgb.Dataset(val, val_label)
    num_round =60

    model = lgb.train(param, train_data, num_round)
    if model_path is not None:
        model.save_model(model_path)
    return model

def predict_lightgbm(model, test):
    test_result_lgb = model.predict(test, num_iteration=model.best_iteration)
    return test_result_lgb

def eval_model(test_result_lgb, test_label,thresh_hold = 0.5):
    from sklearn import metrics
    
    # oof_test_final = (test_lgb >= 0.69) & (test_lgb <=0.7)
    oof_test_final = test_result_lgb >= thresh_hold
    print(metrics.accuracy_score(test_label, oof_test_final))
    print(metrics.confusion_matrix(test_label, oof_test_final))
    tp = np.sum(((oof_test_final == 1) & (test_label == 1)))
    pp = np.sum(oof_test_final == 1)
    print('accuracy:%.3f'% (tp/(pp)))
    
    return oof_test_final


def backtest_lightgbm(oof_test_final, test_result_lgb):

    test_date_min = '2021-04-01'
    test_date_max = '2022-02-23'
    test_data_idx = (df['date'] >= test_date_min) & (df['date'] <= test_date_max)

    test_postive_idx = np.argwhere(oof_test_final == 1).reshape(-1)
    test_all_idx = np.argwhere(np.array(test_data_idx)).reshape(-1)
    choose_fea.append('date')
    choose_fea.append('code')

    tmp_df = df[choose_fea].iloc[test_all_idx[test_postive_idx]].reset_index()
    tmp_df['label_prob'] = test_result_lgb[test_postive_idx]


    tmp_df['is_limit_up'] = tmp_df['close'] == tmp_df['high']

    buy_df = tmp_df[(tmp_df['is_limit_up']==False)].reset_index()
    buy_df.drop(['index', 'level_0'], axis=1, inplace=True)

    from imp import reload
    import account
    reload(account)
    money_init = 100000
    account = account.Account(money_init, max_hold_period=20, stop_loss_rate=-0.07, stop_profit_rate=0.12)
    df_copy = df.copy()

    # 读取指数信息
    index_df = pd.read_csv('../../data/sh.csv')
    tmp_idx = (index_df['date'] >= test_date_min) & (index_df['date'] <= test_date_max)
    index_df = index_df.loc[tmp_idx].reset_index()
    index_df.drop('index', axis=1, inplace=True)

    account.BackTest(buy_df, df_copy, index_df, buy_price='open')


if __name__ == '__main__':


    # df = pd.read_csv('../data/sh.601225.csv')
    df = pd.read_csv('../../data/stock_days_fq/sh.600157.csv')


    choose_fea = [
        'open',
        'high','low',
        'close',
        # 'volume',
        'turn',
        'cc',
        'co','ol','oh','cl','ch','lh',
        'ma5',
        'ma10',
        # 'ma20',
        # 'ma30',
        'ewm5',
        'ewm10',
        # 'ewm20',

    ]
    #特征
    df = features_create(df)
    df = Y_create(df)

    #生成训练数据
    train_date_min = '2014-01-03'
    train_date_max = '2021-01-01'
    val_date_min = '2021-01-01'
    val_date_max = '2021-04-01'
    test_date_min = '2021-04-01'
    test_date_max = '2022-02-23'

    train_data_idx = (df['date'] >= train_date_min) & (df['date'] <= train_date_max) & (df['high']!=df['close'])   #涨停不处理
    val_data_idx = (df['date'] >= val_date_min) & (df['date'] <= val_date_max)
    test_data_idx = (df['date'] >= test_date_min) & (df['date'] <= test_date_max)
    


    train = df[train_data_idx][choose_fea]
    train_label = df[train_data_idx]['label'].values

    val = df[val_data_idx][choose_fea]
    val_label = df[val_data_idx]['label'].values 

    test = df[test_data_idx][choose_fea]
    test_label = df[test_data_idx]['label'].values



    #训练
    model = train_lightgbm(train, train_label, val, val_label, model_path='./lightgbm.model')

    #预测测试集
    test_result_lgb = predict_lightgbm(model, test)

    #评估
    oof_test_final = eval_model(test_result_lgb, test_label)

    #回测
    backtest_lightgbm(oof_test_final, test_result_lgb)