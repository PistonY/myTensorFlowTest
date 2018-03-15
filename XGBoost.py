import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import MinMaxScaler

train1 = pd.read_csv('train_data/train_set1.csv')
test1 = pd.read_csv('train_data/test_merge1.csv')
train2 = pd.read_csv('train_data/train_set2.csv')
test2 = pd.read_csv('train_data/test_merge2.csv')

train3 = pd.read_csv('train_data/train_set3.csv')

dataset = pd.read_csv('train_data/pred_merge.csv')
dataset_preds = dataset[['user_id','coupon_id','date_received']]
dataset_x = dataset.drop(['user_id','coupon_id','merchant_id', 'discount_rate', 'date_received',
                          'all_merht_rate', 'all_coupon_rate'],axis=1)

train = pd.concat([train1, train2], axis=0)
tests = pd.concat([test1, test2], axis=0)

train.drop_duplicates(inplace=True)
tests.drop_duplicates(inplace=True)

y = train.lable
X = train.drop(['lable', 'all_merht_rate', 'all_coupon_rate', 'day_gap_before', 'day_gap_after'], axis=1)
# X = X.iloc[:, :57]

test_y = tests.lable
test_X = tests.drop(['lable', 'all_merht_rate', 'all_coupon_rate', 'day_gap_before', 'day_gap_after'], axis=1)
# test_X = test_X.iloc[:, :57]

xgb_train = xgb.DMatrix(X, label=y)
xgb_test = xgb.DMatrix(test_X, label=test_y)
dataset = xgb.DMatrix(dataset_x)

params = {
    'booster': 'gbtree',
    'objective': 'rank:pairwise',
    'eval_metric': 'auc',
    'gamma': 0,
    'min_child_weight': 1.1,
    'max_depth': 8,
    'lambda': 50,
    'base_score': 0.11,
    'min_child_weight': 100,
    'kvDelimiter': ':',
    'subsample': 0.4,
    'colsample_bytree': 0.6,
    'colsample_bylevel': 0.7,
    'eta': 0.08,
    'tree_method': 'exact',
    'nthread': 12
}

num_rounds = 2000
watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
model = xgb.train(params, xgb_train, num_rounds, watchlist)

# dataset_preds['label'] = model.predict(dataset)
# dataset_preds.label = MinMaxScaler().fit_transform(dataset_preds.label.reshape(-1, 1))
# dataset_preds.sort_values(by=['coupon_id','label'],inplace=True)
# dataset_preds.to_csv("predData/xgb_preds.csv",index=None,header=None)

feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key, value) in feature_score:
    fs.append("{0},{1}\n".format(key, value))

with open('predData/xgb_feature_score.csv', 'w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)


