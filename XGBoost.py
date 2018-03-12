import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

train = pd.read_csv('train_data/model_merge1.csv')
tests = pd.read_csv('train_data/test_merge1.csv')

train_xy, val = train_test_split(train, test_size=0.3, random_state=1)
y = train_xy.lable
X = train_xy.drop(['lable'], axis=1)

val_y = val.lable
val_X = val.drop(['lable'], axis=1)

xgb_val = xgb.DMatrix(val_X, label=val_y)
xgb_train = xgb.DMatrix(X, label=y)
xgb_test = xgb.DMatrix(tests)

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'gamma': 0.1,
    'max_depth': 8,
    'max_leaf_nodes': 10,
    'scale_pos_weight': 1,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 0,
    'eta': 0.02,
    'eval_metric': 'auc'
}

plst = list(params.items())
num_rounds = 5000
watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]
model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)
model.save_model('models/xgb.model')

print("best best_ntree_limit",model.best_ntree_limit)