import pandas as pd
import numpy as np

off_train = pd.read_csv(r'C:\Users\soulf\PycharmProjects\TC-data\o2o\ccf_offline_stage1_train\ccf_offline_stage1_train.csv', header=0, keep_default_na=False)
off_train.columns=['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']

feature1 = off_train[(off_train.date >= '20160101')&(off_train.date <= '20160413')|((off_train.date == 'null')&(off_train.date_received >= '20160101')&(off_train.date_received <= '20160413'))]
test1 = off_train[(off_train.date_received >= '20160414')&(off_train.date_received <= '20160514')]

feature2 = off_train[(off_train.date >= '20160201')&(off_train.date <= '20160514')|((off_train.date == 'null')&(off_train.date_received >= '20160201')&(off_train.date_received <= '20160514'))]
test2 = off_train[(off_train.date_received >= '20160515')&(off_train.date_received <= '20160615')]

f1 = test2.apply(lambda x:x)
f1['lable'] = f1.apply(lambda x: 1 if x.date != 'null' else 0, axis=1)

f2 = f1.apply(lambda x: x)

#特征读取
coupon_test = pd.read_csv('data1/coupon_feature2.csv', header=0)
merchant_test = pd.read_csv('data1/merchant_feature2.csv', header=0)
user_off_test = pd.read_csv('data1/off_feature2.csv', header=0)
user_on_test = pd.read_csv('data1/on_feature2.csv', header=0)
user_coupon_test = pd.read_csv('data1/user_coupon_feature2.csv', header=0)
user_merchant_test = pd.read_csv('data1/user_merchant_feature2.csv', header=0)


#融合样本
# coupon_test = pd.read_csv('data1/coupon_testture1.csv', header=0)
coupon_test['coupon_id'] = coupon_test['coupon_id'].astype('str')

model_merge = pd.merge(f2, coupon_test, on=['coupon_id'], how='left')

# merchant_test = pd.read_csv('data1/merchant_testture1.csv', header=0)

model_merge = pd.merge(model_merge, merchant_test, on=['merchant_id'], how='left')

# user_off_test = pd.read_csv('data1/off_feature1.csv', header=0)

model_merge = pd.merge(model_merge, user_off_test, on=['user_id'], how='left')

# user_on_test = pd.read_csv('data1/on_feature1.csv', header=0)

model_merge = pd.merge(model_merge, user_on_test, on=['user_id'], how='left')

other_fea = pd.read_csv('data1/other_feature.csv', header=0)
other_fea['coupon_id'] = other_fea['coupon_id'].astype('str')
other_fea = other_fea.drop(['date_received'], axis=1)

model_merge = pd.merge(model_merge, other_fea, on=['user_id', 'coupon_id'], how='left')

# user_coupon_test = pd.read_csv('data1/user_coupon_testture1.csv', header=0)
user_coupon_test['coupon_id'] = user_coupon_test['coupon_id'].astype('str')

model_merge = pd.merge(model_merge, user_coupon_test, on=['user_id', 'coupon_id'], how='left')

# user_merchant_test = pd.read_csv('data1/user_merchant_testture1.csv', header=0)

model_merge = pd.merge(model_merge, user_merchant_test, on=['user_id', 'merchant_id'], how='left')


model_merge.drop(['user_id','merchant_id','coupon_id','discount_rate','date_received','date'], axis=1, inplace=True)

model_merge.drop_duplicates(inplace=True)
model_merge = model_merge.replace('null', np.nan)
model_merge.to_csv('train_data/test_merge2.csv',index=None)

