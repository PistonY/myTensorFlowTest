import pandas as pd


off_test = pd.read_csv(r'C:\Users\soulf\PycharmProjects\TC-data\o2o\ccf_offline_stage1_test_revised.csv',header=0)
off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']

f1 = off_test.apply(lambda x: x)


#特征读取
coupon_fea = pd.read_csv('data1/coupon_feature3.csv', header=0)
merchant_fea = pd.read_csv('data1/merchant_feature3.csv', header=0)
user_off_fea = pd.read_csv('data1/off_feature3.csv', header=0)
user_on_fea = pd.read_csv('data1/on_feature3.csv', header=0)
user_coupon_fea = pd.read_csv('data1/user_coupon_feature3.csv', header=0)
user_merchant_fea = pd.read_csv('data1/user_merchant_feature3.csv', header=0)


#融合样本
# coupon_fea = pd.read_csv('data1/coupon_feature1.csv', header=0)
coupon_fea['coupon_id'] = coupon_fea['coupon_id'].astype('str')

model_merge = pd.merge(f1, coupon_fea, on=['coupon_id'], how='left')


# merchant_fea = pd.read_csv('data1/merchant_feature1.csv', header=0)

model_merge = pd.merge(model_merge, merchant_fea, on=['merchant_id'], how='left')

# user_off_fea = pd.read_csv('data1/off_feature1.csv', header=0)

model_merge = pd.merge(model_merge, user_off_fea, on=['user_id'], how='left')


# user_on_fea = pd.read_csv('data1/on_feature1.csv', header=0)

model_merge = pd.merge(model_merge, user_on_fea, on=['user_id'], how='left')


other_fea = pd.read_csv('data1/other_feature.csv', header=0)
other_fea['coupon_id'] = other_fea['coupon_id'].astype('str')
other_fea = other_fea.drop(['date_received'], axis=1)

model_merge = pd.merge(model_merge, other_fea, on=['user_id', 'coupon_id'], how='left')


# user_coupon_fea = pd.read_csv('data1/user_coupon_feature1.csv', header=0)
user_coupon_fea['coupon_id'] = user_coupon_fea['coupon_id'].astype('str')

model_merge = pd.merge(model_merge, user_coupon_fea, on=['user_id', 'coupon_id'], how='left')


# user_merchant_fea = pd.read_csv('data1/user_merchant_feature1.csv', header=0)

model_merge = pd.merge(model_merge, user_merchant_fea, on=['user_id', 'merchant_id'], how='left')


model_merge.to_csv('train_data/pred_merge.csv',index=None)