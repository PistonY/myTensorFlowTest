import pandas as pd

off_train = pd.read_csv(r'C:\Users\soulf\PycharmProjects\TC-data\o2o\ccf_offline_stage1_train\ccf_offline_stage1_train.csv', header=0, keep_default_na=False)
off_train.columns=['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']

feature1 = off_train[(off_train.date >= '20160101')&(off_train.date <= '20160413')|((off_train.date == 'null')&(off_train.date_received >= '20160101')&(off_train.date_received <= '20160413'))]
test1 = off_train[(off_train.date_received >= '20160414')&(off_train.date_received <= '20160514')]


f1 = feature1[((feature1.coupon_id != 'null')&(feature1.date == 'null'))|((feature1.coupon_id != 'null')&(feature1.date != 'null'))]
f1['lable'] = f1.apply(lambda x: 0 if (x.coupon_id != 'null')&(x.date == 'null') else 1, axis=1)

# 正样本
samp = 35000
f2 = f1[f1.lable == 1]
f2 = f2.sample(samp)


# 负样本
f3 = f1[f1.lable == 0]
f3 = f3.sample(8 * samp)


#融合样本
coupon_fea = pd.read_csv('data/coupon_feature1.csv', header=0)
coupon_fea['coupon_id'] = coupon_fea['coupon_id'].astype('str')

t_model_merge = pd.merge(f2, coupon_fea, on=['coupon_id'], how='left')
f_model_merge = pd.merge(f3, coupon_fea, on=['coupon_id'], how='left')

merchant_fea = pd.read_csv('data/merchant_feature1.csv', header=0)

t_model_merge = pd.merge(t_model_merge, merchant_fea, on=['merchant_id'], how='left')
f_model_merge = pd.merge(f_model_merge, merchant_fea, on=['merchant_id'], how='left')

user_off_fea = pd.read_csv('data/off_feature1.csv', header=0)

t_model_merge = pd.merge(t_model_merge, user_off_fea, on=['user_id'], how='left')
f_model_merge = pd.merge(f_model_merge, user_off_fea, on=['user_id'], how='left')

user_on_fea = pd.read_csv('data/on_feature1.csv', header=0)

t_model_merge = pd.merge(t_model_merge, user_on_fea, on=['user_id'], how='left')
f_model_merge = pd.merge(f_model_merge, user_on_fea, on=['user_id'], how='left')

other_fea = pd.read_csv('data/other_feature1.csv', header=0)
other_fea['coupon_id'] = other_fea['coupon_id'].astype('str')
other_fea = other_fea.drop(['date_received'], axis=1)

t_model_merge = pd.merge(t_model_merge, other_fea, on=['user_id', 'coupon_id'], how='left')
f_model_merge = pd.merge(f_model_merge, other_fea, on=['user_id', 'coupon_id'], how='left')

user_coupon_fea = pd.read_csv('data/user_coupon_feature1.csv', header=0)
user_coupon_fea['coupon_id'] = user_coupon_fea['coupon_id'].astype('str')

t_model_merge = pd.merge(t_model_merge, user_coupon_fea, on=['user_id', 'coupon_id'], how='left')
f_model_merge = pd.merge(f_model_merge, user_coupon_fea, on=['user_id', 'coupon_id'], how='left')

user_merchant_fea = pd.read_csv('data/user_merchant_feature1.csv', header=0)

t_model_merge = pd.merge(t_model_merge, user_merchant_fea, on=['user_id', 'merchant_id'], how='left')
f_model_merge = pd.merge(f_model_merge, user_merchant_fea, on=['user_id', 'merchant_id'], how='left')

t_lable = t_model_merge['lable']
f_lable = f_model_merge['lable']
t_model_merge.drop(['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date', 'lable'], axis=1, inplace=True)
f_model_merge.drop(['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date', 'lable'], axis=1, inplace=True)

t_model_merge.insert(len(t_model_merge.columns), 'lable', t_lable)
f_model_merge.insert(len(f_model_merge.columns), 'lable', f_lable)

t_model_merge.fillna(value=-1., inplace=True)
f_model_merge.fillna(value=-1., inplace=True)

model_merge = pd.concat([t_model_merge, f_model_merge])
model_merge = model_merge.sample(frac=1)
model_merge.to_csv('train_data/model_merge1.csv',index=None)