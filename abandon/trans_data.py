import numpy as np
import pandas as pd

off_train = pd.read_csv(r'C:\Users\soulf\PycharmProjects\TC-data1\o2o\ccf_offline_stage1_train\ccf_offline_stage1_train.csv', header=0, keep_default_na=False)
off_train.columns=['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']

off_test = pd.read_csv(r'C:\Users\soulf\PycharmProjects\TC-data1\o2o\ccf_offline_stage1_test_revised.csv',header=0, keep_default_na=False)
off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']

# on_train = pd.read_csv(r'C:\Users\soulf\PycharmProjects\TC-data1\o2o\ccf_online_stage1_train\ccf_online_stage1_train.csv', header=0, keep_default_na=False)
# on_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']


feature1 = off_train[(off_train.date >= '20160101')&(off_train.date <= '20160413')|((off_train.date == 'null')&(off_train.date_received >= '20160101')&(off_train.date_received <= '20160413'))]
test1 = off_train[(off_train.date_received >= '20160414')&(off_train.date_received <= '20160514')]

feature2 = (off_train.date >= '20160201')&(off_train.date <= '20160514')|((off_train.date == 'null')&(off_train.date_received >= '20160201')&(off_train.date_received <= '20160514'))
test2 = (off_train.date_received >= '20160515')&(off_train.date_received <= '20160615')

# 用户特征u
# 线下领取优惠券但没有使用的次数 u1
u1 = feature1[(feature1.coupon_id != 'null')&(feature1.date == 'null')]
u1['user_received_not_use'] = 1
u1 = u1.groupby('user_id')['user_received_not_use'].agg('sum').reset_index()

# 线下普通消费次数 u2
u2 = feature1[(feature1.coupon_id == 'null')&(feature1.date != 'null')]
u2['user_not_use_coupon'] = 1
u2 = u2.groupby('user_id')['user_not_use_coupon'].agg('sum').reset_index()

# 线下使用优惠券消费的次数 u3
u3 = feature1[(feature1.coupon_id != 'null')&(feature1.date != 'null')]
u3['user_use_coupon'] = 1
u3 = u3.groupby('user_id')['user_use_coupon'].agg('sum').reset_index()

# 线下平均正常消费间隔 u4
u4 = feature1[(feature1.coupon_id == 'null')&(feature1.date != 'null')]
u4 = u4.groupby('user_id')['date'].apply(lambda x: (int(max(x)) - int(min(x))) / len(x)).reset_index()
u4.columns = ['user_id', 'average_time_interval_of_common_consumption']

# 线下平均优惠券消费间隔 u5
u5 = feature1[(feature1.coupon_id != 'null')&(feature1.date != 'null')]
u5 = u5.groupby('user_id')['date'].apply(lambda x: (int(max(x)) - int(min(x))) / len(x)).reset_index()
u5.columns = ['user_id', 'average_time_interval_of_coupon_consumption']

# u3/u1 使用优惠券次数与没使用优惠券次数比值 u6
u6 = pd.merge(u1, u3, on=['user_id'], how='left')
u6.fillna(value = 0, inplace=True)
u6['use_num_divide_not_use_num'] = u6.apply(lambda x: int(x.user_use_coupon)/x.user_received_not_use, axis=1)
u6 = u6[['user_id', 'use_num_divide_not_use_num']]

# u3/(u2+u3) 表示用户使用优惠券消费占比 u7
u7 = pd.merge(u2, u3, on=['user_id'], how='outer')
u7.fillna(value = 0, inplace=True)
u7['use_coupon_rate'] = u7.apply(lambda x: x.user_use_coupon/(x.user_use_coupon + x.user_not_use_coupon), axis=1)
u7 = u7[['user_id', 'use_coupon_rate']]

# u4/15 代表15除以用户普通消费间隔，可以看成用户15天内平均会普通消费几次，值越小代表用户在15天内普通消费概率越大 u8
u8 = u4.apply(lambda x:x)
u8['user_15days_not_use_coupon'] = u8.apply(lambda x: x.average_time_interval_of_common_consumption / 15, axis=1)
u8 = u8[['user_id', 'user_15days_not_use_coupon']]

# u5/15 代表15除以用户优惠券消费间隔，可以看成用户15天内平均会普通消费几次，值越大代表用户 在15天内普通消费概率越大 u9
u9 = u5.apply(lambda x:x)
u9['user_15days_use_coupon'] = u9.apply(lambda x: x.average_time_interval_of_coupon_consumption / 15, axis=1)
u9 = u9[['user_id', 'user_15days_use_coupon']]

# 领取优惠券到使用优惠券的平均间隔时间 u10
u10 = feature1[(feature1.coupon_id != 'null')&(feature1.date != 'null')]
u10['avg_use_coupon_gap_time'] = u10.apply(lambda x: int(x.date) - int(x.date_received), axis=1)
u10 = u10.groupby('user_id')['avg_use_coupon_gap_time'].agg('mean').reset_index()

# u10/15 表示在15天内使用掉优惠券的值大小，值越小越有可能，值为0表示可能性最大 u11
u11 = u10.apply(lambda x:x)
u11['rate_iner_15days_use_coupon'] = u11.apply(lambda x: x.avg_use_coupon_gap_time / 15, axis=1)
u11 = u11[['user_id', 'avg_use_coupon_gap_time']]

# 领取优惠券到使用优惠券间隔小于15天的次数 u12
u12 = feature1[(feature1.coupon_id != 'null')&(feature1.date != 'null')]
u12['use_gap_time'] = u12.apply(lambda x: int(x.date) - int(x.date_received), axis=1)
u12 = u12[u12.use_gap_time < 15]
u12['use_gap_time_less_15days_num'] = 1
u12 = u12.groupby('user_id')['use_gap_time_less_15days_num'].agg('sum').reset_index()


# u12/u3 表示用户15天使用掉优惠券的次数除以使用优惠券的次数，表示在15天使用掉优惠券的可能，值越大越好。 u13
u13 = pd.merge(u3, u12, on=['user_id'], how='left')
u13.fillna(value = 0, inplace=True)
u13['user_use_coupon_less_15days_possible'] = u13.apply(lambda x:x.use_gap_time_less_15days_num / x.user_use_coupon, axis= 1)
u13 = u13[['user_id', 'user_use_coupon_less_15days_possible']]

# u12/u1 F014 表示用户15天使用掉优惠券的次数除以领取优惠券未消费的次数，表示在15天使用掉优惠券的可能，值越大越好。 u14
u14 = pd.merge(u1, u12, on=['user_id'], how='left')
u14.fillna(value=0, inplace=True)
u14['user_use_coupon_less_15days_possible_2'] = u14.apply(lambda x:x.use_gap_time_less_15days_num / x.user_use_coupon, axis= 1)
u14 = u14[['user_id', 'user_use_coupon_less_15days_possible_2']]

# u1+u3 领取优惠券的总次数 u15
u15 = feature1[(feature1.coupon_id != 'null')]
u15['get_coupon_num'] = 1
u15 = u15.groupby('user_id')['get_coupon_num'].agg('sum').reset_index()

# u12/u15 F016 表示用户15天使用掉优惠券的次数除以领取优惠券的总次数，表示在15天使用掉优惠券的可能，值越大越好。 u16
u16 = pd.merge(u15, u12, on=['user_id'], how='left')
u16.fillna(value=0, inplace=True)
u16['user_use_coupon_less_15days_possible_3'] = u16.apply(lambda x:x.use_gap_time_less_15days_num / x.get_coupon_num, axis=1)
u16 = u16[['user_id', 'user_use_coupon_less_15days_possible_3']]

# u1+u2 一共消费多少次 u17
u17 = pd.merge(u1, u2, on=['user_id'], how='outer')
u17.fillna(value=0, inplace=True)
u17['u1_u2_num'] = u17.apply(lambda x:x.user_received_not_use + x.user_not_use_coupon, axis=1)
u17 = u17[['user_id', 'u1_u2_num']]

# 最近一次优惠券消费到当前领券的时间间隔 u18
u18 = feature1[(feature1.coupon_id != 'null')&(feature1.date != 'null')]
u18 = u18.groupby('user_id')['date'].agg('max').reset_index()
ut = feature1[(feature1.coupon_id != 'null')&(feature1.date != 'null')]
ut = ut[['user_id', 'date_received', 'date']]
u18 = pd.merge(u18, ut, on=['user_id', 'date'], how='left')
u18['last_consume_gap_get_coupon'] = u18.apply(lambda x:int(x.date) - int(x.date_received), axis=1)
u18 = u18[['user_id', 'last_consume_gap_get_coupon']]

# 最近一次消费到当前领券的时间间隔 u19
u19 = feature1[feature1.date != 'null']
u19 = u19.groupby('user_id')['date'].agg('max').reset_index()


# 用户当天领取的优惠券数目 u20
# 用户前第i天领取的优惠券数目 u20si
# 用户后第i天领取的优惠券数目 u20ai
# 用户前7天领取的优惠券数目 u21
# 用户前3天领取的优惠券数目 u22
# u22/u21 u23
# u20/u22 u24
# 用户后7天领取的优惠券数目 u25
# 用户后3天领取的优惠券数目 u26
# u26/u25 u27
# u20/u26 u28
# 用户训练、预测时间领取的优惠券数目 u29
# 用户当天领取的不同优惠券数目 u30
# 用户前第i天领取的不同优惠券数目 u30si
# 用户后第i天领取的不同优惠券数目 u30ai
# 用户训练、预测时间领取的不同优惠券数目 u31
# 按照7/4/2分解训练、预测时间，提取此段窗口时间的特征
# 用户7/4/2天领取的优惠券数目 u32_i
# 用户7/4/2天所领取的优惠券优惠率r1/r2/r3/r4排名 u_ri_ranki
# 用户7/4/2天所领取的优惠券优惠率r1/r2/r3/r4排名 u_ri_dense _ranki
# u32_4/u32_7 u33
# u32_2/u32_4 u34
# u32_2/u32_7 u35
# u20/u32_2 u36
#
# 线上领取优惠券未使用的次数 action=2 uo1
#
# 线上特价消费次数 action=1 and cid=0 and drate=”fixed” uo2
# 线上使用优惠券消费的次数 uo3
# 线上普通消费次数 action=1 and cid=0 and drate=”null” uo4
# 线上领取优惠券的次数 uo1+uo3 uo5
# uo3/uo5 线上使用优惠券次数除以线上领取优惠券次数，正比 uo6
# uo3/uo4 线上使用优惠券次数除以线上普通消费次数，正比 uo7
# uo2/uo4线上特价消费次数除以线上普通消费次数 uo8
#
# 加入训练预测时间前一个月的窗口特征
#
# 线下领取优惠券但没有使用的次数 uw1
#
# 线下普通消费次数 uw2
# 线下使用优惠券消费的次数 uw3
# 线下平均正常消费间隔 uw4
# 线下平均优惠券消费间隔 uw5
# uw3/uw1 使用优惠券次数与没使用优惠券次数比值 uw6
# uw3/(uw2+uw3) 表示用户使用优惠券消费占比 uw7
# uw4/15 代表15除以用户普通消费间隔，可以看成用户15天内平均会普通消费几次，值越小代表用户在15天内普通消费概率越大 uw8
# uw5/15 代表15除以用户优惠券消费间隔，可以看成用户15天内平均会普通消费几次，值越大代表用户在15天内普通消费概率越大 uw9
# 领取优惠券到使用优惠券的平均间隔时间 uw10
# uw10/15 表示在15天内使用掉优惠券的值大小，值越小越有可能，值为0表示可能性最大 uw11
# 领取优惠券到使用优惠券间隔小于15天的次数 uw12
# uw12/uw3 表示用户15天使用掉优惠券的次数除以使用优惠券的次数，表示在15天使用掉优惠券的可能，值越大越好。 uw13
# uw12/uw1 F014 表示用户15天使用掉优惠券的次数除以领取优惠券未消费的次数，表示在15天使用掉优惠券的可能，值越大越好。 uw14
# uw1+uw3 领取优惠券的总次数 uw15
# uw12/uw15 F016 表示用户15天使用掉优惠券的次数除以领取优惠券的总次数，表示在15天使用掉优惠券的可能，值越大越好。 uw16
# F01+F02 一共消费多少次 uw17