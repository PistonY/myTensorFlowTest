import numpy as np
import pandas as pd

off_train = pd.read_csv(r'C:\Users\soulf\PycharmProjects\TC-data\o2o\ccf_offline_stage1_train\ccf_offline_stage1_train.csv', header=0, keep_default_na=False)
off_train.columns=['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']

off_test = pd.read_csv(r'C:\Users\soulf\PycharmProjects\TC-data\o2o\ccf_offline_stage1_test_revised.csv',header=0, keep_default_na=False)
off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']

on_train = pd.read_csv(r'C:\Users\soulf\PycharmProjects\TC-data\o2o\ccf_online_stage1_train\ccf_online_stage1_train.csv', header=0, keep_default_na=False)
on_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']


feature1 = off_train[(off_train.date >= '20160101')&(off_train.date <= '20160413')|((off_train.date == 'null')&(off_train.date_received >= '20160101')&(off_train.date_received <= '20160413'))]
test1 = off_train[(off_train.date_received >= '20160414')&(off_train.date_received <= '20160514')]

feature2 = (off_train.date >= '20160201')&(off_train.date <= '20160514')|((off_train.date == 'null')&(off_train.date_received >= '20160201')&(off_train.date_received <= '20160514'))
test2 = (off_train.date_received >= '20160515')&(off_train.date_received <= '20160615')

feature3 = on_train[(on_train.date >= '20160101')&(on_train.date <= '20160413')|((on_train.date == 'null')&(on_train.date_received >= '20160101')&(on_train.date_received <= '20160413'))]
test3 = on_train[(on_train.date_received >= '20160414')&(on_train.date_received <= '20160514')]

# 用户线下相关的特征u
#
# 用户领取优惠券次数u1
u1 = feature1[feature1.coupon_id != 'null']
u1['user_get_coupon_num'] = 1
u1 = u1.groupby('user_id')['user_get_coupon_num'].agg('sum').reset_index()

# 用户获得优惠券但没有消费的次数u2
u2 = feature1[(feature1.coupon_id != 'null')&(feature1.date == 'null')]
u2['user_received_not_use'] = 1
u2 = u2.groupby('user_id')['user_received_not_use'].agg('sum').reset_index()

# 用户获得优惠券并核销次数u3
u3 = feature1[(feature1.coupon_id != 'null')&(feature1.date != 'null')]
u3['user_use_coupon'] = 1
u3 = u3.groupby('user_id')['user_use_coupon'].agg('sum').reset_index()

# 用户领取优惠券后进行核销率u4
u4 = pd.merge(u2, u3, on=['user_id'], how='left')
u4.fillna(value = 0, inplace=True)
u4['use_num_divide_not_use_num'] = u4.apply(lambda x: int(x.user_use_coupon)/x.user_received_not_use, axis=1)
u4 = u4[['user_id', 'use_num_divide_not_use_num']]

# 用户满200~500 减的优惠券核销率u5
u5 = feature1[feature1.coupon_id != 'null']


def get_man(t):
    sp = float(t.split(':')[0])
    if sp < 1.:
        return 0.
    else:
        return sp


def get_jian(t):
    sp = float(t.split(':')[0])
    if sp < 1.:
        return 0.
    else:
        return float(t.split(':')[1])


u5['man'] = u5.discount_rate.apply(get_man)
u5['jian'] = u5.discount_rate.apply(get_jian)
u5 = u5[(u5.man >= 200)&(u5.man <= 500)]
u5 = u5[['user_id', 'man', 'jian', 'date']]

ut1 = u5.apply(lambda x:x)
ut1['twoH_fiveH_num'] = 1
ut1= ut1.groupby('user_id')['twoH_fiveH_num'].agg('sum').reset_index()

ut2 = u5.apply(lambda x:x)
ut2['twoH_fiveH_use'] = ut2.apply(lambda x: 0 if x.date == 'null' else 1, axis=1)
ut2 = ut2.groupby('user_id')['twoH_fiveH_use'].agg('sum').reset_index()

u5 = pd.merge(ut1, ut2, on=['user_id'], how='left')
u5['200_500_use_rate'] = u5.apply(lambda x: x.twoH_fiveH_use / x.twoH_fiveH_num, axis=1)
u5 = u5[['user_id', '200_500_use_rate']]

# 用户核销满200~500减的优惠券占所有核销优惠券的比重u6
u6 = pd.merge(u3, ut2, on=['user_id'], how='left')
u6.fillna(value=0., inplace=True)
u6['200_500_use_all_rate'] = u6.apply(lambda x: x.twoH_fiveH_use / x.user_use_coupon, axis=1)
u6 = u6[['user_id', '200_500_use_all_rate']]

# 用户核销优惠券的平均/最低/最高消费折率u7
u7 = feature1[(feature1.coupon_id != 'null')&(feature1.date != 'null')]
u7 = u7[['user_id', 'discount_rate']]


def get_discount_rate(t):
    sp = float(t.split(':')[0])
    if sp < 1.:
        return str(1 - sp)
    else:
        return str(float(t.split(':')[1]) / sp)


u7.discount_rate = u7.discount_rate.apply(get_discount_rate)
u7 = u7.groupby('user_id')['discount_rate'].agg(lambda x:':'.join(x)).reset_index()
u7['max_use_disc_rate'] = u7.discount_rate.apply(lambda x: max([float (d) for d in x.split(':')]))
u7['min_use_disc_rate'] = u7.discount_rate.apply(lambda x: min([float (d) for d in x.split(':')]))
u7['mean_use_disc_rate'] = u7.discount_rate.apply(lambda x: sum([float (d) for d in x.split(':')]) / len(x.split(':')))
u7 = u7[['user_id', 'max_use_disc_rate', 'min_use_disc_rate', 'mean_use_disc_rate']]


# 用户核销过优惠券的不同商家数量，及其占所有不同商家的比重u8
u8 = feature1[(feature1.coupon_id != 'null')&(feature1.date != 'null')]
u8['diff_merht_num'] = 1
u8 = u8.groupby(['user_id', 'merchant_id'])['diff_merht_num'].agg('sum').reset_index()
u8['diff_merht_num'] = 1
u8 = u8.groupby(['user_id'])['diff_merht_num'].agg('sum').reset_index()
u8['all_merht_rate'] = u8.apply(lambda x:x.diff_merht_num / 5948, axis=1)


# 用户核销过的不同优惠券数量，及其占所有不同优惠券的比重u9
u9 = feature1[(feature1.coupon_id != 'null')&(feature1.date != 'null')]
u9['diff_coupon_num'] = 1
u9 = u9.groupby(['user_id', 'coupon_id'])['diff_coupon_num'].agg('sum').reset_index()
u9['diff_coupon_num'] = 1
u9 = u9.groupby(['user_id'])['diff_coupon_num'].agg('sum').reset_index()
u9['all_coupon_rate'] = u9.apply(lambda x:x.diff_coupon_num / 2709, axis=1)


# 用户平均核销每个商家多少张优惠券u10
u10 = pd.merge(u3, u8, on=['user_id'])
u10['ave_on_difft_merht'] = u10.apply(lambda x: x.user_use_coupon / x.diff_merht_num, axis=1)
u10 = u10[['user_id', 'ave_on_difft_merht']]

# 用户核销优惠券中的平均/最大/最小用户-商家距离u11
u11 = feature1[(feature1.coupon_id != 'null')&(feature1.date != 'null')]
u11['distance'] = u11['distance'].replace('null', '0')
u11 = u11.groupby('user_id')['distance'].agg(lambda x:':'.join(x)).reset_index()
u11['max_distance'] = u11.distance.apply(lambda x: max([float (d) for d in x.split(':')]))
u11['min_distance'] = u11.distance.apply(lambda x: min([float (d) for d in x.split(':')]))
u11['mean_distance'] = u11.distance.apply(lambda x: sum([float (d) for d in x.split(':')]) / len(x.split(':')))
u11 = u11[['user_id', 'max_distance', 'min_distance', 'mean_distance']]

u_mr = feature1[['user_id']]
u_mr = u_mr['user_id'].drop_duplicates().reset_index()[['user_id']]
u_mr = pd.merge(u_mr, u1, on=['user_id'])
u_mr = pd.merge(u_mr, u2, on=['user_id'])
u_mr = pd.merge(u_mr, u3, on=['user_id'])
u_mr = pd.merge(u_mr, u4, on=['user_id'])
u_mr = pd.merge(u_mr, u5, on=['user_id'])
u_mr = pd.merge(u_mr, u6, on=['user_id'])
u_mr = pd.merge(u_mr, u7, on=['user_id'])
u_mr = pd.merge(u_mr, u8, on=['user_id'])
u_mr = pd.merge(u_mr, u9, on=['user_id'])
u_mr = pd.merge(u_mr, u10, on=['user_id'])
u_mr = pd.merge(u_mr, u11, on=['user_id'])
u_mr.fillna(value=0., inplace=True)

u_mr.to_csv('data/off_feature1.csv',index=None)
# 用户线上相关的特征o

# 用户线上操作次数o1
o1 = feature3.apply(lambda x:x)
o1['user_online_action_num'] = 1
o1 = o1.groupby('user_id')['user_online_action_num'].agg('sum').reset_index()

# 用户线上点击率o2
o2 = feature3.apply(lambda x:x)
o2['action'] = o2['action'].astype('str')
o2 = o2.groupby('user_id')['action'].agg(lambda x: ':'.join(x)).reset_index()


def get_action_rate(t):
    l = [int(d) for d in t.split(':')]
    return (l.count(0) + l.count(2)) / len(l)


def get_buy_rate(t):
    l = [int(d) for d in t.split(':')]
    return (l.count(1)) / len(l)


def get_coupon_rate(t):
    l = [int(d) for d in t.split(':')]
    return (l.count(2)) / len(l)


o2['action_rate'] = o2.action.apply(get_action_rate)
# 用户线上购买率o3
o2['buy_rate'] = o2.action.apply(get_buy_rate)
# 用户线上领取率o4
o2['coupon_rate'] = o2.action.apply(get_coupon_rate)

# 用户线上不消费次数o5
def get_not_buy(t):
    l = [int(d) for d in t.split(':')]
    return l.count(0) + l.count(2)


o2['user_not_buy_num'] = o2.action.apply(get_not_buy)

# 用户线上优惠券核销次数o6
def get_buy_num(t):
    l = [int(d) for d in t.split(':')]
    return l.count(1)


o2['user_buy_num'] = o2.action.apply(get_buy_num)


# 用户线上优惠券核销率o7
ot1 = feature3[(feature3.date != 'null')&(feature3.coupon_id != 'null')]
ot1['user_use_coupon_num'] = 1
ot1 = ot1.groupby('user_id')['user_use_coupon_num'].agg('sum').reset_index()

ot2 = feature3[feature3.coupon_id != 'null']
ot2['user_get_coupon'] = 1
ot2 = ot2.groupby('user_id')['user_get_coupon'].agg('sum').reset_index()

o7 = pd.merge(ot2, ot1, on=['user_id'], how='left')
o7.fillna(value=0., inplace=True)
o7['on_user_use_coupon_rate'] = o7.apply(lambda x: x.user_use_coupon_num / x.user_get_coupon, axis=1)
o7 = o7[['user_id', 'on_user_use_coupon_rate']]

# 用户线下不消费次数占线上线下总的不消费次数的比重o8
ot3 = o2[['user_id', 'user_not_buy_num']]
o8 = pd.merge(u2, ot3, on=['user_id'], how='left')
o8.fillna(value=0., inplace=True)
o8['off_not_buy_div_all_not_buy'] = o8.apply(lambda x: x.user_received_not_use / (x.user_received_not_use + x.user_not_buy_num), axis=1)
o8 = o8[['user_id', 'off_not_buy_div_all_not_buy']]

# 用户线下的优惠券核销次数占线上线下总的优惠券核销次数的比重o9
ot4 = o2[['user_id', 'user_buy_num']]
o9 = pd.merge(u3, ot4, on=['user_id'], how='left')
o9.fillna(value=0., inplace=True)
o9['off_buy_div_all_buy'] = o9.apply(lambda x: x.user_use_coupon / (x.user_use_coupon + x.user_buy_num), axis=1)
o9 = o9[['user_id', 'off_buy_div_all_buy']]

# 用户线下领取的记录数量占总的记录数量的比重o10
ot5 = o2[['user_id', 'action']]
ot5['on_get_coupon_num'] = ot5.action.apply(lambda x: [int(d) for d in x.split(':')].count(2))
ot5 = ot5[['user_id', 'on_get_coupon_num']]
o10 = pd.merge(u1, ot5, on=['user_id'], how='left')
o10.fillna(value=0., inplace=True)
o10['off_get_div_all_get'] = o10.apply(lambda x: x.user_get_coupon_num / (x.user_get_coupon_num + x.on_get_coupon_num), axis=1)
o10 = o10[['user_id', 'off_get_div_all_get']]

o2 = o2[['user_id', 'action_rate', 'buy_rate', 'coupon_rate', 'user_not_buy_num', 'user_buy_num']]


o_mr = feature3[['user_id']]
o_mr = o_mr['user_id'].drop_duplicates().reset_index()[['user_id']]
o_mr = pd.merge(o_mr, o1, on=['user_id'])
o_mr = pd.merge(o_mr, o2, on=['user_id'])
o_mr = pd.merge(o_mr, o7, on=['user_id'])
o_mr = pd.merge(o_mr, o8, on=['user_id'])
o_mr = pd.merge(o_mr, o9, on=['user_id'])
o_mr = pd.merge(o_mr, o10, on=['user_id'])
o_mr.fillna(value=0., inplace=True)

o_mr.to_csv('data/on_feature1.csv',index=None)
# 商家相关的特征
#
# 商家优惠券被领取次数

# 商家优惠券被领取后不核销次数
# 商家优惠券被领取后核销次数
# 商家优惠券被领取后核销率
# 商家优惠券核销的平均/最小/最大消费折率
# 核销商家优惠券的不同用户数量，及其占领取不同的用户比重
# 商家优惠券平均每个用户核销多少张
# 商家被核销过的不同优惠券数量
# 商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重
# 商家平均每种优惠券核销多少张
# 商家被核销优惠券的平均时间率
# 商家被核销优惠券中的平均/最小/最大用户-商家距离


# 用户-商家交互特征
#
# 用户领取商家的优惠券次数
# 用户领取商家的优惠券后不核销次数
# 用户领取商家的优惠券后核销次数
# 用户领取商家的优惠券后核销率
# 用户对每个商家的不核销次数占用户总的不核销次数的比重
# 用户对每个商家的优惠券核销次数占用户总的核销次数的比重
# 用户对每个商家的不核销次数占商家总的不核销次数的比重
# 用户对每个商家的优惠券核销次数占商家总的核销次数的比重


# 优惠券相关的特征
#
# 优惠券类型(直接优惠为0, 满减为1)
# 优惠券折率
# 满减优惠券的最低消费
# 历史出现次数
# 历史核销次数
# 历史核销率
# 历史核销时间率
# 领取优惠券是一周的第几天
# 领取优惠券是一月的第几天
# 历史上用户领取该优惠券次数
# 历史上用户消费该优惠券次数
# 历史上用户对该优惠券的核销率


# 其它特征
#
# 这部分特征利用了赛题leakage，都是在预测区间提取的。
#
# 用户领取的所有优惠券数目
# 用户领取的特定优惠券数目
# 用户此次之后/前领取的所有优惠券数目
# 用户此次之后/前领取的特定优惠券数目
# 用户上/下一次领取的时间间隔
# 用户领取特定商家的优惠券数目
# 用户领取的不同商家数目
# 用户当天领取的优惠券数目
# 用户当天领取的特定优惠券数目
# 用户领取的所有优惠券种类数目
# 商家被领取的优惠券数目
# 商家被领取的特定优惠券数目
# 商家被多少不同用户领取的数目
# 商家发行的所有优惠券种类数目