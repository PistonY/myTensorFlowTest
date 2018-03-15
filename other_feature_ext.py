import pandas as pd
import numpy as np
from datetime import date
import datetime as dt


off_test = pd.read_csv(r'C:\Users\soulf\PycharmProjects\TC-data\o2o\ccf_offline_stage1_test_revised.csv',header=0)
off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']



#使数据集3等于test集
dataset3 = off_test

# 其它特征
#
# 这部分特征利用了赛题leakage，都是在预测区间提取的。
#
# 用户领取的所有优惠券数目pu1
# 用户领取的特定优惠券数目pu2
# 用户此次之后/前领取的所有优惠券数目pu3
# 用户此次之后/前领取的特定优惠券数目pu4
# 用户上/下一次领取的时间间隔pu5
# 用户领取特定商家的优惠券数目pu6
# 用户领取的不同商家数目pu7
# 用户当天领取的优惠券数目pu8
# 用户当天领取的特定优惠券数目pu9
# 用户领取的所有优惠券种类数目pu10
# 商家被领取的优惠券数目pm1
# 商家被领取的特定优惠券数目pm2
# 商家被多少不同用户领取的数目pm3
# 商家发行的所有优惠券种类数目pm4

#对于测试集3，提取用户的Id
t = dataset3[['user_id']]
#相当于给原有数据加上一列，这个月用户收取的所有优惠券数目，并初始化为1
t['this_month_user_receive_all_coupon_count'] = 1

#将t按照用户id进行分组，然后统计所有用户收取的优惠券数目,并初始化一个索引值
t = t.groupby('user_id').agg('sum').reset_index()
#提取数据集3的优惠券Id和用户Id
t1 = dataset3[['user_id','coupon_id']]
#提取这个月用户收到的相同的优惠券的数量
t1['this_month_user_receive_same_coupn_count'] = 1
t1 = t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()

#提取数据集3的用户id，优惠券id以及优惠券接收的时间
t2 = dataset3[['user_id','coupon_id','date_received']]

#将数据转换为str类型
t2.date_received = t2.date_received.astype('str')
#如果出现相同的用户接收相同的优惠券在接收时间上用‘：’连接上第n次接受优惠券的时间
t2 = t2.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
#将接收时间的一组按着':'分开，这样就可以计算接受了优惠券的数量,apply是合并
t2['receive_number'] = t2.date_received.apply(lambda s:len(s.split(':')))
t2 = t2[t2.receive_number > 1]
#最大接受的日期
t2['max_date_received'] = t2.date_received.apply(lambda s:max([int (d) for d in s.split(':')]))
#最小的接收日期
t2['min_date_received'] = t2.date_received.apply(lambda s:min([int (d) for d in s.split(':')]))
t2 = t2[['user_id','coupon_id','max_date_received','min_date_received']]

t3 = dataset3[['user_id','coupon_id','date_received']]
#将两表融合只保留左表数据,这样得到的表，相当于保留了最近接收时间和最远接受时间
t3 = pd.merge(t3,t2,on=['user_id','coupon_id'],how='left')
#这个优惠券最近接受时间
t3['this_month_user_receive_same_coupon_lastone']= t3.max_date_received-t3.date_received.astype(int)
#这个优惠券最远接受时间
t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received.astype(int)-t3.min_date_received

def is_firstlastone(x):
    if x==0:
        return 1
    elif x>0:
        return 0
    else:
        return -1 #表明这个优惠券只接受了一次
t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
t3 = t3[['user_id','coupon_id','date_received','this_month_user_receive_same_coupon_lastone','this_month_user_receive_same_coupon_firstone']]
#将表格中接收优惠券日期中为最近和最远的日期时置为1其余为0，若只接受了一次优惠券为-1

#提取第四个特征,一个用户所接收到的所有优惠券的数量
t4 = dataset3[['user_id','date_received']]
t4['this_day_receive_all_coupon_count'] = 1
t4 = t4.groupby(['user_id','date_received']).agg('sum').reset_index()

#提取第五个特征,一个用户不同时间所接收到不同优惠券的数量
t5 = dataset3[['user_id','coupon_id','date_received']]
t5['this_day_user_receive_same_coupon_count'] = 1
t5 = t5.groupby(['user_id','coupon_id','date_received']).agg('sum').reset_index()
#一个用户不同优惠券 的接受时间
t6 = dataset3[['user_id','coupon_id','date_received']]
t6.date_received = t6.date_received.astype('str')
t6 = t6.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
#重命名inplace代表深拷贝
t6.rename(columns={'date_received':'dates'},inplace = True)

def get_day_gap_before(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        #将时间差转化为天数
        this_gap = (dt.date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-dt.date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
def get_day_gap_after(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (dt.datetime(int(d[0:4]),int(d[4:6]),int(d[6:8]))-dt.datetime(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)

t7 = dataset3[['user_id','coupon_id','date_received']]
#将t6和t7融合
t7 = pd.merge(t7,t6,on=['user_id','coupon_id'],how='left')
#注意这里所有的时间格式都已经是'str'格式
t7['date_received_date'] = t7.date_received.astype('str')+'-'+t7.dates
#print(t7)
t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
t7 = t7[['user_id','coupon_id','date_received','day_gap_before','day_gap_after']]

#将所有特征融合在一张表中
other_feature3 = pd.merge(t1,t,on='user_id')
other_feature3 = pd.merge(other_feature3,t3,on=['user_id','coupon_id'])
other_feature3 = pd.merge(other_feature3,t4,on=['user_id','date_received'])
other_feature3 = pd.merge(other_feature3,t5,on=['user_id','coupon_id','date_received'])
other_feature3 = pd.merge(other_feature3,t7,on=['user_id','coupon_id','date_received'])
other_feature3.to_csv('data1/other_feature.csv',index=None)
#print(other_feature3)