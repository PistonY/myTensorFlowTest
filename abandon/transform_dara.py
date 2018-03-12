import csv
from itertools import islice
from sklearn.preprocessing import PolynomialFeatures

# 数据转化
# 数据高次化（构造高次方项）

def set_pdynome_degree(degree, lis):
    lis = [lis]
    ploy = PolynomialFeatures(degree)
    result = ploy.fit_transform(lis)
    result = result.tolist()
    result = result[0]
    return result


with open('H:/TCdata/ccf_offline_stage1_train/ccf_offline_stage1_train.csv', 'r') as fin:
    reader = csv.reader(fin)
    with open('2nd-process/instance.csv', 'w', newline='') as fout:
        writer = csv.writer(fout)
        lable = []
        for row in islice(reader, 1, None):
            #row[2]:Cid
            #row[6]:Date
            if row[6] == 'null' and row[2] != 'null':
                lable = 0.
            elif row[6] != 'null' and row[2] == 'null':
                continue
                lable = 0.
            elif row[6] != 'null' and row[2] != 'null':
                lable = 1.
            else:
                lable = 0.
                print("Nerver see:" + " Date:" + row[6] + " Cid:" +row[2])


            result = []
            threshold = 0.
            #Discount_rate
            if row[3] == 'null':
                print(row[3])
                row[3] = 1.
            elif float((row[3].split(':'))[0]) >= 1.0:
                tp = row[3].split(':')
                row[3] = (float(tp[0]) - float(tp[1])) / float(tp[0])
                threshold = float(tp[0])
            else:
                threshold = 0.

            result.append(threshold)
            result.append(row[3])

            #Distance
            if row[4] == 'null':
                if lable == 0.:
                    row[4] = 10.
                else:
                    row[4] = 4.
            result.append(float(row[4]))
            # print(result)
            result = set_pdynome_degree(5, result)
            result.append(lable)
            # result = row[3:5] + lable
            writer.writerow(result)
