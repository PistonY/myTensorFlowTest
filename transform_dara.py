import csv
from itertools import islice


with open('TCdata/ccf_offline_stage1_train/ccf_offline_stage1_train.csv', 'r') as fin:
    reader = csv.reader(fin)
    with open('instance.csv', 'w', newline='') as fout:
        writer = csv.writer(fout)
        lable = []
        for row in islice(reader, 1, None):
            #row[2]:Cid
            #row[6]:Date
            if row[6] == 'null' and row[2] != 'null':
                lable = ['0']
            elif row[6] != 'null' and row[2] == 'null':
                lable = ['0']
            elif row[6] != 'null' and row[2] != 'null':
                lable = ['1']
            else:
                lable = ['0']
                print("Nerver see:" + " Date:" + row[6] + " Cid:" +row[2])

            #Discount_rate
            if row[3] == 'null':
                row[3] = '0'
            elif float((row[3].split(':'))[0]) >= 1.0:
                tp = row[3].split(':')
                row[3] = str(10000 * float(tp[1]) / (float(tp[0]) ** 2))

            #Distance
            if row[4] == 'null':
                row[4] = '100'
            ans = row[3:5] + lable
            writer.writerow(ans)
