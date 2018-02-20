import csv

#offine行数
line_num = 1754880
# with open('instance.csv', 'r') as fin:
#     print(len(fin.readlines()))

train_num = int(line_num / 5 * 3)
cross_num = int(line_num / 5)
test_num = int(line_num / 5)

#数据切割
with open('instance.csv', 'r') as fin:
    with open('tarin_data.csv', 'w', newline='') as fout:
        for _ in range(train_num):
            fout.write(fin.readline())
    with open('cross_data.csv', 'w', newline='') as fout:
        for _ in range(cross_num):
            fout.write(fin.readline())
    with open('test_data.csv', 'w', newline='') as fout:
        for _ in range(test_num):
            fout.write(fin.readline())