# 数据切割
# offine行数
line_num = 1053280
# with open('2nd-process/instance.csv', 'r') as fin:
#     print(len(fin.readlines()))

train_num = int(line_num / 5 * 3)
cross_num = int(line_num / 5)
test_num = int(line_num / 5)

#数据切割
with open('2nd-process/instance.csv', 'r') as fin:
    with open('2nd-process/tarin_data.csv', 'w', newline='') as fout:
        for _ in range(train_num):
            fout.write(fin.readline())
    with open('2nd-process/cross_data.csv', 'w', newline='') as fout:
        for _ in range(cross_num):
            fout.write(fin.readline())
    with open('2nd-process/test_data.csv', 'w', newline='') as fout:
        for _ in range(test_num):
            fout.write(fin.readline())