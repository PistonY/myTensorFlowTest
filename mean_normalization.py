import csv

# 特征缩放

def get_row_col_num(file_path):
    with open(file_path, 'r') as fin:
        row_num = len(fin.readlines())
    with open(file_path, 'r') as fin:
        reader = csv.reader(fin)
        col_num = len(next(reader))
    return row_num, col_num


def get_mu_S(file_path):
    with open(file_path, 'r') as fin:
        reader = csv.reader(fin)
        line_num, col_num = get_row_col_num(file_path)
        mu = [0. for _ in range(col_num)]
        S = [0. for _ in range(col_num)]
        max_V = [0. for _ in range(col_num)]
        min_V = [10000000. for _ in range(col_num)]
        sum_V = [0. for _ in range(col_num)]

        for line in reader:
            for i in range(col_num):
                f_i = float(line[i])
                sum_V[i] = sum_V[i] + f_i
                if f_i < min_V[i]:
                    min_V[i] = f_i
                if f_i > max_V[i]:
                    max_V[i] = f_i
        for i in range(col_num):
            mu[i] = sum_V[i] / line_num
            S[i] = max_V[i] - min_V[i]
            # if S[i] == 0.:
            #     print(i, col_num)
        # print(mu, S)
        return mu, S


file_path = '2nd-process/instance.csv'
mu, S = get_mu_S(file_path)
with open(file_path, 'r') as fin:
    _, col_num = get_row_col_num(file_path)
    with open('2nd-process/MN_data.csv', 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            ps_data = []
            ps_data.append(float(row[0]))
            for i in range(col_num - 1):
                if i == 0:
                    continue
                ps_data.append((float(row[i]) - mu[i]) / S[i])
            ps_data.append(float(row[col_num - 1]))
            writer.writerow(ps_data)







