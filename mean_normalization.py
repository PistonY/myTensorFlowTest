import csv

def get_line_num(file_path):
    with open(file_path, 'r') as fin:
        line_num = len(fin.readlines())
    return line_num

def get_mu_S(file_path):
    with open(file_path, 'r') as fin:

        mu = [0., 0.]
        S = [0., 0.]
        max_V = [0., 0.]
        min_V = [100., 100.]
        sum_V = [0., 0.]
        reader = csv.reader(fin)
        line_num = get_line_num(file_path)

        for line in reader:
            f_0 = float(line[0])
            f_1 = float(line[1])
            # print(line[0], line[1])
            sum_V[0], sum_V[1] = sum_V[0] + f_0, sum_V[1] + f_1

            if f_0 < min_V[0]:
                min_V[0] = f_0
            if f_1 < min_V[1]:
                min_V[1] = f_1
            if f_0 > max_V[0]:
                max_V[0] = f_0
            if f_1 > max_V[1]:
                max_V[1] = f_1
        mu[0], mu[1] = sum_V[0] / line_num, sum_V[1] / line_num
        S[0], S[1] = max_V[0] - min_V[0], max_V[1] - min_V[1]
        # print(mu, S)
        return mu, S

file_path = 'tarin_data.csv'
mu, S = get_mu_S(file_path)
with open(file_path, 'r') as fin:
    with open('MN_data.csv', 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            ps_data = []
            ps_data.append((float(row[0]) - mu[0]) / S[0])
            ps_data.append((float(row[1]) - mu[1]) / S[1])
            ps_data.append(float(row[2]))
            writer.writerow(ps_data)







