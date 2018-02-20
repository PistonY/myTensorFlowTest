import csv
from itertools import islice

with open('TCdata/sample_submission.csv', 'r') as fin:

    # mu = [0., 0.]
    # S = [0., 0.]
    # max_V = [0., 0.]
    # min_V = [100., 100.]
    # sum_V = [0., 0.]
    reader = csv.reader(fin)

    for line in reader:
        # f_0 = float(line[0])
        # f_1 = float(line[1])
        print(line[0], line[1])
    #     sum_V[0], sum_V[1] = sum_V[0] + f_0, sum_V[1] + f_1
    #
    #     if f_0 < min_V[0]:
    #         min_V[0] = f_0
    #     if f_1 < min_V[1]:
    #         min_V[1] = f_1
    #     if f_0 > max_V[0]:
    #         max_V[0] = f_0
    #     if f_1 > max_V[1]:
    #         max_V[1] = f_1
    # mu[0], mu[1] = sum_V[0] / line_num, sum_V[1] / line_num
    # S[0], S[1] = max_V[0] - min_V[0], max_V[1] - min_V[1]
    # print(mu, S)





