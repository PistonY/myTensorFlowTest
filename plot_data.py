import matplotlib.pyplot as plt
import csv

with open('MN_data.csv', 'r') as fin:
    reader = csv.reader(fin)
    i = 0
    for line in reader:
        flg = ''
        if float(line[2]) == 0.:
            flg = 'b.'
        else:
            flg = 'r.'
        plt.plot([line[0]], [line[1]], flg)
        i = i + 1
        if i == 2000:
            break
    plt.show()
