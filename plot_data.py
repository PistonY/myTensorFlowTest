import matplotlib.pyplot as plt
import csv

with open('tarin_data.csv', 'r') as fin:
    reader = csv.reader(fin)
    i = 0
    for line in reader:
        flg = ''
        if line[2] == '0':
            flg = 'bo'
        else:
            flg = 'ro'
        plt.plot([line[0]], [line[1]], flg)
        i = i + 1
        if i == 5000:
            break
    plt.show()
