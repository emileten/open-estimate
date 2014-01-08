import sys
sys.path.append("../lib/models")

import numpy as np
from scipy.stats import norm
import csv, math, generate

# Generate sliding normal
def sliding_normal(xx, yy):
    # header row
    table = [['ddp1']]
    table[0].extend(yy)
    # data rows
    for ii in range(len(xx)):
        row = [xx[ii]]        
        cdf = norm.cdf(yy, loc=xx[ii] / 8, scale=1)
        cdf = np.concatenate((cdf, [1]))
        row.extend(list(np.diff(cdf)))
        table.append(row)

    return table

def as_logs(table):
    table[0][0] = 'ddp2'
    for ii in range(len(table) - 1):
        table[ii+1][1:] = map(lambda x: math.log(x) if x > 0 else '-inf', table[ii+1][1:])

    return table

def write_table(table, filename):
    with open(filename, 'wb') as fp:
        writer = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(table)

nx = 40
ny = 40
xx = np.linspace(0, 40, nx)
yy = np.linspace(-10, 10, ny)

table = sliding_normal(xx, yy)
write_table(table, "ddp1.csv")
write_table(as_logs(table), "ddp2.csv")
write_table(generate.uniform_constant(xx, yy, 0, 5), "ddp3.csv")
