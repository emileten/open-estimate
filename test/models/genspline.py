import numpy as np
import csv, math

def gaussian_spline(filename):
    with open(filename, 'wb') as fp:
        fp.write("dpc1,mean,.025,.975\n")
        writer = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for xx in np.linspace(0, 1, 80):
            diff = (.5 + abs(xx - 1) / 3.0)*(.1 + abs(math.sin(2*math.pi*xx))) + xx
            writer.writerow([xx, math.cos(2*math.pi*xx), math.cos(2*math.pi*xx) - diff, math.cos(2*math.pi*xx) + diff])

gaussian_spline("spline-cicurve.csv")
