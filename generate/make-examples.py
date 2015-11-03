import sys
sys.path.append("../../lib/models")

import os, csv, random, tarfile
import numpy as np
from scipy.interpolate import UnivariateSpline
from spline_model import SplineModelConditional

def make_tar(name, write_file):
    if not os.path.exists(name):
        os.mkdir(name)

    with open("geography/fips_codes.csv", 'rU') as fipsfp:
        reader = csv.reader(fipsfp, dialect=csv.excel_tab, delimiter=',')
        reader.next()
        for row in reader:
            if row[-1] != "County":
                continue
            with open(os.path.join(name, row[1] + row[2] + '.csv'), 'wb') as csvfp:
                writer = csv.writer(csvfp, quoting=csv.QUOTE_MINIMAL)
                write_file(writer, row)

    os.system("tar -czf " + name + ".tar.gz " + name)
    os.system("rm -r " + name)

# Fraction of expected yields
def write_yields_grain(writer, row):
    print row
    writer.writerow(["year", "fraction"])
    for year in range(2010, 2100+1):
        writer.writerow([year, random.uniform(.5, 1.5) * (1 - (year - 2010) / 200.0)])

# Robberies per 1e6 people, for incomes up to $100,000
def write_robberies(writer, row):
    print row
    writer.writerow(["year", "income0", "income1", "coeff0", "coeff1", "coeff2"])
    incomes = np.linspace(0, 1e6, 20)
    for year in range(2010, 2100+1):
        crime_fraction = np.sin(np.pi*incomes / 1e6) + random.uniform(.5, 1.5) * (1 + (year - 2010) / 200.0)
        crime = 200 * crime_fraction
        spline = UnivariateSpline(incomes, crime, s=10, k=2)
        conditional = SplineModelConditional.make_conditional_from_spline(spline, [0, 1e6])
        for jj in range(conditional.size()):
            row = [year, conditional.y0s[jj], conditional.y1s[jj]]
            row.extend(conditional.coeffs[jj])
            writer.writerow(row)

make_tar("yields-grain", write_yields_grain)
make_tar("robberies", write_robberies)
#with tarfile.open('yields-grain.tar.gz', 'w:gz') as tgfp:
