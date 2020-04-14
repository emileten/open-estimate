import csv, random, math

def generate_random_us(pathto=""):
    bycounty = {}

    with open(pathto + "elevation.csv", 'rU') as fipsfp:
        reader = csv.reader(fipsfp, delimiter=',')
        for row in reader:
            bycounty[row[0]] = float(row[1]) if row[1] != "NA" else 0

    with open(pathto + "us_latitudes.csv", 'rU') as fipsfp:
        reader = csv.reader(fipsfp, delimiter=',')
        for row in reader:
            avg = 18 - (float(row[1]) - 40) # Averages by latitude
            avg -= bycounty[row[0]] * 6.5 / 1000 # Lapse rate
            avg += random.triangular(-5, 5, 0) # Some noise
            if avg < 0:
                avg = 0
            bycounty[row[0]] = avg

    return bycounty

# Growing season temp is \frac{1}{t_2 - t_1} \int_{t_1}^{t_2} \bar{T} - A cos(2\pi t / 365)
#   = \bar{T} - \frac{A}{t_2 - t_1} \frac{365}{2 \pi} (sin(2\pi t_2 / 365) - sin(2\pi t_1 / 365))
def generate_growing_season(pathto, cropfile):
    bycounty = generate_random_us(pathto)

    with open(cropfile, 'rU') as fipsfp:
        reader = csv.reader(fipsfp, delimiter=',')
        for row in reader:
            if row[1] == "None":
                continue

            plantday = float(row[1])
            harvestday = float(row[2])
            temp = bycounty[row[0]] - (10 / (harvestday - plantday)) * (365 / (2*math.pi)) * (math.sin(2*math.pi * harvestday / 365.0) - math.sin(2*math.pi * plantday / 365.0))
            yield row[0], temp

def make_constant_365years(year0, const, num):
    yyyyddd = []
    values = []
    for year in range(year0, year0 + num):
        yyyyddd = yyyyddd + list(range(year * 1000 + 1, year * 1000 + 366))
        values = values + [const] * 365

    return yyyyddd, values

def make_sampling_365years(year0, value0, value1, num):
    yyyyddd = []
    values = []
    byyear = []
    for year in range(year0, year0 + num):
        yyyyddd = yyyyddd + list(range(year * 1000 + 1, year * 1000 + 366))
        values = values + [value0 + (value1 - value0) * float(year - year0) / num] * 365
        byyear.append(value0 + (value1 - value0) * float(year - year0) / num)

    return yyyyddd, values, byyear

