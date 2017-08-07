import os, csv
import numpy as np

county_dir = "/home/dmr/county_text/access1-3/rcp45/tas"

def date_to_datestr(date):
    return ''.join([date.year, date.month, date.day])

def get_crop_calendar(cropfile):
    cropcals = {}
    with open(cropfile, 'rU') as fipsfp:
        reader = csv.reader(fipsfp, delimiter=',')
        for row in reader:
            if row[1] == "None":
                continue

            plantday = int(row[1])
            harvestday = int(row[2])
            cropcals[row[0]] = (plantday, harvestday)

    return cropcals

# assumes temppath has rows YYYYMMDD,#### and yields (year, temp)
#   allows negative plantday
def growing_seasons_mean_reader(reader, plantday, harvestday):
    prevtemps = None
    row = reader.next()
    more_rows = True
    while more_rows:
        year = row[0][0:4]
        temps = [float(row[1]) if row[1] != '-99.99' else float('NaN')]

        more_rows = False
        for row in reader:
            if row[0][0:4] != year:
                more_rows = True
                break

            temps.append(float(row[1]) if row[1] != '-99.99' else float('NaN'))

        if plantday < 0:
            if prevtemps is not None:
                temp = np.mean(prevtemps[plantday:] + temps[0:harvestday])
                yield (int(year), temp)

            prevtemps = temps
        else:
            temp = np.mean(temps[plantday:harvestday])
            yield (int(year), temp)

# allows negative plantday
def growing_seasons_mean_ncdf(yyyyddd, weather, plantday, harvestday):
    if plantday < 0:
        year0 = yyyyddd[0] // 1000
        seasons = np.array_split(weather, range(plantday - 1, len(yyyyddd), 365))
    else:
        year0 = yyyyddd[0] // 1000 + 1
        seasons = np.array_split(weather, range(plantday - 1 + 365, len(yyyyddd), 365))
    year1 = yyyyddd[-1] // 1000

    for chunk in zip(range(year0, year1 + 1), seasons):
        yield (chunk[0], np.mean(chunk[1][0:harvestday-plantday+1]))

    # Version 1
    #ii = 0
    #while ii < len(yyyyddd):
    #    year = yyyyddd[ii] // 1000
    #    if ii + plantday - 1 >= 0 and ii + harvestday <= len(yyyyddd):
    #        mean = np.mean(weather[ii:ii+365][plantday-1:harvestday])
    #        ii += 365
    #        yield (year, mean)
    #    else:
    #        ii += 365

# allows negative plantday
def growing_seasons_daily_ncdf(yyyyddd, weather, plantday, harvestday):
    if plantday < 0:
        year0 = yyyyddd[0] // 1000
        index0 = plantday - 1
    else:
        year0 = yyyyddd[0] // 1000 + 1
        index0 = plantday - 1 + 365
    year1 = yyyyddd[-1] // 1000

    if isinstance(weather, list):
        seasons = np.array_split(weather, range(plantday - 1, len(yyyyddd), 365))
        for chunk in zip(range(year0, year1 + 1), seasons):
            yield (chunk[0], chunk[1][0:harvestday-plantday+1])
    else:
        seasons = {}
        for variable in weather:
            seasons[variable] = np.array_split(weather[variable], range(plantday - 1, len(yyyyddd), 365))

        for year in range(year0, year1 + 1):
            yield (year, {variable: seasons[variable][year - year0][0:harvestday-plantday+1] for variable in seasons})

    # Version 1
    #ii = 0
    #while ii < len(yyyyddd):
    #    year = yyyyddd[ii] // 1000
    #    if ii + plantday - 1 >= 0 and ii + harvestday <= len(yyyyddd):
    #        if isinstance(weather, list):
    #            yield (year, weather[ii:ii+365][plantday-1:harvestday])
    #        else:
    #            season = {}
    #            for variable in weather:
    #                season[variable] = weather[variable][ii:ii+365][plantday-1:harvestday]
    #            yield (year, season)
    #        ii += 365
    #    else:
    #        ii += 365


def yearly_daily_ncdf(yyyyddd, weather):
    year0 = int(yyyyddd[0]) // 1000
    year1 = int(yyyyddd[-1]) // 1000
    chunks = zip(range(year0, year1+1), np.array_split(weather, range(365, len(yyyyddd), 365)))
    for chunk in chunks:
        yield chunk

    # Version 2
    #for ii in xrange(0, len(yyyyddd), 365):
    #    yield (yyyyddd[ii] // 1000, weather[ii:ii+365])

    # Version 1
    #ii = 0
    #while ii < len(yyyyddd):
    #    year = yyyyddd[ii] // 1000
    #    if ii + 365 <= len(yyyyddd):
    #        yield (year, weather[ii:ii+365])
    #        ii += 365
    #    else:
    #        ii += 365

def xmap_apply_model(xmap, model, pval):
    data = {}
    total = 0
    for (key, val) in xmap:
        total += 1
        if total % 100 == 0:
            print total
        result = model.eval_pval(val, pval, 1e-2)
        if not np.isnan(result):
            yield (key, result)

# effects and scales need to be lists of same length, containing iterators (key, val) with same keys
def combo_effects(effect_dicts, scale_gens):
    numers = {}
    denoms = {}
    for ii in range(len(effect_dicts)):
        for (key, scale) in scale_gens[ii]:
            if scale == 0 or key not in effect_dicts[ii]:
                continue

            if key not in numers:
                numers[key] = 0
                denoms[key] = 0

            numers[key] += effect_dicts[ii][key] * scale
            denoms[key] += scale

    return {key: numers[key] / denoms[key] for key in numers}

def read_scale_file(filepath, factor):
    with open(filepath, "r") as fp:
        reader = csv.reader(fp, delimiter=',')
        for row in reader:
            if row[1] == 'NA':
                continue

            fips = row[0]
            if len(fips) == 4:
                fips = '0' + fips
            yield (fips, float(row[1]) * factor)
