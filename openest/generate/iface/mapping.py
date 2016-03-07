import generators, daily, fake

generator_mapping = {
    'scale': generators.make_scale,
    'instabase': generators.make_instabase,
    'running_average': generators.make_runaverage,
    'weighted_average': generators.make_weighted_average,
    'product': generators.make_product,
    'daily_monthlydaybins': daily.make_daily_bymonthdaybins,
    'daily_yearlydaybins': daily.make_daily_yearlydaybins,
    'daily_averagemonth': daily.make_daily_averagemonth,
    'daily_percent_within': daily.make_daily_percentwithin,
    'linear': fake.make_generator_linear,
    'bilinear': fake.make_generator_bilinear
}

