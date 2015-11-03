import fake_weather, generators, daily, create, effect_bundle

# Generate sample temperatures
yyyyddd, tasmax = fake_weather.make_constant_365years(2000, 303.15, 100)
yyyyddd, pr = fake_weather.make_constant_365years(2000, .01, 100)

# Get the models from DMAS
crime_violent_tasmax_url = "http://dmas.berkeley.edu/collection/retrieve_hierarchical_normal_merge_muid?collection_id=52d9aade434fd77f09ec4f62&simuids=53594263434fd733f817dd3d,53594263434fd733f817dd3f,53594263434fd733f817dd41,53594263434fd733f817dd43,53594263434fd733f817dd45,53594263434fd733f817dd47,53594263434fd733f817dd49,53594263434fd733f817dd4b,53594263434fd733f817dd4d,53594263434fd733f817dd4f,53594264434fd733f817dd51,53594263434fd733f817dd3e,53594263434fd733f817dd40,53594263434fd733f817dd42,53594263434fd733f817dd44,53594263434fd733f817dd46,53594263434fd733f817dd48,53594263434fd733f817dd4a,53594263434fd733f817dd4c,53594263434fd733f817dd4e,53594264434fd733f817dd50,53594264434fd733f817dd52"
crime_violent_pr_url = "http://dmas.berkeley.edu/collection/retrieve_hierarchical_normal_merge_muid?collection_id=53228e61434fd72e07279471&simuids=53594eb9434fd733f817dd53,53594eb9434fd733f817dd55,53594eb9434fd733f817dd57,53594eb9434fd733f817dd59,53594eb9434fd733f817dd5b,53594eb9434fd733f817dd54,53594eb9434fd733f817dd56,53594eb9434fd733f817dd58,53594eb9434fd733f817dd5a,53594eb9434fd733f817dd5c"

model_tasmax = create.ddp_from_url(crime_violent_tasmax_url)
model_pr = create.ddp_from_url(crime_violent_pr_url)

# Run a generator that just calculates the effect of temperature on crime
make_generator = daily.make_daily_bymonthdaybins(model_tasmax, lambda x: 1 + x / 100.0, .5)

for values in effect_bundle.yield_given('crime-violent', yyyyddd, tasmax, make_generator):
    print ','.join(map(str, values))

# Run a generator that computes the effect of temp. and precip. on crime
make_generator = generators.make_instabase(generators.make_product(['tasmax', 'pr'], [
    daily.make_daily_bymonthdaybins(model_tasmax, lambda x: 1 + x / 100.0, .5),
    daily.make_daily_bymonthdaybins(model_pr, lambda x: 1 + x / 100.0, .5, lambda x: x * (x > 0))]),
                               2012)

for values in effect_bundle.yield_given('crime-violent', yyyyddd, {'tasmax': tasmax, 'pr': pr}, make_generator):
    print ','.join(map(str, values))
