import fake_weather, generators, daily

# Generate sample temperatures
yyyyddd, tasmax = fake_weather.make_constant_365years(2000, 30, 100)
yyyyddd, pr = fake_weather.make_constant_365years(2000, .01, 100)

# Get the models
crime_violent_tasmax_url = "http://dmas.berkeley.edu/collection/retrieve_hierarchical_normal_merge_muid?collection_id=52d9aade434fd77f09ec4f62&simuids=53594263434fd733f817dd3d,53594263434fd733f817dd3f,53594263434fd733f817dd41,53594263434fd733f817dd43,53594263434fd733f817dd45,53594263434fd733f817dd47,53594263434fd733f817dd49,53594263434fd733f817dd4b,53594263434fd733f817dd4d,53594263434fd733f817dd4f,53594264434fd733f817dd51,53594263434fd733f817dd3e,53594263434fd733f817dd40,53594263434fd733f817dd42,53594263434fd733f817dd44,53594263434fd733f817dd46,53594263434fd733f817dd48,53594263434fd733f817dd4a,53594263434fd733f817dd4c,53594263434fd733f817dd4e,53594264434fd733f817dd50,53594264434fd733f817dd52"
crime_violent_pr_url = "http://dmas.berkeley.edu/collection/retrieve_hierarchical_normal_merge_muid?collection_id=53228e61434fd72e07279471&simuids=53594eb9434fd733f817dd53,53594eb9434fd733f817dd55,53594eb9434fd733f817dd57,53594eb9434fd733f817dd59,53594eb9434fd733f817dd5b,53594eb9434fd733f817dd54,53594eb9434fd733f817dd56,53594eb9434fd733f817dd58,53594eb9434fd733f817dd5a,53594eb9434fd733f817dd5c"

def create_from_url(url, create_func):
    if len(url) < 7 or url[0:7] != "http://":
        url = "http://" + server.public_domain + url

    print url

    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
           'Accept-Encoding': 'none',
           'Accept-Language': 'en-US,en;q=0.8'}

    req = urllib2.Request(url, headers=hdr)
    fp = urllib2.urlopen(req)
    output = StringIO.StringIO()
    output.write(fp.read())
    output.seek(0)

    model = create_func(output)

    fp.close()
    output.close()

    return model

model_tasmax = create_from_url(crime_violent_tasmax_url, lambda file: SplineModel().init_from_spline_file(file, ','))
model_pr = create_from_url(crime_violent_pr_url, lambda file: SplineModel().init_from_spline_file(file, ','))

generator = gnerator.instabase(generators.make_product(['tasmax', 'pr'], [
    daily.make_daily_bymonthdaybins(model_tasmax, lambda x: 1 + x / 100.0, .5),
    daily.make_daily_bymonthdaybins(model_pr, lambda x: 1 + x / 100.0, .5, lambda x: x * (x > 0))]),
                               2012)

for values in effect_bundle.yield_given('crime-violent', yyyyddd, {'tasmas': tasmax, 'pr': pr}):
    print values

