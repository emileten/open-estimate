import urllib2, StringIO
from ..models.ddp_model import DDPModel
from ..models.spline_model import SplineModel
from ..models.bin_model import BinModel

def from_url(url, create_func):
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
           'Accept-Encoding': 'none',
           'Accept-Language': 'en-US,en;q=0.8'}

    print url

    req = urllib2.Request(url, headers=hdr)
    fp = urllib2.urlopen(req)
    output = StringIO.StringIO()
    output.write(fp.read())
    output.seek(0)

    model = create_func(output)

    fp.close()
    output.close()

    return model

def spline_from_url(url):
    return from_url(url, lambda file: SplineModel().init_from(file, ','))

def ddp_from_url(url):
    return from_url(url, lambda file: DDPModel().init_from(file, ',', source=url))

def any_from_url(url):
    return from_url(url, lambda file: choose_model(file, source=url))

def choose_model(fp, source=None):
    alltext = fp.read()
    model_type = alltext[:4]
    fp = StringIO.StringIO(alltext)
    if model_type == 'bin1':
        return BinModel().init_from_bin_file(fp, ',', init_submodel=choose_model)
    if model_type in ['ddp1', 'ddp2']:
        return DDPModel().init_from(fp, ',', source=url)
    if model_type == 'spp1':
        return SplineModel().init_from_spline_file(fp, ',')

    raise BaseException("Unknown model type: " + model_type)


        
