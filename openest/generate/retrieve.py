import urllib2, StringIO
from ..models.ddp_model import DDPModel
from ..models.spline_model import SplineModel
from ..models.bin_model import BinModel

def from_url(url, create_func):
    '''
    Returns a :py:class:`StringIO.StringIO` buffer with the contents of the response from `url`

    .. todo::
        
        * response from `urllib2.urlopen(req)` is already a buffer. Is writing to a new buffer necessary?

    '''
    
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
    '''
    Returns a :py:class:`~.models.spline_model.SplineModel` from the argument `url`
    '''
    return from_url(url, lambda file: SplineModel().init_from(file, ','))

def ddp_from_url(url):
    '''
    Returns a :py:class:`~.models.ddp_model.DDPModel` from the argument `url`
    '''
    return from_url(url, lambda file: DDPModel().init_from(file, ',', source=url))

def any_from_url(url):
    '''
    Returns a model retrieved from the argument `url`

    `any_from_url` is a wrapper around :py:func:`~.generate.retrieve.from_url`. It 
    returns a model chosen by :py:func:`~.generate.retrieve.choose_model`. 
    Therefore, the file reader returned by :py:func:`~.generate.retrieve.from_url` 
    must have one of the allowed model types as the first four characters in the 
    document.

    Parameters
    ----------
    url : str
        URL of file to retrieve

    Returns
    -------
    object
        Model chosen by :py:func:`~.generate.retrieve.choose_model`
    '''


    return from_url(url, lambda file: choose_model(file, source=url))

def choose_model(fp, source=None):
    '''
    Reads a file object and returns a model based on file header

    The file is converted into a :py:class:`~.models.bin_model.BinModel`, 
    :py:class:`~.models.ddp_model.DDPModel`, or 
    :py:class:`~.models.spline_model.SplineModel` depending on the first four 
    characters of the file.

    To use choose_model, the first four characters of the file reader object 
    must be one of the following:

        * ``bin1``, in which case a :py:class:`~.models.bin_model.BinModel` will be returned, 
        * ``ddp1`` or ``ddp2``, in which case a :py:class:`~.models.ddp_model.DDPModel` will be returned, or 
        * ``spp1``, in which case a :py:class:`~.models.spline_model.SplineModel` will be returned.

    If the model type is not one of the types listed above, a `BaseException` 
    will be raised.

    .. todo::

        * Change exception type - sublcassing :py:class:`BaseException` is not PEP compliant. 
            Custom exceptions should inherrit from :py:class:`Exception` or 
            other built-in exceptions. This is so that ``except Exception`` will 
            catch all exceptions except for :py:class:`KeyboardInterrupt` and 
            :py:class:`SystemExit`, which are not errors, but user-triggered 
            events. See `PEP-352 <https://www.python.org/dev/peps/pep-0352/>`_.


    Parameters
    ----------
    fp : file reader object
        file reader object to be converted into a model. 

    source : str
        Meta-information about url the file was recovered from

    Returns
    -------
    object
        Model of class :py:class:`~.models.bin_model.BinModel`, 
        :py:class:`~.models.ddp_model.DDPModel`, or 
        :py:class:`~.models.spline_model.SplineModel`.

    
    '''

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


        
