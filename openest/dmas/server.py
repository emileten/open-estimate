import urllib.request, urllib.error, urllib.parse

# The domain we include in emails and such
public_domain = "dmas.berkeley.edu"

def domain_url(location):
    if location[0] != '/':
        location = '/' + location
    return 'http://' + public_domain + location

def open_url(url):
    """
    Returns an open file to the URL.
    """
    if len(url) < 7 or url[0:7] != "http://":
        url = domain_url(url)

    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
           'Accept-Encoding': 'none',
           'Accept-Language': 'en-US,en;q=0.8'}

    req = urllib.request.Request(url, headers=hdr)
    return urllib.request.urlopen(req)


