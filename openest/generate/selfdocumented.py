import formatting, juliatools, latextools

def get_repstr(obj, lang):
    if isinstance(obj, SelfDocumenting):
        return obj.get_repstr(lang)

    return formatting.get_repstr(obj)

def format(obj, lang):
    if isinstance(obj, SelfDocumenting):
        return obj.format(lang)

    if isinstance(obj, str):
        return {obj: formatting.ParameterFormatElement(formatting.get_parametername(obj, lang), obj, None)}

    repstr = formatting.get_variable()
    return {repstr: formatting.FormatElement(repstr, None, "Unknown: %s" % obj)}

class SelfDocumenting(object):
    def get_repstr(self, lang):
        raise NotImplementedError()
    
    def format(self, lang):
        raise NotImplementedError()

class DocumentedFunction(SelfDocumenting):
    def __init__(self, func, unit, description=None, docargs=[]):
        self.func = func
        self.unit = unit
        self.description = description
        self.docargs = docargs # {name -> FormatElement}, my claimed arguments (may not match actual)

    def __call__(self, *args, **kwargs):
        self.func(*args, **kwargs)

    def get_repstr(self, lang):
        elements = self.format(lang)
        return elements['main'].repstr
        
    def format(self, lang):
        if lang == 'latex':
            return latextools.call(self.func, self.unit, self.description, *self.docargs)
        elif lang == 'julia':
            return juliatools.call(self.func, self.unit, self.description, *self.docargs)
        else:
            raise RuntimeError("Unknown language %s" % lang)
