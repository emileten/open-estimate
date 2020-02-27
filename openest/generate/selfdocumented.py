from . import formatting, juliatools, latextools

def get_repstr(obj, lang):
    if isinstance(obj, SelfDocumenting):
        return obj.get_repstr(lang)

    return formatting.get_repstr(obj)

def get_dependencies(obj, lang):
    if isinstance(obj, SelfDocumenting):
        return obj.format(lang)['main'].dependencies

    return []

def format_nomain(obj, lang):
    if isinstance(obj, SelfDocumenting):
        result = obj.format(lang)
        result[obj] = result['main']
        del result['main']
        return result

    if isinstance(obj, str):
        return {obj: formatting.ParameterFormatElement(formatting.get_parametername(obj, lang), obj)}

    repstr = formatting.get_variable()
    return {repstr: formatting.FormatElement(repstr, "Unknown: %s" % obj)}

class SelfDocumenting(object):
    def get_repstr(self, lang):
        raise NotImplementedError()
    
    def format(self, lang):
        raise NotImplementedError()

class DocumentedFunction(SelfDocumenting):
    def __init__(self, func, description=None, docfunc=None, docargs=[]):
        self.func = func
        self.description = description
        self.docfunc = docfunc if docfunc is not None else func
        self.docargs = docargs # list may contain DocumentedFunction or value

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def get_repstr(self, lang):
        elements = self.format(lang)
        return elements['main'].repstr
        
    def format(self, lang):
        # Format the arguments
        argelts = []
        elements = {}
        alldeps = []
        for arg in self.docargs:
            if isinstance(arg, DocumentedFunction):
                elements.update(arg.format(lang))
                argelts.append(elements['main'])
                alldeps.extend(elements['main'].dependencies)
            elif isinstance(arg, str):
                elements[arg] = formatting.ParameterFormatElement(arg, formatting.get_parametername(arg, lang))
                argelts.append(formatting.get_parametername(arg, lang))
                alldeps.append(arg)
            else:
                argelts.append(formatting.FormatElement(str(arg), is_primitive=True))

        if self.description in formatting.functions_known:
            # This is within a documentation context like imperics.py
            knownelt = formatting.functions_known[self.description]
            if lang == 'latex':
                elements.update({'main': formatting.FormatElement("%s(%s)" % (knownelt.repstr, ', '.join(map(formatting.get_repstr, argelts))), [knownelt.extname] + alldeps)})
            else:
                elements.update({'main': formatting.FormatElement("%s(%s)" % (knownelt.repstr, ', '.join(map(formatting.get_repstr, argelts))), [knownelt.extname] + alldeps)})
                
            elements[knownelt.extname] = knownelt

            return elements
                
        if lang == 'latex':
            elements.update(latextools.call(self.docfunc, self.description, *argelts))
        elif lang == 'julia':
            elements.update(juliatools.call(self.docfunc, self.description, *argelts))
        else:
            raise RuntimeError("Unknown language %s" % lang)

        return elements
