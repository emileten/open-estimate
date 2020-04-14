import re
from collections import deque
import numpy as np

class FormatElement(object):
    def __init__(self, repstr, dependencies=None, is_abstract=False, is_primitive=False):
        self.repstr = repstr
        
        if dependencies is None:
            self.dependencies = []
        else:
            assert isinstance(dependencies, list) or isinstance(dependencies, set) or isinstance(dependencies, tuple)
            self.dependencies = dependencies
        
        self.is_abstract = is_abstract # Is this an English description?
        self.is_primitive = is_primitive # Can this be inserted straight into an equation?

    def __str__(self):
        return "FormatElement(\"%s\"; %s)" % (self.repstr, self.dependencies)

    def __repr__(self):
        return "FormatElement(\"%s\"; %s)" % (self.repstr, self.dependencies)

class ParameterFormatElement(FormatElement):
    def __init__(self, extname, repstr, dependencies=None):
        super(ParameterFormatElement, self).__init__(repstr, dependencies=dependencies)
        self.extname = extname

def build_format(reppattern, *formatargs):
    mainargs = []
    alldeps = []
    allelts = {}
    for formatarg in formatargs:
        assert not formatarg['main'].is_abstract
        mainargs.append(formatarg['main'].repstr)
        alldeps.extend(formatarg['main'].dependencies)
        allelts.update(formatarg)

    allelts['main'] = FormatElement(reppattern % tuple(mainargs), alldeps)
    return allelts

def build_recursive(reppatterns, lang, *formattableargs):
    assert lang in reppatterns
    formatargs = [arg.format(lang) for arg in formattableargs]
    return build_format(reppatterns[lang], *formatargs)

def build_adddepend(allelts, label, elt):
    allelts['main'].dependencies.append(label)
    allelts[label] = elt
        
def format_iterate(elements):
    main = elements['main']
    yield main # only case without key
    used_keys = set(['main'])
    queue = deque(main.dependencies)
    
    while queue:
        key = queue.popleft()
        if key in used_keys:
            continue

        if key not in elements:
            yield key, None
            continue

        used_keys.add(key)
        yield key, elements[key]
        queue.extend(elements[key].dependencies)

def format_latex(elements, parameters=None):
    if parameters is None:
        parameters = {}
    iter = format_iterate(elements)
    main = next(iter)
    content = "Main calculation\n\\[\n  %s\n\\]\n\n" % main.repstr

    content += "\\begin{description}"
    for key, element in iter:
        if element is None:
            pass
        elif isinstance(element, ParameterFormatElement):
            if element.extname in parameters:
                value = parameters[element.extname]
                if isinstance(value, np.ndarray):
                    valstr = '\\left(' + ', '.join(map(str, value)) + '\\right)'
                else:
                    valstr = str(value)
                content.append("\n  \\item[$%s$]\n    %s\n" % (key, valstr))
            else:
                content.append("\n  \\item[$%s$]\n    (parameter)\n" % key)
        elif element.is_abstract:
            content += "\n  \\item[$%s$]\n    %s\n" % (key, element.repstr)
        else:
            content += "\n  \\item[$%s$]\n    $%s$\n" % (key, element.repstr)
    content += "\\end{description}\n"

    return content

def format_julia(elements, parameters=None, include_comments=True):
    if parameters is None:
        parameters = {}
    iter = format_iterate(elements)
    main = next(iter)
    if include_comments:
        content = ["\n# Main calculation\n%s" % main.repstr]
    else:
        content = [main.repstr]
        
    for key, element in iter:
        if element is None:
            if include_comments:
                content.append("# %s missing" % key)
        elif isinstance(element, ParameterFormatElement):
            if element.extname in parameters:
                value = parameters[element.extname]
                if isinstance(value, np.ndarray):
                    if len(value.shape) == 2:
                        valstr = '[' + '; '.join([' '.join(map(juliarep, xxs)) for xxs in value]) + ']'
                    else:
                        valstr = '[' + ', '.join(map(juliarep, value)) + ']'
                else:
                    valstr = juliarep(value)
                if include_comments:
                    content.append("%s = %s # %s" % (element.repstr, valstr, key))
                else:
                    content.append("%s = %s" % (element.repstr, valstr))
            elif include_comments:
                content.append("# %s (parameter) %s" % (element.repstr, key))
        elif element.is_abstract:
            if include_comments:
                content.append("\n# %s:\n# %s" % (key, element.repstr))
        else:
            if include_comments:
                content.append("\n#:\n%s = %s" % (key, element.repstr))
            else:
                content.append("%s = %s" % (key, element.repstr))

    return "\n".join(reversed(content))

def juliarep(num):
    if isinstance(num, float) and np.isnan(num):
        return "missing"
    return str(num)

def format_reset():
    global functions_count, variables_count, format_labels, functions_known, betaonly_count

    functions_count = variables_count = betaonly_count = 0
    format_labels = []
    functions_known = {}

functions_count = 0
functions_vars = ['f', 'g', 'h']
functions_known = {}

variables_count = 0
variables_vars = ['x', 'y', 'z']

betaonly_count = 0

format_labels = []

def call_argvar(elt):
    if isinstance(elt, FormatElement):
        return get_variable(elt)
    else:
        return elt

def get_function(func=None):
    global functions_count, functions_known

    if func is not None:
        if func in functions_known:
            return functions_knowns[func]

    funcvar = functions_vars[functions_count % len(functions_vars)]
    if functions_count / len(functions_vars) > 0:
        funcvar += str(functions_count / len(functions_vars) + 1)
    functions_count += 1

    functions_known[func] = funcvar
    
    return funcvar

def get_variable(element=None):
    global variables_count

    if element and element.is_primitive:
        return element

    varvar = variables_vars[variables_count % len(variables_vars)]
    if variables_count / len(variables_vars) > 0:
        varvar += str(variables_count / len(variables_vars) + 1)
    variables_count += 1
    return varvar

def get_beta(lang):
    global betaonly_count
    betaonly_count += 1

    if lang == 'latex':
        return "\\beta_%d" % betaonly_count
    else:
        return "beta%d" % betaonly_count

def get_repstr(content):
    if isinstance(content, str):
        return content

    return content.repstr

def call_argvar(arg):
    if isinstance(arg, FormatElement):
        return get_variable()
    else:
        return arg

def get_parametername(extname, lang):
    if lang == 'julia':
        return re.sub('[^0-9a-zA-Z]', '_', extname)

    if lang == 'latex':
        return re.sub('[^0-9a-zA-Z]', '', extname)

def add_label(label, elements):
    format_labels.append((label, elements))
    
def interpret1(func):
    """
    Try to determine the processing of `func`.
    """
    try:
        if func('sillystring') == 'sillystring':
            return "identity"
    except Exception as ex:  # CATBELL
        import traceback; print("".join(traceback.format_exception(ex.__class__, ex, ex.__traceback__)))  # CATBELL
        return "unknown"

def interpret2(func):
    """
    Try to determine the processing of `func`.
    """
    try:
        if func(555., 111.) == 5.:
            return '/'
        if func(555., 111.) == 444.:
            return '-'
        if func(555., 111.) == 555. * 111.:
            return '*'
    except Exception as ex:  # CATBELL
        import traceback; print("".join(traceback.format_exception(ex.__class__, ex, ex.__traceback__)))  # CATBELL
        return "unknown"
