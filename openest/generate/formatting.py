from collections import deque

class FormatElement(object):
    def __init__(self, repstr, unit, dependencies=[], is_abstract=False, is_primitive=False):
        self.repstr = repstr
        self.unit = unit
        self.dependencies = dependencies
        self.is_abstract = is_abstract # Is this an English description?
        self.is_primitive = is_primitive # Can this be inserted straight into an equation?

    def __str__(self):
        return "FormatElement(\"%s\")" % self.repstr

    def __repr__(self):
        return "FormatElement(\"%s\")" % self.repstr
        

def format_iterate(calculation, format):
    elements = calculation.format(format)

    main = elements['main']
    yield main # only case without key
    used_keys = set(['main'])
    queue = deque(main.dependencies)
    
    while queue:
        key = queue.popleft()
        if key in used_keys or key not in elements:
            continue

        used_keys.add(key)
        yield key, elements[key]
        queue.extend(elements[key].dependencies)

def format_latex(calculation):
    format_reset()
    iter = format_iterate(calculation, 'latex')
    main = iter.next()
    content = "Main calculation (Units: %s)\n\\[\n  %s\n\\]\n\n" % (main.unit, main.repstr)

    content += "\\begin{description}"
    for key, element in iter:
        if element.is_abstract:
            content += "\n  \\item[$%s$ (%s)]\n    %s\n" % (key, element.unit, element.repstr)
        else:
            content += "\n  \\item[$%s$ (%s)]\n    $%s$\n" % (key, element.unit, element.repstr)
    content += "\\end{description}\n"

    return content

def format_julia(calculation):
    format_reset()
    iter = format_iterate(calculation, 'julia')
    main = iter.next()
    content = ["\n# Main calculation [%s]\n%s" % (main.unit, main.repstr)]
    
    for key, element in iter:
        if element.is_abstract:
            content.append("\n# %s [%s]:\n# %s" % (key, element.unit, element.repstr))
        else:
            content.append("\n# [%s]:\n%s = %s" % (element.unit, key, element.repstr))

    return "\n".join(reversed(content))

def format_reset():
    global functions_count, variables_count

    functions_count = variables_count = 0

functions_count = 0
functions_vars = ['f', 'g', 'h']

variables_count = 0
variables_vars = ['x', 'y', 'z']

def get_function():
    global functions_count
    
    funcvar = functions_vars[functions_count % len(functions_vars)]
    if functions_count / len(functions_vars) > 0:
        funcvar += str(functions_count / len(functions_vars) + 1)
    functions_count += 1
    return funcvar

def get_variable(element=None):
    global variables_count

    if element and element.is_primitive:
        return element
    
    varvar = variables_vars[variables_count % len(variables_vars)]
    if variables_count / len(variables_vars) > 0:
        funcvar += str(variables_count / len(variables_vars) + 1)
    variables_count += 1
    return varvar

def get_repstr(content):
    if isinstance(content, str):
        return content

    return content.repstr

def interpret1(func):
    """
    Try to determine the processing of `func`.
    """
    try:
        if func('sillystring') == 'sillystring':
            return "identity"
    except:
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
    except:
        return "unknown"
