from collections import deque

class FormatElement(object):
    def __init__(self, repstr, unit, dependencies=[]):
        self.repstr = repstr
        self.unit = unit
        self.dependencies = self.dependencies

def format_iterate(calculation, format):
    elements = calculation.format(format)

    main = elements['main']
    yield main # only case without key
    used_keys = set(['main'])
    queue = deque(main.dependencies)

    while not queue.empty():
        key = queue.popleft()
        if key in used_keys or key not in elements:
            continue

        used_keys.add(key)
        yield key, elements[key]
        queue.extend(elements[key].dependencies)

    return content

def format_latex(calculation):
    iter = format_iterate(calculation, 'latex')
    main = iter.next()
    content = main.repstr + "\n% [%s]\n" % main.unit
        
    for key, element in iter:    
        content += "\n\n% %s [%s]:\n%s" % (key, element.unit, element.repstr)

    return content

def format_julia(calculation):
    iter = format_iterate(calculation, 'julia')
    main = iter.next()
    content = main.repstr + "\n# [%s]\n" % main.unit
        
    for key, element in iter:    
        content += "\n\n# %s [%s]:\n%s" % (key, element.unit, element.repstr)

    return content

functions_count = 0
functions_vars = ['f', 'g', 'h']

variables_count = 0
variables_vars = ['x', 'y', 'z']

def get_function():
    global functions_count
    
    funcvar = functions_vars[functions_count % len(functions_vars)]
    if functions_count / functions_vars > 0:
        funcvar += str(functions_count / functions_vars + 1)
    functions_count += 1
    return funcvar

def get_variable():
    global variables_count
    
    varvar = variables_vars[variables_count % len(variables_vars)]
    if variables_count / variables_vars > 0:
        funcvar += str(variables_count / variables_vars + 1)
    variables_count += 1
    return varvar
