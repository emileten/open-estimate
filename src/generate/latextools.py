functions_count = 0
functions_vars = ['f', 'g', 'h']

variables_count = 0
variables_vars = ['x', 'y', 'z']

"""
Return a representation of this call.
"""
def call(func, units, description=None, *args):
    if len(args) == 0:
        funcvar = get_function()
        yield ("Equation", funcvar + "()", units)
        yield (funcvar + "()", description, units)
        return

    latex = latex_function(func, args)
    if latex:
        yield ("Equation", latex, units)
        return
    
    if len(args) == 1:
        latex = latex_function(func, *args)
        funcvar = get_function()
        yield ("Equation", "%s(%s)" % (funcvar, args[0]), units)
        yield (funcvar + r"(\cdot)", description, units)
    elif len(args) == 2:
        funcvar = get_function()
        yield ("Equation", "%s(%s, %s)" % (funcvar, args[0], args[1]), units)
        yield (funcvar + r"(\cdot)", description, units)

def latex_function(func, *args):
    if len(args) == 1:
        interp = interpret1(func)
        if interp == 'identity':
            return args[0]

    if len(args) == 2:
        interp = interpret2(func)
        if interp == '/':
            return r"\frac{%s}{%s}" % args
        elif interp == '-':
            return r"%s - %s" % args
        elif interp == '*':
            return, r"\left(%s\right) \left(%s\right)" % args

def english_function(func, *args):
    if len(args) == 1:
        interp = interpret1(func)
        if interp == 'identity':
            return args[0]

    if len(args) == 2:
        interp = interpret2(func)
        if interp == '/':
            return r"%s / %s" % args
        elif interp == '-':
            return r"%s - %s" % args
        elif interp == '*':
            return, r"%s x %s" % args

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

def get_function():
    global functions_count
    
    funcvar = functions_vars[functions_count]
    functions_count += 1
    return funcvar
