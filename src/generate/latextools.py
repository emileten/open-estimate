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
    elif len(args) == 1:
        interp = interpret1(func)
        if interp == 'identity':
            yield ("Equation", args[0], units)
        else:
            funcvar = get_function()
            yield ("Equation", "%s(%s)" % (funcvar, args[0]), units)
            yield (funcvar + r"(\cdot)", description, units)
    elif len(args) == 2:
        interp = interpret2(func)
        if interp == '/':
            yield ("Equation", r"\frac{%s}{%s}" % args, units)
        elif interp == '-':
            yield ("Equation", r"%s - %s" % args, units)
        elif interp == '*':
            yield ("Equation", r"\left(%s\right) \left(%s\right)" % args, units)
        else:
            funcvar = get_function()
            yield ("Equation", "%s(%s, %s)" % (funcvar, args[0], args[1]), units)
            yield (funcvar + r"(\cdot)", description, units)

"""
Try to determine the processing of `func`.
"""
def interpret1(func):
    try:
        if func('sillystring') == 'sillystring':
            return "identity"
    except:
        return "unknown"

"""
Try to determine the processing of `func`.
"""
def interpret2(func):
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
