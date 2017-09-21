import formatting
from formatting import FormatElement

def call(func, units, description=None, *args):
    """Return a representation of this call.  Any elements in args can
    be given their own FormatElements in the final dictionary.
    """
    
    if len(args) == 0:
        funcvar = formatting.get_function()
        return {'main': FormatElement(funcvar + "()", units, [funcvar + '()']),
                funcvar + '()': FormatElement(description, units)}

    latex = latex_function(func, args)
    if latex:
        return {'main': FormatElement(latex, units, args)}
    
    if len(args) == 1:
        latex = latex_function(func, *args)
        funcvar = formatting.get_function()
        return {'main': FormatElement("%s(%s)" % (funcvar, args[0]), units,
                                     funcvar + r"(\cdot)"),
               funcvar + r"(\cdot)": FormatElement(description, units, args)}
    elif len(args) == 2:
        funcvar = formatting.get_function()
        return {'main': FormatElement("%s(%s, %s)" % (funcvar, args[0], args[1]),
                                      units, [funcvar + r"(\cdot)"]),
                funcvar + r"(\cdot)": FormatElement(description, units, args)}

def latex_function(func, *args):
    if len(args) == 1:
        interp = formatting.interpret1(func)
        if interp == 'identity':
            return args[0]

    if len(args) == 2:
        interp = formatting.interpret2(func)
        if interp == '/':
            return r"\frac{%s}{%s}" % args
        elif interp == '-':
            return r"%s - %s" % args
        elif interp == '*':
            return r"\left(%s\right) \left(%s\right)" % args

def english_function(func, *args):
    if len(args) == 1:
        interp = formatting.interpret1(func)
        if interp == 'identity':
            return args[0]

    if len(args) == 2:
        interp = formatting.interpret2(func)
        if interp == '/':
            return r"%s / %s" % args
        elif interp == '-':
            return r"%s - %s" % args
        elif interp == '*':
            return r"%s x %s" % args

