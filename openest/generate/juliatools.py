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

    julia = julia_function(func, args)
    if julia:
        return {'main': FormatElement(julia, units, args)}
    
    if len(args) == 1:
        julia = julia_function(func, *args)
        funcvar = formatting.get_function()
        return {'main': FormatElement("%s(%s)" % (funcvar, args[0]), units,
                                     funcvar + "(x)"),
               funcvar + "(x)": FormatElement(description, units, args)}
    elif len(args) == 2:
        funcvar = formatting.get_function()
        return {'main': FormatElement("%s(%s, %s)" % (funcvar, args[0], args[1]),
                                      units, [funcvar + "(x, y)"]),
                funcvar + "(x, y)": FormatElement(description, units, args)}

def julia_function(func, *args):
    if len(args) == 1:
        interp = formatting.interpret1(func)
        if interp == 'identity':
            return args[0]

    if len(args) == 2:
        interp = formatting.interpret2(func)
        if interp == '/':
            return "(%s) / (%s)" % args
        elif interp == '-':
            return "%s - (%s)" % args
        elif interp == '*':
            return "(%s) * (%s)" % args
