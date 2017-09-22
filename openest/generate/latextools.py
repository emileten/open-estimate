import formatting
from formatting import FormatElement

def call(func, units, description=None, *args):
    """Return a representation of this call.  Any elements in args can
    be given their own FormatElements in the final dictionary.
    """
    
    if len(args) == 0:
        funcvar = formatting.get_function()
        return {'main': FormatElement(funcvar + "()", units, [funcvar + '()'], is_primitive=True),
                funcvar + '()': FormatElement(description, units, is_abstract=True)}

    latex = latex_function(func, *tuple(map(formatting.get_repstr, args)))
    if latex:
        return {'main': FormatElement(latex, units)}
    
    if len(args) == 1:
        funcvar = formatting.get_function()
        argvar = formatting.get_variable(args[0])

        if isinstance(argvar, FormatElement):
            return {'main': FormatElement("%s(%s)" % (funcvar, argvar.repstr), units,
                                          [funcvar + r"(\cdot)"] + argvar.dependencies, is_primitive=True),
                    funcvar + r"(\cdot)": FormatElement(description, units, is_abstract=True)}
        elif isinstance(args[0], FormatElement):
            return {'main': FormatElement("%s(%s)" % (funcvar, argvar), units,
                                          [funcvar + r"(\cdot)", argvar], is_primitive=True),
                    funcvar + r"(\cdot)": FormatElement(description, units, is_abstract=True),
                    argvar: args[0]}
        else:
            return {'main': FormatElement("%s(%s)" % (funcvar, args[0]), units,
                                          [funcvar + r"(\cdot)"], is_primitive=True),
                    funcvar + r"(\cdot)": FormatElement(description, units, is_abstract=True)}
    elif len(args) == 2:
        funcvar = formatting.get_function()

        if isinstance(args[0], FormatElement):
            argname0 = formatting.get_variable()
        else:
            argname0 = args[0]
        if isinstance(args[1], FormatElement):
            argname1 = formatting.get_variable()
        else:
            argname1 = args[1]
            
        result = {'main': FormatElement("%s(%s, %s)" % (funcvar, argname0, argname1),
                                        units, [funcvar + r"(\cdot)"], is_primitive=True),
                  funcvar + r"(\cdot)": FormatElement(description, units, is_abstract=True)}
        if isinstance(args[0], FormatElement):
            result[argname0] = args[0]
            result['main'].dependencies.append(argname0)
        if isinstance(args[1], FormatElement):
            result[argname1] = args[1]
            result['main'].dependencies.append(argname1)

        return result
            
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

