from .formatting import FormatElement

def join(glue, subelementsets):
    dependencies = []
    elements = {}
    mains = []
    for subelements in subelementsets:
        dependencies.extend(subelements['main'].dependencies)
        elements.update(subelements)
        mains.append(subelements['main'].repstr)

    elements['main'] = FormatElement(glue.join(mains), dependencies)
    return elements
