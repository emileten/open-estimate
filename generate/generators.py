import numpy as np
import effect_bundle

def make_scale(make_generator, scale_dict, func=lambda x, y: x*y):
    """Scale the results by the value in scale_dict, or the mean value (if it is set).
    make_generator: we encapsulate this function, passing in data and opporting on outputs
    func: default operation is to multiple (scale), but can do other things (e.g., - for re-basing)
    """

    def generate(fips, yyyyddd, temps, **kw):
        # Prepare the generator from our encapsulated operations
        generator = make_generator(fips, yyyyddd, temps, **kw)
        # Scale each result
        for (year, result) in generator:
            if fips in scale_dict:
                yield (year, func(result, scale_dict[fips]))
            else:
                yield (year, func(result, scale_dict['mean']))

    return generate


## make-apply logic for generating make_generators

def make(handler, make_generator, *handler_args, **handler_kw):
    """Construct a generator from a function, operating on the results of another generator.
    handler(generator, *handler_args, **handler_kw) takes an enumerator and returns an enumerator
    """

    # The make_generator function to return
    def generate(fips, yyyyddd, temps, *args, **kw):
        if fips == effect_bundle.FIPS_COMPLETE:
            # Pass on signal for end
            print "completing make"
            make_generator(fips, yyyyddd, temps, *args, **kw).next()
            return

        # Pass on data
        generator = make_generator(fips, yyyyddd, temps, *args, **kw)
        # Apply function to results of data
        for yearresult in handler(generator, *handler_args, **handler_kw):
            yield yearresult

    return generate

def apply(generator, func, unshift=False):
    """Apply a non-enumerator to all elements of a function.
    if unshift, tack on the result to the front of a sequence of results.
    Calls func with each year and value; returns the newly computed value
    """

    for yearresult in generator:
        # Call func to get a new value
        newresult = func(yearresult[0], yearresult[1])

        # Construct a new year, value result
        if unshift:
            yield [yearresult[0], newresult] + yearresult[1:]
        else:
            yield (yearresult[0], newresult)

def make_instabase(make_generator, baseyear, func=lambda x, y: x / y):
    """Re-base the results of make_generator(...) to the values in baseyear
    Default func constructs a porportional change; x - y makes simple difference.
    """

    # Use the instabase function to do operations
    return make(instabase, make_generator, baseyear, func)

def instabase(generator, baseyear, func=lambda x, y: x / y, skip_on_missing=True):
    """Re-base the results of make_generator(...) to the values in baseyear
    baseyear is the year to use as the 'denominator'; None for the first year
    Default func constructs a porportional change; x - y makes simple difference.
    skip_on_missing: If we never encounter the year and this is false,
      still print out the existing results.
    Tacks on the value to the front of the results
    """

    denom = None # The value in the baseyear
    pastresults = [] # results before baseyear

    for yearresult in generator:
        year = yearresult[0]
        result = yearresult[1]

        # Should we base everything off this year?
        if year == baseyear or (baseyear is None and denom is None):
            denom = result

            # Print out all past results, relative to this year
            for pastresult in pastresults:
                yield [pastresult[0], func(pastresult[1], denom)] + list(pastresult[1:])

        if denom is None:
            # Keep track of this until we have a base
            pastresults.append(yearresult)
        else:
            # calculate this and tack it on
            yield [year, func(result, denom)] + list(yearresult[1:])

    if denom is None and skip_on_missing:
        # Never got to this year: just write out results
        for pastresult in pastresults:
            yield pastresult

def make_runaverage(make_generator, priors, weights, unshift=False):
    """Generate results as an N-year running average;
    priors: list of size N, with the values to use before we get data
    weights: list of size N, with the weights of the years (earliest first)
    """

    # Use the runaverage function to do all the operations
    return make(runaverage, make_generator, priors, weights, unshift)

def runaverage(generator, priors, weights, unshift=False):
    """Generate results as an N-year running average;
    priors: list of size N, with the values to use before we get data (first value not used)
    weights: list of size N, with the weights of the years (earliest first)
    unshift: if true, tack on result at front of result list
    """

    values = list(priors) # Make a copy of the priors list
    totalweight = sum(weights) # Use as weight denominator

    for yearresult in generator:
        # The set of values to average ends with the new value
        values = values[1:] + [yearresult[1]]
        # Calculate weighted average
        smoothed = sum([values[ii] * weights[ii] for ii in range(len(priors))]) / totalweight

        # Produce the new result
        if unshift:
            yield [yearresult[0], smoothed] + list(yearresult[1:])
        else:
            yield (yearresult[0], smoothed)

def make_weighted_average(make_generators, weights):
    """This produces a weighted average of results from *multiple generators*.
    make_generators: list of make_generator functions; all must produce identical years
    weights: list of weight dictionaries ({FIPS: weight})
    len(make_generators) == len(weights)
    """

    def generate(fips, yyyyddd, weather, **kw):
        # Is this county represented in any of the weight dictionaries?
        inany = False
        for weight in weights:
            if fips in weight and weight[fips] > 0:
                inany = True
                break

        if not inany:
            return # Produce no results

        # Construct a list of generators from make_generators
        generators = [make_generators[ii](fips, yyyyddd, weather, **kw) for ii in range(len(make_generators))]

        # Iterate through all generators simultaneously
        for values in generators[0]:
            # Construct a list of the value from each generator
            values = [values] + [generator.next() for generator in generators[1:]]
            # Ensure that year is identical across all
            for ii in range(1, len(generators)):
                assert(values[0][0] == values[ii][0])

            # Construct (year, result) where result is a weighted average using weights
            yield (values[0][0], np.sum([values[ii][1] * weights[ii].get(fips, 0) for ii in range(len(generators))]) /
                   np.sum([weights[ii].get(fips, 0) for ii in range(len(generators))]))

    return generate

def make_product(vars, make_generators):
    """This produces a product of results from *multiple generators*.
    vars: a list of single variables to pass into each generator
    make_generators: list of make_generator functions; all must produce identical years
    len(make_generators) == len(vars)
    """

    def generate(fips, yyyyddd, weather, **kw):
        if fips == effect_bundle.FIPS_COMPLETE:
            # Pass on signal for end
            for make_generator in make_generators:
                make_generator(fips, yyyyddd, weather, **kw).next()
            return

        # Construct a list of generators from make_generators
        generators = [make_generators[ii](fips, yyyyddd, weather[vars[ii]], **kw) for ii in range(len(make_generators))]

        # Iterate through all generators simultaneously
        for values in generators[0]:
            values = [values] + [generator.next() for generator in generators[1:]]
            # Ensure that year is identical across all
            for ii in range(1, len(generators)):
                assert(values[0][0] == values[ii][0])

            # Construct (year, result) where result is a product
            yield (values[0][0], np.product([values[ii][1] for ii in range(len(generators))]))

    return generate
