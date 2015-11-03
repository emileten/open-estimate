# -*- coding: utf-8 -*-
################################################################################
# Copyright 2014, Distributed Meta-Analysis System
################################################################################

"""Software structure for generating Monte-Carlo collections of results.

NOTE: Highest resolution regions are implicitly assumed to be
FIPS-coded counties, but the logic does not require them to be.  FIPS
language should be replaced with generic ID references.

A key structure is the make_generator(fips, times, values) function.
make_generator is passed to the functions that iterate through
different weather forecasts, such as make_tar_ncdf.  It is then called
with each location and daily weather data.  fips is a single county
code, times is a list of yyyyddd formated date values, values is a
list of weather values.

The output of make_generator() is a generator, producing tuples (year,
effect), for whichever years an effect can be computed.

Output file structure:

Each bundle of output impact results of a given type and for a given
weather forecast are in a gzipped tar file containing a single
directory <name>, containing a separate csv file (an effect file) for each
region.  The format of the csv file is:
  year,<label>[,<other labels>]*
  <year>,<impact>[,<prior calculated impact>]*

Basic processing logic:

Some functions, like find_ncdfs_allreal, discover collections of
forecasted variables (within the WDS directory structure), and provide
through enumerators.  Variable collections are dictionaries {variable:
REFERENCE}, where REFERENCE may be a filename, a netCDF, or a
dictionary of {original: netCDF object, data: [days x counties],
times: [yyyyddd]}. [VRD]

Controllers (elsewhere) loop through these, and for each available
forecast call a make_tar_* function passing in a make_generator
function.  The make_tar_* functions call make_generator with each
individual region, retrieving a set of results, and then package those
results into the output file format.

Temporary directories (characterized by random letters) are used to
hold the results as they're being generated (before being bundled into
tars).
"""

__copyright__ = "Copyright 2014, Distributed Meta-Analysis System"

__author__ = "James Rising"
__credits__ = ["James Rising"]
__maintainer__ = "James Rising"
__email__ = "jar2234@columbia.edu"

__status__ = "Production"
__version__ = "$Revision$"
# $Source$

import tarfile, os, csv, re, random, string
import numpy as np
try:
    # this is required for nc4's, but we can wait to fail
    from netCDF4 import Dataset
except:
    pass

FIPS_COMPLETE = '__complete__' # special FIPS code for the last county

### Effect Bundle Generation

## Temporary directory management

def enter_local_tempdir(prefix=''):
    """Create and set the working directory as a new temporary directory.

    Returns the name of the temporary directory (to be passed to
    exit_local_tempdir).
    """

    suffix = ''.join(random.choice(string.lowercase) for i in range(6))

    os.mkdir(prefix + suffix)
    os.chdir(prefix + suffix)

    return prefix + suffix

def exit_local_tempdir(tempdir, killit=True):
    """Return to the root output directory (and optionally delete the
    temporary directory).

    tempdir is the output of enter_local_tempdir.
    """

    os.chdir("..")
    if killit:
        kill_local_tempdir(tempdir)

def kill_local_tempdir(tempdir):
    """Remove all contents of a temporary directory.

    Call after exit_local_tempdir is called, only if killit=False.
    """

    os.system("rm -r " + tempdir)

## General helper functions for creation

def send_fips_complete(make_generator):
    """Call after the last county of a loop of counties, to clean up any memory.
    """

    print "Complete the FIPS"
    try:
        iterator = make_generator(FIPS_COMPLETE, None, None).next()
        print "Success"
    except StopIteration, e:
        pass
    except Exception, e:
        print e
        pass

def get_target_path(targetdir, name):
    """Helper function to use the targetdir directory if its provided.
    """

    if targetdir is not None:
        return os.path.join(targetdir, name)
    else:
        return name

def write_effect_file(path, fips, generator, collabel):
    """Write the effects for a single FIPS-coded county.

    path: relative path for file
    fips: the unique id of the region
    generator: a enumerator of tuples/lists with individual rows
    collabel: label for one (string) or more (list) columns after the
    year column
    """

    # Create the CSV file
    with open(os.path.join(path, fips + '.csv'), 'wb') as csvfp:
        writer = csv.writer(csvfp, quoting=csv.QUOTE_MINIMAL)

        # Write the header row
        if not isinstance(collabel, list):
            writer.writerow(["year", collabel])
        else:
            writer.writerow(["year"] + collabel)

        # Write all data rows
        for values in generator:
            writer.writerow(values)

## Top-level bundle creation functions

def make_tar_dummy(name, acradir, make_generator, targetdir=None, collabel="fraction"):
    """Constructs a tar of files for each county, using NO DATA.
    Calls make_generator for each county, using a filename of
    counties.

    name: the name of the effect bundle.
    acradir: path to the DMAS acra directory.
    make_generator(fips, times, daily): returns an iterator of (year, effect).
    targetdir: path to a final destination for the bundle
    collabel: the label for the effect column
    """

    tempdir = enter_local_tempdir()
    os.mkdir(name) # directory for county files

    # Generate a effect file for each county in regionsA
    with open(os.path.join(acradir, 'regions/regionsANSI.csv')) as countyfp:
        reader = csv.reader(countyfp)
        reader.next() # ignore header

        # Each row is a county
        for row in reader:
            fips = canonical_fips(row[0])
            print fips

            # Call generator (with no data)
            generator = make_generator(fips, None, None)
            if generator is None:
                continue

            # Construct the effect file
            write_effect_file(name, fips, generator, collabel)

    send_fips_complete(make_generator)

    # Generate the bundle tar
    target = get_target_path(targetdir, name)
    os.system("tar -czf " + os.path.join("..", target) + ".tar.gz " + name)

    # Remove the working directory
    exit_local_tempdir(tempdir)

def make_tar_duplicate(name, filepath, make_generator, targetdir=None, collabel="fraction"):
    """Constructs a tar of files for each county that is described in
    an existing bundle.  Passes NO DATA to make_generator.

    name: the name of the effect bundle.
    filepath: path to an existing effect bundle
    make_generator(fips, times, daily): returns an iterator of (year, effect).
    targetdir: path to a final destination for the bundle
    collabel: the label for the effect column
    """

    tempdir = enter_local_tempdir()
    os.mkdir(name)

    # Iterate through all FIPS-titled files in the effect bundle
    with tarfile.open(filepath) as tar:
        for item in tar.getnames()[1:]:
            fips = item.split('/')[1][0:-4]
            print fips

            # Call make_generator with no data
            generator = make_generator(fips, None, None)
            if generator is None:
                continue

            # Construct the effect file
            write_effect_file(name, fips, generator, collabel)

    send_fips_complete(make_generator)

    # Generate the bundle tar
    target = get_target_path(targetdir, name)
    os.system("tar -czf " + os.path.join("..", target) + ".tar.gz " + name)

    # Remove the working directory
    exit_local_tempdir(tempdir)

def make_tar_ncdf(name, weather_ncdf, var, make_generator, targetdir=None, collabel="fraction"):
    """Constructs a tar of files for each county, describing yearly results.

    name: the name of the effect bundle.
    weather_ncdf: str for one, or {variable: filename} for calling
      generator with {variable: data}.
    var: str for one, or [str] for calling generator with {variable: data}
    make_generator(fips, times, daily): returns an iterator of (year, effect).
    targetdir: path to a final destination for the bundle, or a
      function to take the data
    collabel: the label for the effect column
    """

    # If this is a function, we just start iterating
    if hasattr(targetdir, '__call__'):
        call_with_generator(name, weather_ncdf, var, make_generator, targetdir)
        return

    # Create the working directory
    tempdir = enter_local_tempdir()
    os.mkdir(name)

    # Helper function for calling write_effect_file with collabel
    def write_csv(name, fips, generator):
        write_effect_file(name, fips, generator, collabel)

    # Iterate through the data
    call_with_generator(name, weather_ncdf, var, make_generator, write_csv)

    # Create the effect bundle
    target = get_target_path(targetdir, name)
    os.system("tar -czf " + os.path.join("..", target) + ".tar.gz " + name)

    # Remove the working directory
    exit_local_tempdir(tempdir)

def yield_given(name, yyyyddd, weather, make_generator):
    """Yields (as an iterator) rows of the result of applying make_generator to the given weather.

    name: the name of the effect bundle.
    yyyyddd: YYYYDDD formated date values.
    weather: a dictionary to call generator with {variable: data}.
    make_generator(fips, times, daily): returns an iterator of (year, effect).
    """
    generator = make_generator(0, yyyyddd, weather)
    if generator is None:
        return

    # Call targetfunc with the result
    for values in generator:
        yield values

    # Signal the end of the counties
    send_fips_complete(make_generator)

def call_with_generator(name, weather_ncdf, var, make_generator, targetfunc):
    """Helper function for calling make_generator with each variable
    set.  In cases with multiple weather datasets, assumes all use the
    same clock (sequence of times) and geography (sequence of
    counties).

    name: the name of the effect bundle.
    weather_ncdf: str for one, or {variable: filename} for calling
      generator with {variable: data}.
    var: str for one, or [str] for calling generator with {variable: data}
    make_generator(fips, times, daily): returns an iterator of (year, effect).
    targetfunc: function(name, fips, generator) to handle results
    """

    if isinstance(weather_ncdf, dict) and isinstance(var, list):
        # In this case, we generate a dictionary of variables
        weather = {}
        times = None # All input assumed to have same clock

        # Filter by the variables in var
        for variable in var:
            # Retrieve the netcdf object (rootgrp) and add to weather dict
            if isinstance(weather_ncdf[variable], str):
                # Open this up as a netCDF and read data into array
                rootgrp = Dataset(weather_ncdf[variable], 'r+', format='NETCDF4')
                weather[variable] = rootgrp.variables[variable][:,:]
            elif isinstance(weather_ncdf[variable], dict):
                # This is an {original, data, times} dictionary
                rootgrp = weather_ncdf[variable]['original']
                weather[variable] = weather_ncdf[variable]['data']
                if 'times' in weather_ncdf[variable]:
                    times = weather_ncdf[variable]['times']
            else:
                # This is already a netcdf object
                rootgrp = weather_ncdf[variable]
                weather[variable] = rootgrp.variables[variable][:,:]

            # Collect additional information from netcdf object
            counties = rootgrp.variables['fips']
            lats = rootgrp.variables['lat']
            lons = rootgrp.variables['lon']
            if times is None:
                times = rootgrp.variables['time']
    else:
        # We just want a single variable (not a dictionary of them)
        # Retrieve the netcdf object (rootgrp) and add to weather dict
        if isinstance(weather_ncdf, str):
            # Open this up as a netCDF and read into array
            rootgrp = Dataset(weather_ncdf, 'r+', format='NETCDF4')
            weather = rootgrp.variables[var][:,:]
        elif isinstance(weather_ncdf, dict):
            # This is an {original, data, times} dictionary
            rootgrp = weather_ncdf['original']
            weather = weather_ncdf['data']
        else:
            # This is already a netcdf object
            rootgrp = weather_ncdf
            weather = rootgrp.variables[var][:,:]

        # Collect additional information from netcdf object
        counties = rootgrp.variables['fips']
        lats = rootgrp.variables['lat']
        lons = rootgrp.variables['lon']
        times = rootgrp.variables['time']

    # Loop through counties, calling make_generator with each
    for ii in range(len(counties)):
        fips = canonical_fips(counties[ii])
        print fips

        # Extract the weather just for this county
        if not isinstance(weather, dict):
            daily = weather[:,ii]
        else:
            daily = {}
            for variable in weather:
                daily[variable] = weather[variable][:,ii]

        # Call make_generator for this county
        generator = make_generator(fips, times, daily, lat=lats[ii], lon=lons[ii])
        if generator is None:
            continue

        # Call targetfunc with the result
        targetfunc(name, fips, generator)

    # Signal the end of the counties
    send_fips_complete(make_generator)

def make_tar_ncdf_profile(weather_ncdf, var, make_generator):
    """Like make_tar_ncdf, except that just goes through the motions,
    and only for 100 counties
    weather_ncdf: str for one, or {variable: filename} for calling
      generator with {variable: data}.
    var: str for one, or [str] for calling generator with {variable: data}
    """

    # Open a single netCDF if only one filename passed in
    if isinstance(weather_ncdf, str):
        # Collect the necessary info
        rootgrp = Dataset(weather_ncdf, 'r+', format='NETCDF4')
        counties = rootgrp.variables['fips']
        lats = rootgrp.variables['lat']
        lons = rootgrp.variables['lon']
        times = rootgrp.variables['time']
        weather = rootgrp.variables[var][:,:]
    else:
        # Open all netCDF referenced in var
        weather = {} # Construct a dictionary of [yyyyddd x county] arrays
        for variable in var:
            rootgrp = Dataset(weather_ncdf[variable], 'r+', format='NETCDF4')
            counties = rootgrp.variables['fips']
            lats = rootgrp.variables['lat']
            lons = rootgrp.variables['lon']
            times = rootgrp.variables['time']
            weather[variable] = rootgrp.variables[variable][:,:]

    # Just do 100 counties
    for ii in range(100):
        # Always using 5 digit fips
        fips = canonical_fips(counties[ii])
        print fips

        # Construct the input array for this county
        if not isinstance(weather, dict):
            daily = weather[:,ii]
        else:
            daily = {}
            for variable in weather:
                daily[variable] = weather[variable][:,ii]

        # Generate the generator
        generator = make_generator(fips, times, daily, lat=lats[ii], lon=lons[ii])
        if generator is None:
            continue

        # Just print out the results
        print "year", "fraction"

        for (year, effect) in generator:
            print year, effect

### Effect calculation functions

## make_generator functions

def load_tar_make_generator(targetdir, name, column=None):
    """Load existing data for additional calculations.
    targetdir: relative path to a directory of effect bundles.
    name: the effect name (so the effect bundle is at <targetdir>/<name>.tar.gz
    """

    # Extract the existing tar into a loader tempdir
    tempdir = enter_local_tempdir('loader-')
    os.system("tar -xzf " + os.path.join("..", targetdir, name + ".tar.gz"))
    exit_local_tempdir(tempdir, killit=False)

    def generate(fips, yyyyddd, temps, *args, **kw):
        # When all of the counties are done, kill the local dir
        if fips == FIPS_COMPLETE:
            print "Remove", tempdir
            # We might be in another tempdir-- check
            if os.path.exists(tempdir):
                kill_local_tempdir(tempdir)
            else:
                kill_local_tempdir(os.path.join('..', tempdir))
            return

        # Open up the effect for this bundle
        fipspath = os.path.join(tempdir, name, fips + ".csv")
        if not os.path.exists(fipspath):
            fipspath = os.path.join('..', fipspath)
            if not os.path.exists(fipspath):
                # If we can't find this, just return a single year with 0 effect
                print fipspath + " doesn't exist"
                yield (yyyyddd[0] / 1000, 0)
                raise StopIteration()

        with open(fipspath) as fp:
            reader = csv.reader(fp)
            reader.next() # ignore header

            # yield the same values that generated this effect file
            for row in reader:
                if column is None:
                    yield [int(row[0])] + map(float, row[1:])
                else:
                    yield (int(row[0]), float(row[column]))

    return generate

### Aggregation from counties to larger regions

def aggregate_tar(name, scale_dict=None, targetdir=None, collabel="fraction", get_region=None, report_all=False):
    """Aggregates results from counties to larger regions.
    name: the name of an impact, already constructed into an effect bundle
    scale_dict: a dictionary of weights, per county
    targetdir: directory holding both county bundle and to hold region bundle
    collabel: Label for result column(s)
    get_region: either None (uses first two digits of FIPS-- aggregates to state),
      True (combine all counties-- aggregate to national),
      or a function(fips) => code which aggregates each set of counties producing the same name
    report_all: if true, include a whole sequence of results; otherwise, just take first one
    """

    # Get a region name and a get_region function
    region_name = 'region' # final bundle will use this as a suffix

    if get_region is None: # aggregate to state
        get_region = lambda fips: fips[0:2]
        region_name = 'state'
    elif get_region is True: # aggregate to nation
        get_region = lambda fips: 'national'
        region_name = 'national'
    else:
        # get a title, if get_region returns one for dummy-fips "_title_"
        try:
            title = get_region('_title_')
            if title is not None:
                region_name = title
        except:
            pass

    regions = {} # {region code: {year: (numer, denom)}}
    # This is the effect bundle to aggregate
    target = get_target_path(targetdir, name)

    # Generate a temporary directory to extract county results
    tempdir = enter_local_tempdir()
    # Extract all of the results
    os.system("tar -xzf " + os.path.join("..", target) + ".tar.gz")

    # Go through all counties
    for filename in os.listdir(name):
        # If this is a county file
        match = re.match(r'(\d{5})\.csv', filename)
        if match:
            code = match.groups(1)[0] # get the FIPS code

            # Check that it's in the scale_dict
            if scale_dict is not None and code not in scale_dict:
                continue

            # Check which region it is in
            region = get_region(code)
            if region is None:
                continue

            # Prepare the dictionary of results for this region, if necessary
            if region not in regions:
                regions[region] = {} # year => (numer, denom)

            # Get out the current dictioanry of years
            years = regions[region]

            # Go through every year in this effect file
            with open(os.path.join(name, filename)) as csvfp:
                reader = csv.reader(csvfp, delimiter=',')
                reader.next()

                if report_all: # Report entire sequence of results
                    for row in reader:
                        # Get the numerator and denominator for this weighted sum
                        if row[0] not in years:
                            numer, denom = (np.array([0] * (len(row)-1)), 0)
                        else:
                            numer, denom = years[row[0]]

                        # Add on one more value to the weighted sum
                        try:
                            numer = numer + np.array(map(float, row[1:])) * (scale_dict[code] if scale_dict is not None else 1)
                            denom = denom + (scale_dict[code] if scale_dict is not None else 1)
                        except Exception, e:
                            print e

                        # Put the weighted sum calculation back in for this year
                        years[row[0]] = (numer, denom)
                else: # Just report the first result
                    for row in reader:
                        # Get the numerator and denominator for this weighted sum
                        if row[0] not in years:
                            numer, denom = (0, 0)
                        else:
                            numer, denom = years[row[0]]

                        # Add on one more value to the weighted sum
                        numer = numer + float(row[1]) * (scale_dict[code] if scale_dict is not None else 1)
                        denom = denom + (scale_dict[code] if scale_dict is not None else 1)

                        # Put the weighted sum calculation back in for this year
                        years[row[0]] = (numer, denom)

    # Remove all county results from extracted tar
    os.system("rm -r " + name)

    # Start producing directory of region results
    dirregion = name + '-' + region_name
    if not os.path.exists(dirregion):
        os.mkdir(dirregion)

    # For each region that got a result
    for region in regions:
        # Create a new CSV effect file
        with open(os.path.join(dirregion, region + '.csv'), 'wb') as csvfp:
            writer = csv.writer(csvfp, quoting=csv.QUOTE_MINIMAL)
            # Include a header row
            if not isinstance(collabel, list):
                writer.writerow(["year", collabel])
            else:
                writer.writerow(["year"] + collabel)

            # Construct a sorted list of years from the keys of this region's dictionary
            years = map(str, sorted(map(int, regions[region].keys())))

            # For each year, output the weighted average
            for year in years:
                if regions[region][year][1] == 0: # the denom is 0-- never got a value
                    writer.writerow([year, 'NA'])
                else:
                    # Write out the year's result
                    if report_all:
                        writer.writerow([year] + list(regions[region][year][0] / float(regions[region][year][1])))
                    else:
                        writer.writerow([year, float(regions[region][year][0]) / regions[region][year][1]])

    # Construct the effect bundle
    target = get_target_path(targetdir, dirregion)
    os.system("tar -czf " + os.path.join("..", target) + ".tar.gz " + dirregion)

    # Clean up temporary directory
    exit_local_tempdir(tempdir)
