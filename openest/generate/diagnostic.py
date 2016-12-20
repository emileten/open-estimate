# -*- coding: utf-8 -*-
"""Diagnostic Manager for calculation results

The DiagnosticManager class manages calculation data for a diagnostic
file.  It is based around the following principles:

- Each DiagnosticManager manages one file of diagnostics.

- Each row in the diagnostic file is specific to a region and a year.

- As soon as all of the material for a given region-year combination
  is available, it should be written to the file and removed from
  memory.

- The number of columns for the file are not known until a region-year
  is complete, since diagnostics may come from any buried level of the
  calculation.

- There is typically only one calculation system in action at a time,
  and all results from these should be written to the same file.

A typical workflow is as follows:

```
import diagnostic

diagnostic.begin("my-calculation.csv")

# Set up the applications of the calculation
calculation = ...
applications = {region: calculation.apply(region) for region in regions}

for year in weather_years():
    for region in regions:
        for yearresult in applications[region].push(weather_data(region, year))
            # This yearresult may not correspond to the weather data year
            diagnostic.record(region, yearresult[0], "final-result", yearresult[1])

            # Do something with the yearresult
            ...

            diagnostic.finish(region, yearresult[0])

diagnostic.close()
```
"""

import os, csv

"""
The DiagnosticManager used by `begin`, `close`, and `record` below.
"""
default_manager = None

def begin(filepath):
    """Open up a new default DiagnosticManager, if none is currently in
    use.
    """
    global default_manager

    if default_manager is not None:
        raise RuntimeError("Default manager already in use.  Please run close() first.")

    default_manager = DiagnosticManager(filepath)

def close():
    """Close the existing default DiagnosticManager, writing out any
    available data.
    """
    global default_manager
    if default_manager is None:
        raise RuntimeError("No default manager in use.  Please run begin(filepath) first.")

    default_manager.close()
    default_manager = None

def record(region, year, column, value):
    """Record a value for a given region and year.  Will be written out
    when finish(region, year) is called.
    """
    if default_manager is None:
        return # Silently drop the record

    default_manager.record(region, year, column, value)

def is_recording():
    return default_manager is not None

def finish(region, year):
    """Write out the information for the given region, year combination.
    """
    if default_manager is None:
        raise RuntimeError("No default manager in use.  Please run begin(filepath) first.")

    default_manager.finish(region, year)

class DiagnosticManager(object):
    """Manager of the data for one diagnostic file.

    Properties:
        filepath: The path to the diagnostic file.
        initialized: Has the header been written?
        header: The list of columns, corresponding to the values for each region-year.
        incomplete: The data not yet written to the file.
    """

    def __init__(self, filepath):
        """Create a new diagnostic manager.  Does not write any data yet."""

        self.filepath = filepath
        self.initialized = False
        self.header = []
        #self.header_info = [] # by header column
        self.incomplete = {} # {(region, year) : [columns]}}

    def __del__(self):
        self.close()

    def close(self):
        """Write out any incomplete data and clean up."""

        # Write any region-years out to file
        if self.incomplete:
            with self._open() as writer:
                for region, year in incomplete:
                    self._writerow(writer, region, year, delete=False)

            self.incomplete = {}

    def record(self, region, year, column, value):
        """Record a single value for a region-year combination."""
        if column not in self.header:
            self.header.append(column)

        index = self.header.index(column)

        if (region, year) not in self.incomplete:
            columns = []
            self.incomplete[(region, year)] = columns
        else:
            columns = self.incomplete[(region, year)]

        if index < len(columns):
            columns[index] = value
        else:
            for ii in range(len(columns), index): # one fewer than index
                columns.append("NA")
            columns.append(value)

    def finish(self, region, year):
        """Finish a given region-year combination, and write out its data."""

        with self._open() as writer:
            self._writerow(writer, region, year)

    # Internal operations

    def _open(self):
        """Return a writer for the file, writting the header if needed."""

        if not initialized:
            fp = open(filepath, 'w')
            writer = CSVWriterFile(fp)
            writer.writerow(['region', 'year'] + self.header)
            self.initialized = True
        else:
            fp = open(filepath, 'a')
            writer = CSVWriterFile(fp)

        return writer

    def _writerow(self, writer, region, year, delete=True):
        """Write a given row, optionally deleting the corresponding data."""

        writer.writerow([region, year] + self.incomplete[(region, year)])
        if delete:
            del self.incomplete[(region, year)]

class CSVWriterFile(object):
    """A wrapper on csv.writer.

    Like csv.writer, but allows the use of the with statement.
    Tough to do as subclass of csv.DictWriter, because csv.writer is in C.
    """
    def __init__(self, csvfile, *args, **kwds):
        self.csvfile = csvfile
        self.writer = csv.writer(csvfile, *args, **kwds)

    def __enter__(self):
        self.csvfile.__enter__()

    def __exit__(self):
        self.csvfile.__exit__()

    def writerow(self, row):
        self.writer.writerow(row)
