# Projection architecture

Projection calculations require multiple steps, which the Imperics
system describes as a structure of objects, each of which performs a
small part of these steps with a number of checks to ensure that
interfaces between these objects are properly defined.

The top-level structure for these calculations is a data structure of
`Calculation` objects. The Calculation objects include information on
all of the steps necessary to take climate and socioeconomic data and
estimate future impacts.

Conceptually, there are two kinds of Calculations. One kind contains
other Calculations and performs a function on their results. For
example, a `Sum` Calculation sums the result of other
Calculations. The other kind operates directly on weather
inputs. These typically contain a `CurveGenerator` object to represent
the function to be applied to the weather inputs.

The main function exposed by Calculation objects is `apply`, which
instatiates the Calculation for a given region, returning an
`Application` object. Application objects can have memory, which are
modified over the course of a projection. When Calculation objects
contain other Calculations as sub-calculations, this is reflected by a
corresponding structure in the Application objects.

The main function exposed by Application objects is `push`, which
takes a chunk of weather data and produces one or more years of
projected results. When an Application object contains other
Application or CurveGenerator objects, it will typically pass this
weather data to these, in order to recieve results that it can operate
on.
