# Adding Stata models to DMAS

This document walks you through how to import a model from Stata to the Distributed Meta-Analysis System (DMAS).

## Basic Setup

1. Ensure that all of the information is correct in the Master DMAS Information spreadsheet.
  https://docs.google.com/spreadsheets/d/1lyvAeoUTji-FGH_Fz-hGWQOnEWJb2EMhWgc7LKz2ix0/edit

  *Check both the collection sheet for the collection where the new model will be added and the models sheet, where the publication information will be.*

2. Log into your account on DMAS, creating one if necessary.
  http://dmas.berkeley.edu/

  *Any model you create will be associated with your account, and will initially only be viewable by that account.*

3. Create an API key and record it.
  http://dmas.berkeley.edu/api/initialize

  *Your API key will be required for most DMAS operations outside of the website, and uniquely identifies you as the user.*

4. Download the DMAS import .ado files and place them in your Stata’s search path.
  https://github.com/jrising/open-estimate/blob/master/tools/stata/

  *See here for a description of where Stata looks for .ado files: http://www.stata.com/support/faqs/programming/search-path-for-ado-files/*
  *If you already have Stata running, you will need to execute a ‘discard’ command to refresh the function list.*
  *For me, I run in stata `adopath + /Users/jrising/projects/dmas/lib/tools/stata`*

## Adding a model

Now that you have everything set up to add models, the following walks you though adding one.  We will assume that you are sharing your estimate with the GCP community.

1. Identify the "Unique ID" on the Master DMAS Information spreadsheet for the estimate you will be adding, or add a new one.  Switch to the "Models" sheet to find and add unique IDs.
  https://docs.google.com/spreadsheets/d/1lyvAeoUTji-FGH_Fz-hGWQOnEWJb2EMhWgc7LKz2ix0/edit

  *This spreadsheet has all of the meta-information for the estimate, like publication and units.  If you don't have a specific item you want to add, choose any Unique ID.*

1. Run your regression.  If you don't already have a regression that you want to upload, try one of the tutorials below.

3. After you have run your regression, execute the following command:
   ```dmas_put_model [Your API Key] [The GCP Spreadsheet Key]```

  *This will call DMAS with all of the information in your regression, displaying both the URL used and the result, which will be a DMAS ID for your regression.*

  *Note: If the model includes fixed effects or other dummy variables, the upload time can be greatly improved by adding an option "coefficient count" to the `dmas_put_model` command, specifying that only that many coefficients should be uploaded to DMAS.*

4. Record the hexidecimal value it returns, as you will need this in step 6.  If you are sharing this with the GCP community, fill in the DMAS ID returned by dmas_put_model into the master spreadsheet.
  https://docs.google.com/spreadsheets/d/1lyvAeoUTji-FGH_Fz-hGWQOnEWJb2EMhWgc7LKz2ix0/edit

  *Once you have put in the DMAS ID for a given model, subsequent calls to dmas_put_model will replace the information in this object.  You may see the error “server refused to send file”, which just means that the ID returned is the same.*

5. Check to see that your result has shown up in your accounts list of estimates, at
  http://dmas.berkeley.edu/model/list_estimates

  *Estimates are subtly different from models, so you will only see your Stata estimates here.*

6. Execute one of the four "model extraction" commands, defined below.

  *This will actually create the model in DMAS.  You could extract multiple models from a single Stata regression.*

7. Use the link returned by the command to make sure that the model came through as expected.

## Model Extraction Commands and Tutorials

### Single Variable Models: `dmas_extract_single`

A single variable model describes a coefficient through its probability distribution (a Gaussian at the estimated value, with the standard error as its standard deviation).

**Syntax:**
```dmas_extract_single [Your API Key] [Coefficient Name] [GCP Spreadsheet Key] [DMAS Estimate ID]```

**Example:**
```
use http://www.stata-press.com/data/r13/auto
regress mpg weight displ if foreign

dmas_put_model 4sW2Txtsn8o3bkwY FUNKY-TOWN
dmas_extract_single 4sW2Txtsn8o3bkwY displacement FUNKY-TOWN 558329c30e703251ff48ecce
```

### Polynomial estimates: `dmas_extract_polynomial`

Polynomial estimates have some collection of coefficients for an N-order polynomial.

**Syntax:**
dmas_extract_polynomial [Your API Key] [Coefficients, Comma Delimited] [Lower Bound] [Upper Bound] [GCP Spreadsheet ID] [DMAS Estimate ID]

**Example:**
```
use http://www.stata-press.com/data/r13/auto
regress mpg c.weight##c.weight##c.weight displ if foreign

dmas_put_model 4sW2Txtsn8o3bkwY FUNKY-TOWN
dmas_extract_polynomial 4sW2Txtsn8o3bkwY _cons,weight,c.weight#c.weight,c.weight#c.weight#c.weight 1760 4840 FUNKY-TOWN 558de5f0b4a69c1f067b4fe5
```

### Binned Variable Models: `dmas_extract_binned`

A binned variable model describes a non-linear response, using data that falls into specific independent variable bins.

**Syntax:**
```dmas_extract_binned [Your API Key] [End Points, Comma Delimited] [Coefficients, Comma Delimited] [GCP Spreadsheet Key] [DMAS Estimate ID]```

Use the coefficient name `drop` for the bin that is dropped.

**Example:**
```
clear
set obs 1000
gen x = 10*runiform()
gen e = rnormal()
gen y = x^2 + e
gen bin1 = x <= 2
gen bin2 = x > 2 & x <= 4
gen bin3 = x > 4 & x <= 6
gen bin4 = x > 6 & x <= 8
gen bin5 = x > 8

* table is (x < 2, 2 < x < 4, 4 < x < 6, 6 < x < 8, x > 8, y)
mat table = (0, 0, 0, 0, 0, 0)
forvalues group = 1/100 {
  qui: egen groupx1 = total(bin1) if _n >= (`group' - 1) * 10 + 1 & _n <= `group'*10
  qui: egen groupx2 = total(bin2) if _n >= (`group' - 1) * 10 + 1 & _n <= `group'*10
  qui: egen groupx3 = total(bin3) if _n >= (`group' - 1) * 10 + 1 & _n <= `group'*10
  qui: egen groupx4 = total(bin4) if _n >= (`group' - 1) * 10 + 1 & _n <= `group'*10
  qui: egen groupx5 = total(bin5) if _n >= (`group' - 1) * 10 + 1 & _n <= `group'*10
  qui: egen groupy = total(y) if _n >= (`group' - 1) * 10 + 1 & _n <= `group'*10
  mat table = table \ (groupx1[`group'*10], groupx2[`group'*10], groupx3[`group'*10], groupx4[`group'*10], groupx5[`group'*10], groupy[`group'*10])
  drop groupx1 groupx2 groupx3 groupx4 groupx5 groupy
}

clear
set obs 101
svmat table, names(col)
drop if _n == 1

* Drop c3
reg c6 c1 c2 c4 c5

dmas_put_model 4sW2Txtsn8o3bkwY NA
dmas_extract_binned 4sW2Txtsn8o3bkwY -inf,.2,.4,.6,.8,inf c1,c2,drop,c4,c5 NA 5589b6edb4a69c941f9f4c86
```

### Predicted Curves: `dmas_extract_predict`

Predicted curves are arbitrary nonlinear curves, with estimated values and standard errors that change with some independent variable.

**Syntax:**
```dmas_extract_predict [Independent Variable] [Mean Estimate] [Bottom Confidence Interval] [Top Confidence Interval], apikey([Your API Key]) infoid([GCP Spreadsheet ID]) id(DMAS Estimate ID)```

**Example:**
```
set obs 1000
gen year = ceil(_n/100)
gen x=4*uniform()
gen z=2*uniform()
gen w=2*uniform()
gen e = 4*rnormal()
gen y= w + 3*z + 4*x - x^2 + .1*x^3 + e

* Add “save splinedata” after line 174 (after the predictnl command)
plot_rcspline y x, save_data_file(example.dta)
dmas_put_model 4sW2Txtsn8o3bkwY NA

use example, clear
line mean_est_nl indep_var

dmas_extract_predict indep_var mean_est_nl ci_bot_nl ci_top_nl, apikey(4sW2Txtsn8o3bkwY) infoid(NA) id(55931edb6253ea1a3315d9bb)
```

