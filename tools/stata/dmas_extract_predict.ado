program dmas_extract_predict

version 13.1

syntax varlist(min=4 max=4 numeric ts), [apikey(string) infoid(string) id(string)]

local server = "http://127.0.0.1:8080/" /* "http://dmas.berkeley.edu/" */

loc asformat = "%7.3g"

loc indep_var = word("`varlist'", 1)
loc mean_est = word("`varlist'", 2)
loc ci_bot = word("`varlist'", 3)
loc ci_top = word("`varlist'", 4)

tempvar indep_varStr mean_estStr ci_botStr ci_topStr indepStr meanStr citopStr cibotStr

tostring `indep_var', gen(`indep_varStr') force format(`asformat')
tostring `mean_est', gen(`mean_estStr') force format(`asformat')
tostring `ci_bot', gen(`ci_botStr') force format(`asformat')
tostring `ci_top', gen(`ci_topStr') force format(`asformat')

local N = _N

gen `indepStr' = ""
forvalues ii = 1/`N' {
    qui replace `indepStr' = `indepStr' + "," + `indep_varStr'[`ii']
}

gen `meanStr' = ""
forvalues ii = 1/`N' {
    qui replace `meanStr' = `meanStr' + "," + `mean_estStr'[`ii']
}

gen `cibotStr' = ""
forvalues ii = 1/`N' {
    qui replace `cibotStr' = `cibotStr' + "," + `ci_botStr'[`ii']
}

gen `citopStr' = ""
forvalues ii = 1/`N' {
    qui replace `citopStr' = `citopStr' + "," + `ci_topStr'[`ii']
}

local dmas_urlstr = "`server'api/extract_stata_predict?apikey=`apikey'&infoid=`infoid'&id=`id'&ts=$S_TIME&level=5&x=" + `indepStr' + "&mean=" + `meanStr' + "&cibot=" + `cibotStr' + "&citop=" + `citopStr'

disp as txt "`dmas_urlstr'"

tempfile resfile
tempvar result
copy "`dmas_urlstr'" "`resfile'"
gen `result' = fileread("`resfile'")

display as txt "Response:"
if (substr(`result', 1, 6) == "ERROR:") {
    display as txt `result'
}
else {
    local final_urlstr = "`server'model/view?id=" + `result'
    display as txt "`final_urlstr'"
}

end
