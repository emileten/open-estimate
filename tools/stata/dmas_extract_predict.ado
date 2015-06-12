program dmas_extract_predict

version 13.1

syntax varlist(min=4 max=4 numeric ts), [apikey(string) infoid(string) id(string)]

loc indep_var = word("`varlist'", 1)
loc mean_est = word("`varlist'", 2)
loc ci_bot = word("`varlist'", 3)
loc ci_top = word("`varlist'", 4)

tempvar indep_varStr mean_estStr ci_botStr ci_topStr indepStr meanStr citopStr cibotStr

tostring `indep_var', gen(`indep_varStr') force
tostring `mean_est', gen(`mean_estStr') force
tostring `ci_bot', gen(`ci_botStr') force
tostring `ci_top', gen(`ci_topStr') force

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

local dmas_urlstr = "http://dmas.berkeley.edu/api/extract_stata_predict?apikey=`apikey'&infoid=`infoid'&id=`id'&ts=$S_TIME&level=95&x=" + `indepStr' + "&mean=" + `meanStr' + "&cibot=" + `cibotStr' + "&citop=" + `citopStr'

disp as txt "`dmas_urlstr'"

tempfile resfile
copy "`dmas_urlstr'" "`resfile'"
gen `result' = fileread("`resfile'")

display as txt "Response:"
display as txt `result'

end
