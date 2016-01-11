program dmas_extract_predict

version 13.1

syntax varlist(min=4 max=4 numeric ts), [apikey(string) infoid(string) id(string)]

preserve
* Put in dummy data, so we have one row
clear
set obs 1
gen OK = 3

if ("`id'" == "") {
    disp as txt "id not provided; using last result"
    local id = "$DMAS_LAST_RESULT"
}

loc asformat = "%7.4g"

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

dmas_get_api "make_queue", as_model(0)
dmas_get_api "queue_arguments?apikey=`apikey'&infoid=`infoid'&id=`id'&x=" + `indepStr' + "&mean=" + `meanStr', as_model(0)
dmas_get_api "call_with_queue?method=extract_stata_predict&level=5&cibot=" + `cibotStr' + "&citop=" + `citopStr', as_model(1)

restore

end
