program dmas_put_model

version 13.1
args apikey infoid varnum

disp as txt "Uploading to DMAS..."

set more off
preserve
* Put in dummy data, so we have one row
clear
set obs 1
gen OK = 3

/* Improvements:
 * Drop all fixed effects: anything that shows up in cmdline as i.X
 * Don't use queue if don't need to (and then make that one line non-quietly
 */

matrix B = e(b)
if ("`varnum'" == "") {
    local varnum = `= colsof(B)'
}
if (`varnum' > `= colsof(B)') {
    local varnum = `= colsof(B)'
}

dmas_get_api "make_queue", as_model(0) quietly(1)
local progressV = 100 * 4/7
local progressB = 100 * 1/7
local progressNames = 100 * 1/7
local progressOther = 100 * 1/7

tempvar Xstr Vstr Bstr result

local cmdline2 = subinstr("`e(cmdline)'", "#", "%32", .)

gen `Xstr' = "apikey=`apikey'&infoid=`infoid'&N=`e(N)'&df_m=`e(df_m)'&df_r=`e(df_r)'&F=`e(F)'&r2=`e(r2)'&rmse=`e(rmse)'&mss=`e(mss)'&rss=`e(rss)'&r2_a=`e(r2_a)'&ll=`e(ll)'&ll_0=`e(ll_0)'&rank=`e(rank)'&cmdline=`cmdline2'&title=`e(title)'&marginsok=`e(marginsok)'&vce=`e(vce)'&depvar=`e(depvar)'&cmd=`e(cmd)'&properties=`e(properties)'&predict=`e(predict)'&model=`e(model)'&estat_cmd=`e(estat_cmd)'"
* Cluster and xtreg-specific results
qui replace `Xstr' = `Xstr' + "N_clust=`e(N_clust)'&tss=`e(tss)'&df_a=`e(df_a)'&df_b=`e(df_b)'&r2_w=`e(r2_w)'&Tbar=`e(Tbar)'&Tcon=`e(Tcon)'&rho=`e(rho)'&sigma=`e(sigma)'&sigma_e=`e(sigma_e)'&sigma_u=`e(sigma_u)'&r2_b=`e(r2_b)'&r2_o=`e(r2_o)'&corr=`e(corr)'&N_g=`e(N_g)'&g_min=`e(g_min)'&g_max=`e(g_max)'&g_avg=`e(g_avg)'&marginsnotok=`e(marginsnotok)'&ivar=`e(ivar)'&clustvar=`e(clustvar)'&vcetype=`e(vcetype)'"
replace `Xstr' = subinstr(`Xstr', " ", "+", .)

disp as txt "Constructing V matrices..."

gen `Vstr' = ""
matrix V = e(V)
local totalV = `varnum' * `varnum'
forval ii = 1/`varnum' {
    forval jj = 1/`varnum' {
        if (strlen(`Vstr') > 800) { // send to server if too much
            dmas_get_api "queue_arguments?V=" + `Vstr', as_model(0) quietly(1)
            qui replace `Vstr' = ""
            disp "Progress:", `progressV' * (`ii' * `varnum' + `jj') / `totalV', "%"
        }

        local Vij = V[`ii', `jj']
        qui replace `Vstr' = `Vstr' + "," + "`Vij'"
    }
}

disp as txt "Constructing beta vector..."

gen `Bstr' = ""
forval ii = 1/`varnum' {
    if (strlen(`Bstr') > 800) { // send to server if too much
        dmas_get_api "queue_arguments?b=" + `Bstr', as_model(0) quietly(1)
        qui replace `Bstr' = ""
        disp "Progress:", `progressV' + `progressB' * `ii' / `varnum', "%"
    }
    local Bij = B[1, `ii']
    qui replace `Bstr' = `Bstr' + "," + "`Bij'"
}

disp as txt "Constructing coefficient names..."

local names: colnames B
* Remove all but varnum words
if (`varnum' < wordcount("`names'")) {
    local names = subinstr("`names'", " ", ",", `varnum' - 1)
    local names = substr("`names'", 1, strpos("`names'", " ") - 1)
}
local names2 = subinstr(subinstr("`names'", " ", ",", .), "#", "%23", .)

while (strlen("`names2'") > 800) {
    local sendnames = substr("`names2'", 1, 800)
    dmas_get_api "queue_arguments?names=`sendnames'", as_model(0) quietly(1)
    local names2 = substr("`names2'", 801, strlen("`names2'"))
    disp "Progress:", `progressV' + `progressB' + `progressNames' * 800 / max(800, strlen("`names2'"))
}

if (strlen(`Xstr') + strlen(`Vstr') + strlen(`Bstr') + strlen("`names2'") > 800) {
    if (strlen(`Xstr') > 250) {
        dmas_get_api "queue_arguments?" + `Xstr', as_model(0) quietly(1)
        qui replace `Xstr' = ""
    }
    if (strlen(`Vstr') > 250) {
        dmas_get_api "queue_arguments?V=" + `Vstr', as_model(0) quietly(1)
        qui replace `Vstr' = ""
    }
    if (strlen(`Bstr') > 250) {
        dmas_get_api "queue_arguments?b=" + `Bstr', as_model(0) quietly(1)
        qui replace `Bstr' = ""
    }
    if (strlen("`names2'") > 250) {
        dmas_get_api "queue_arguments?names=`names2'", as_model(0) quietly(1)
        local names2 = ""
    }
}

disp "Completing model..."

dmas_get_api "call_with_queue?method=put_stata_estimate&" + `Xstr' + "&V=" + `Vstr' + "&b=" + `Bstr' + "&names=" + "`names2'", as_model(0) quietly(1)

restore

end
