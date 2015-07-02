program dmas_put_model

version 13.1
args apikey infoid

disp as txt "Uploading to DMAS..."

tempvar Xstr Vstr Bstr result

local cmdline2 = subinstr("`e(cmdline)'", "#", "%32", .)

gen `Xstr' = "apikey=`apikey'&infoid=`infoid'&N=`e(N)'&df_m=`e(df_m)'&df_r=`e(df_r)'&F=`e(F)'&r2=`e(r2)'&rmse=`e(rmse)'&mss=`e(mss)'&rss=`e(rss)'&r2_a=`e(r2_a)'&ll=`e(ll)'&ll_0=`e(ll_0)'&rank=`e(rank)'&cmdline=`cmdline2'&title=`e(title)'&marginsok=`e(marginsok)'&vce=`e(vce)'&depvar=`e(depvar)'&cmd=`e(cmd)'&properties=`e(properties)'&predict=`e(predict)'&model=`e(model)'&estat_cmd=`e(estat_cmd)'"
replace `Xstr' = subinstr(`Xstr', " ", "+", .)

gen `Vstr' = ""
matrix V = e(V)
forval ii = 1/`= rowsof(V)' {
  forval jj = 1/`= colsof(V)' {
    local Vij = V[`ii', `jj']
    qui replace `Vstr' = `Vstr' + "," + "`Vij'"
  }
}

gen `Bstr' = ""
matrix B = e(b)
forval ii = 1/`= colsof(B)' {
  local Bij = B[1, `ii']
  qui replace `Bstr' = `Bstr' + "," + "`Bij'"
}

mat b = e(b)
local names: colnames b
local names2 = subinstr(subinstr("`names'", " ", ",", .), "#", "%23", .)

local dmas_urlstr = "put_stata_estimate?" + `Xstr' + "&V=" + `Vstr' + "&b=" + `Bstr' + "&names=" + "`names2'"

dmas_get_api "`dmas_urlstr'", as_model(0)

end
