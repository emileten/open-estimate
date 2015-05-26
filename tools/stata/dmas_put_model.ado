program dmas_put_model

version 13.1
args infoid

use http://www.stata-press.com/data/r13/auto
regress mpg weight displ if foreign

//eret list

disp as txt "Uploading to DMAS..."

gen Xstr = "N=`e(N)'&df_m=`e(df_m)'&df_r=`e(df_r)'&F=`e(F)'&r2=`e(r2)'&rmse=`e(rmse)'&mss=`e(mss)'&rss=`e(rss)'&r2_a=`e(r2_a)'&ll=`e(ll)'&ll_0=`e(ll_0)'&rank=`e(rank)'&cmdline=`e(cmdline)'&title=`e(title)'&marginsok=`e(marginsok)'&vce=`e(vce)'&depvar=`e(depvar)'&cmd=`e(cmd)'&properties=`e(properties)'&predict=`e(predict)'&model=`e(model)'&estat_cmd=`e(estat_cmd)'"

gen Vstr = ""
matrix V = e(V)
forval ii = 1/`= rowsof(V)' {
  forval jj = 1/`= colsof(V)' {
    local Vij = V[`ii', `jj']
    replace Vstr = Vstr + "," + "`Vij'"
  }
}

gen Bstr = ""
matrix B = e(b)
forval ii = 1/`= colsof(B)' {
  local Bij = B[1, `ii']
  replace Bstr = Bstr + "," + "`Bij'"
}

disp Xstr + "&V=" + Vstr + "&b=" + Bstr

gen result = fileread("http://dmas.berkeley.edu/api/echo?data=Hello!")

display as txt "Response:"
display as txt result

end
