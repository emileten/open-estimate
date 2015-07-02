program dmas_get_api

version 13.1
syntax anything(name=methodargs), as_model(integer)

local server = "http://dmas.berkeley.edu"

local dmas_urlstr = "`server'/api/" + `methodargs'

disp as txt "`dmas_urlstr'"

tempfile resfile
tempvar result
copy "`dmas_urlstr'" "`resfile'"
gen `result' = fileread("`resfile'")

display as txt "Response:"
if (!`as_model' | substr(`result', 1, 6) == "ERROR:") {
    display as txt `result'
}
else {
    local final_urlstr = "`server'/model/view?id=" + `result'
    display as txt "`final_urlstr'"
}

end
