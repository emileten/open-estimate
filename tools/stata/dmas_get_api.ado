program dmas_get_api

version 13.1
syntax anything(name=methodargs), as_model(integer) quietly(integer)

* Change the server for all commands here
local server = "http://dmas.berkeley.edu"
* Local: "http://127.0.0.1:8080/"

local dmas_urlstr = "`server'/api/" + `methodargs'

if (!`quietly') {
    disp as txt "`dmas_urlstr'"
}

tempfile resfile
tempvar result
copy "`dmas_urlstr'" "`resfile'"
gen `result' = fileread("`resfile'")

if (!`quietly' | `result' != "OK") {
    display as txt "Response:"
    if (!`as_model' | substr(`result', 1, 6) == "ERROR:") {
        display as txt `result'
    }
    else {
        local final_urlstr = "`server'/model/view?id=" + `result'
        display as txt "`final_urlstr'"
    }
}

end
