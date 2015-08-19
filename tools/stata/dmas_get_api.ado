program dmas_get_api

version 13.1
syntax anything(name=methodargs), as_model(integer) [quietly(integer 1)]

* Change the server for all commands here
local server = "http://dmas.berkeley.edu"
* Local: "http://127.0.0.1:8080/"

local dmas_urlstr = "`server'/api/" + `methodargs'

if (!`quietly') {
    disp as txt "`dmas_urlstr'"
}

tempfile resfile
tempvar result

* Repeat the request until we get a non-network error
local rc = 2
while inlist(`rc', 2, 631, 677) {
    capture copy "`dmas_urlstr'" "`resfile'"
    local rc = _rc
    if (`rc' != 0) {
        disp "Connection error.  Waiting and trying again..."
        sleep 1000
    }
}
if (`rc' != 0) {
    error `rc'
}

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
