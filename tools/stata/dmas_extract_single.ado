program dmas_extract_single

version 13.1
args apikey coeff infoid id

local server = "http://127.0.0.1:8080/" /* "http://dmas.berkeley.edu/" */

local dmas_urlstr = "`server'api/extract_stata_single?apikey=`apikey'&coeff=`coeff'&infoid=`infoid'&id=`id'&ts=$S_TIME"

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

