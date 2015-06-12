program dmas_extract_single

version 13.1
args apikey endpoints coeffs infoid id

local dmas_urlstr = "http://dmas.berkeley.edu/api/extract_stata_binned?apikey=`apikey'&endpoints=`endpoints'&coeffs=`coeffs'&infoid=`infoid'&id=`id'&ts=$S_TIME"

disp as txt "`dmas_urlstr'"

tempfile resfile
copy "`dmas_urlstr'" "`resfile'"
gen `result' = fileread("`resfile'")

display as txt "Response:"
display as txt `result'

end

