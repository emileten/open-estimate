program dmas_extract_single

version 13.1
args apikey coeff infoid id

local dmas_urlstr = "http://dmas.berkeley.edu/api/extract_stata_single?apikey=`apikey'&coeff=`coeff'&infoid=`infoid'&id=`id'&ts=$S_TIME"

disp as txt "`dmas_urlstr'"

tempfile resfile
copy "`dmas_urlstr'" "`resfile'"
gen `result' = fileread("`resfile'")

display as txt "Response:"
display as txt `result'

end

