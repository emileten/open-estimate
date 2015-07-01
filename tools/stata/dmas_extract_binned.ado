program dmas_extract_binned

version 13.1
args apikey endpoints coeffs infoid id

local dmas_urlstr = "extract_stata_binned?apikey=`apikey'&endpoints=`endpoints'&coeffs=`coeffs'&infoid=`infoid'&id=`id'&ts=$S_TIME"

dmas_get_api "`dmas_urlstr'", as_model(1)

end

