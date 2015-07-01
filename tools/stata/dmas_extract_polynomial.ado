program dmas_extract_polynomial

version 13.1
args apikey coeffs lowbound highbound infoid id

local coeffs2 = subinstr("`coeffs'", "#", "%23", .)

local dmas_urlstr = "extract_stata_polynomial?apikey=`apikey'&coeffs=`coeffs2'&lowbound=`lowbound'&highbound=`highbound'&infoid=`infoid'&id=`id'&ts=$S_TIME"

dmas_get_api "`dmas_urlstr'", as_model(1)

end

