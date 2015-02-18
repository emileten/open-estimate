cd ~/groups/aggregator/trunk/lib/simple

import delim data.csv

xi: reg satell width i.color
xi: reg satell width i.color, vce(hc2)
xi: reg satell width i.color, vce(cluster color)
