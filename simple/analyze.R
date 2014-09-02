setwd("~/projects/dmas/simple")

tbl <- read.csv("data.csv")

mod <- lm(satell ~ width + factor(color), data=tbl)

source("tools_reg.R")

get.coef.clust(mod, tbl$color)
