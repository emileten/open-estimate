setwd("~/projects/dmas/simple")

source("tools_reg.R")
tbl <- read.csv("data.csv")

mod <- lm(satell ~ width + factor(color), data=tbl)

summary(mod)
get.coef.robust(mod)
get.coef.clust(mod, tbl$color)
