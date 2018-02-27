# Plot using R, plotting with Python is so hard...
setwd("~/OneDrive/OneDrive - UW-Madison/2018 Spring Wisc/STAT 628/628Groupwork2/Code/Lixia Yi")
library(ggplot2)
x = read.table("other_stars.txt", sep = "\t")
stars = x[,2]

qplot(data = stars, geom = 'histogram', binwidth = 1)
