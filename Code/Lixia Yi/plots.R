# Plot using R, plotting with Python is so hard...
rm(list = ls())
setwd("~/OneDrive/OneDrive - UW-Madison/2018 Spring Wisc/STAT 628/628Groupwork2/Code/Lixia Yi")
library(ggplot2)
# x = read.table("other_stars.txt", sep = "\t")
# stars = x[,2]
# 
# qplot(data = stars, geom = 'histogram', binwidth = 1)

data = read.csv("/Users/yilixia/Downloads/lyi_small.csv")
data = data[,c('stars','text')]

colnames(data)
head(data)
data[2,]

others = read.table("/Users/yilixia/Downloads/lyi_other_lan.txt", sep = ",", header = F)
others = as.numeric(others)

chinese = read.table("/Users/yilixia/Downloads/lyi_chinese.txt", sep = ",", header = F)
chinese = as.numeric(chinese)

hist(data$stars, freq = T)
hist(data$stars[others-400000], freq = T)
hist(data$stars[chinese-400000], freq = T)

stars = NULL
other_stars = NULL
chinese_stars = NULL
for(i in 1:5){
  stars = c(stars, sum(data$stars[-c(others-400000, chinese-400000)] == i))
  other_stars = c(other_stars, sum(data$stars[others-400000] == i))
  chinese_stars = c(chinese_stars, sum(data$stars[chinese-400000] == i))
}
stars = stars/sum(stars)
other_stars = other_stars/sum(other_stars)
chinese_stars = chinese_stars/sum(chinese_stars)

chisq.test(stars, other_stars)
chisq.test(stars, chinese_stars)

