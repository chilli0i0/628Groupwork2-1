library(ggplot2)
name = c("P(4)","P(2)","P(5)","P(3)","P(1)","2016","letter number","punctuation number"
         ,"2010","2011","word number","2015","2012","unique word number",
         "2009","sentence number","stopword number","avg. word len.",
         "capital word number","unique word perc.","2013","2014","punctuation perc.",
         "lower words number")
score = c(267,258,256,247,220,33,32,31,15,15,15,14,12,11,10,10,10,9,8,8,6,5,4,3)
importance = data.frame(name,score)
importance$name = factor(importance$name,levels = importance$name[order(importance$score)])



ggplot(data=importance, aes(x=name, y=score))+
  geom_bar(stat="identity",width = 0.8,fill="steelblue")+ xlab("Features") +ylab("F Score")+ 
  coord_flip() +
  #geom_text(aes(label=score), position = position_stack(vjust = 1.1), size=3)
  ggtitle("Feature Importance")+theme(plot.title = element_text(hjust = 0.5))
  theme_gray()

ggsave("feature importance.png")              
       
