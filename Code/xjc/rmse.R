df = data.frame(Method = c("NB","DT","K-NN","Logistic Regression","GBDT","Linear Regression","XGBoot","LSTM","NBSVM"),
                RMSE = c(1.36,1.21,1.20,0.96,0.85,0.79,0.78,0.76,0.67))
df2 = data.frame(Method = c("NBSVM","NBSVM + XGBoost","NBSVM + Linear Regression","NBSVM+XGBoost+EF","NBSVM+XGBoost+EF+PT"),
                 RMSE = c(0.68886,0.59237,0.61860,0.58263,0.58016))
library(ggplot2)
library(ggthemes)
library(Cairo)
df$Method = factor(df$Method,levels = unique(as.character(df$Method)))
df2$Method = factor(df2$Method,levels = unique(as.character(df2$Method)))

CairoPNG("offline mse.png", width = 1280, height = 720)
ggplot(data=df, aes(x=Method, y=RMSE,group=1)) +
  geom_line(size = 1.2,color="red")+
  geom_text(aes(label=RMSE),size = 6, hjust=-0.2, vjust=-0.2)+
  geom_point(size = 2)+
  ggtitle("Offline RMSE")+
  labs(x="Method",y="RMSE")+
  theme(
    plot.title = element_text(color="black", size=35,family="Arial",face="bold",hjust = 0.5),
    axis.title.x = element_text(color="black", family="Arial",face="bold",size=28),
    axis.title.y = element_text(color="black",family="Arial", face="bold",size=28),
    axis.text.x = element_text(color="black", size=18),
    axis.text.y = element_text(color="black", size=18),
    plot.background = element_rect(fill = "white"),
    panel.background = element_rect(fill = "slategray2", colour = "white",size = 0.5, linetype = "solid"),
    panel.grid.major.y = element_line(size = 0.5, linetype = 'solid',colour = "white"), 
    panel.grid.major.x = element_blank(),
    panel.grid.minor  = element_blank()
         )
dev.off()


CairoPNG("online mse.png", width = 1280, height = 720)
ggplot(data=df2, aes(x=Method, y=RMSE,group=1)) +
  geom_line(size = 1.2,color="red")+
  geom_text(aes(label=RMSE),size = 6, hjust=-0.2, vjust=-0.2)+
  geom_point(size = 2)+
  ggtitle("Online RMSE")+
  labs(x="Method",y="RMSE")+
  theme(
    plot.title = element_text(color="black", size=35,family="Arial",face="bold",hjust = 0.5),
    axis.title.x = element_text(color="black", family="Arial",face="bold",size=28),
    axis.title.y = element_text(color="black",family="Arial", face="bold",size=28),
    axis.text.x = element_text(color="black", size=18),
    axis.text.y = element_text(color="black", size=18),
    plot.background = element_rect(fill = "white"),
    panel.background = element_rect(fill = "slategray2", colour = "white",size = 0.5, linetype = "solid"),
    panel.grid.major.y = element_line(size = 0.5, linetype = 'solid',colour = "white"), 
    panel.grid.major.x = element_blank(),
    panel.grid.minor  = element_blank()
  )
dev.off()
