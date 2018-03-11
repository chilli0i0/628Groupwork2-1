library(tidyverse)
library(ggthemes)
df = read_csv("train_data.csv")
year = format(df$date,'%Y')
star = df$stars

df1 = data.frame(star,year)
m = aggregate(df1$star, list(df1$year), mean)
m$Group.1 = as.numeric(m$Group.1)

CairoPNG("star_year.png", width = 1280, height = 720)
ggplot(m, aes(x=Group.1, y=x)) +
  geom_point(shape = 1) +    
  geom_smooth(method = lm,formula = y ~ x + I(x^2)) +
  scale_x_continuous(name = "Year") +
  scale_y_continuous(name = "Avg. Stars")+
  ggtitle("Avg. Stars v.s Year")+
  theme_economist() +
  theme(axis.line.x = element_line(size=.5, colour = "black"),
        axis.line.y = element_line(size=.5, colour = "black"),
        axis.text.x=element_text(colour="black", size = 18),
        axis.text.y=element_text(colour="black", size = 18),
        axis.title.x = element_text(color="black",face="bold",size=28),
        axis.title.y = element_text(color="black", face="bold",size=28),
        panel.grid.major = element_line(colour = "#d3d3d3"),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(), panel.background = element_blank(),
        plot.title = element_text(family = "Arial",hjust = 0.5,face="bold",size = 35),
        text=element_text(family="Arial"))
dev.off()
