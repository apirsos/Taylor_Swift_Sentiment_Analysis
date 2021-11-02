# anova comparison by albums

setwd("~/Desktop/NCSU/Fall 2/Text")
dataset = read.csv("shake_sentiment.csv")

ts_neg <- lm(neg ~ album, data=dataset)
anova(ts_neg)
summary(ts_neg)
ts_neg_aov <-  aov(neg ~ album, data=dataset)
tukey.ts_neg <- TukeyHSD(ts_neg_aov)
print(tukey.ts_neg)

ts_neu <- lm(neu ~ album, data=dataset)
anova(ts_neu)
summary(ts_neu)
ts_neu_aov <-  aov(neu ~ album, data=dataset)
tukey.ts_neu <- TukeyHSD(ts_neu_aov)
print(tukey.ts_neu$album[,c(4)])
print(tukey.ts_neu)
#reputation-1989
#Speak Now-1989
#Speak Now-Fearless 


ts_pos <- lm(pos ~ album, data=dataset)
anova(ts_pos)
summary(ts_pos)
ts_pos_aov <-  aov(pos ~ album, data=dataset)
tukey.ts_pos <- TukeyHSD(ts_pos_aov)
print(tukey.ts_pos)
#Speak Now-1989 
#Speak Now-Fearless