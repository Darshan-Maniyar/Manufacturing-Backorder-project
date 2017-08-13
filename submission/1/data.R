library(dplyr)
library(neuralnet)
library(randomForest)


pt=read.csv("product_train.csv",stringsAsFactors = F,na.strings = c("Unknown",""," "))
str(pt)
glimpse(pt)
colSums(is.na(pt)) 
hist(pt$sales_1_month)
table(pt$sales_1_month)

pt$national_inv[pt$national_inv<0]=median(pt$national_inv[pt$national_inv>0])
pt$sku=NULL

pt$potential_issue=ifelse(pt$potential_issue=="Yes",1,0) 
pt$deck_risk=ifelse(pt$deck_risk=="Yes",1,0) 
pt$oe_constraint=ifelse(pt$oe_constraint=="Yes",1,0) 
pt$ppap_risk=ifelse(pt$ppap_risk=="Yes",1,0) 
pt$stop_auto_buy=ifelse(pt$stop_auto_buy=="Yes",1,0) 
pt$rev_stop=ifelse(pt$rev_stop=="Yes",1,0) 
pt$went_on_backorder=ifelse(pt$went_on_backorder=="Yes",1,0) 

table(pt$potential_issue)
length(unique(pt$in_transit_qty))

pt$lead_time[pt$lead_time==0] = median(pt$lead_time)

skl=function(x){
  y=(x-min(x))/(max(x)-min(x))
  return(y)
}

pt=apply(pt,2,skl)
pt=as.data.frame(pt)

a = which(names(pt) %in% "went_on_backorder")
# pt[-a]=scale(pt[-a])
pt[-a]=scale(pt[-a], center = FALSE, scale = apply(pt[-a], 2, sd, na.rm = TRUE))
glimpse(pt)

set.seed(100)
b=sample(1:nrow(pt),nrow(pt)*0.7)
pt_trn=pt[b,]
pt_tes=pt[-b,]

str(pt_trn1)

pt_trn1=pt_trn[-22]

table(pt_tes$went_on_backorder)
prin_comp <- prcomp(pt_trn1, scale. = T)
prin_comp$center
prin_comp$scale
prin_comp$rotation
prin_comp$rotation[1:5,1:4]
summary(prin_comp)
plot(prin_comp$rotation[,1],prin_comp$rotation[,2])
# biplot(prin_comp, scale = 0)
std_dev <- prin_comp$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")


train.data <- data.frame(went_on_backorder = as.factor(pt_trn$went_on_backorder), prin_comp$x)
str(train.data)
train.data=train.data[1:16]
table(pt_trn$went_on_backorder)

str(pt_trn)
pt_trn$went_on_backorder=as.factor(pt_trn$went_on_backorder)



set.seed(1)
fit=randomForest::randomForest(went_on_backorder ~. ,pt_trn,mtry=5, ntree=444,sampsize=c(1138,1138),strata=pt_trn$went_on_backorder,replace=T )#strata=Data$y, classwt = c(1E10,1E-10,1E10),sampsize=c(500,1000),strata=pt_trn$went_on_backorder,classwt = c(0.005,5000)

fit
summary(fit)
plot(fit)
varImpPlot(fit)

set.seed(1)
pre=predict(fit,pt_trn  )
pre
View()
# hist(pre)
# pre1=as.numeric(pre>0.5)
a=table(pt_trn$went_on_backorder,pre)
(a[2,2]/sum(a[2,]))-(a[1,2]/sum(a[1,]))


test.data <- predict(prin_comp, newdata = pt_tes)
test.data <- as.data.frame(test.data)
str(test.data)
test.data=test.data[1:15]

set.seed(1)
pre=predict(fit,pt_tes)
# hist(pre)
# pre1=as.numeric(pre>0.5)
a=table(pt_tes$went_on_backorder,pre)
(a[2,2]/sum(a[2,]))-(a[1,2]/sum(a[1,]))

a=pt_tes$went_on_backorder
library(pROC)
roccuv=roc(as.numeric(a) ,as.numeric(pre))
plot(roccuv)

##============Test==========================



ptest=read.csv("product_test.csv",stringsAsFactors = F,na.strings = c("Unknown",""," "))
str(ptest)
glimpse(ptest)
colSums(is.na(ptest)) 

pt1$national_inv[pt1$national_inv<0]=median(pt1$national_inv[pt1$national_inv>0])

ptest$sku=NULL

ptest$potential_issue=ifelse(ptest$potential_issue=="Yes",1,0) 
ptest$deck_risk=ifelse(ptest$deck_risk=="Yes",1,0) 
ptest$oe_constraint=ifelse(ptest$oe_constraint=="Yes",1,0) 
ptest$ppap_risk=ifelse(ptest$ppap_risk=="Yes",1,0) 
ptest$stop_auto_buy=ifelse(ptest$stop_auto_buy=="Yes",1,0) 
ptest$rev_stop=ifelse(ptest$rev_stop=="Yes",1,0) 


table(ptest$lead_time)
length(unique(ptest$in_transit_qty))

ptest$lead_time[ptest$lead_time==0] = median(ptest$lead_time)

ptest=apply(ptest,2,skl)
ptest=as.data.frame(ptest)

a = which(names(ptest) %in% "went_on_backorder")

# ptest=as.data.frame(scale(ptest))
ptest=scale(ptest, center = FALSE, scale = apply(ptest, 2, sd, na.rm = TRUE))
ptest=as.data.frame(ptest)
str(ptest)

# ptest.data=predict(prin_comp,ptest )



ptest.data <- as.data.frame(ptest.data)
str(ptest.data)
test.data=test.data[1:15]

set.seed(1)
pre=predict(fit,ptest)
hist(pre)
table(pre)





##=======Example================
str(iris)
table(iris$Species)
set.seed(2)
rf = randomForest(Species~., data = iris, classwt = c(1E-1,1E-1,1E5))
rf
plot(fit)

iris$pre=predict(rf,iris)
table(iris$Species,iris$pre)

