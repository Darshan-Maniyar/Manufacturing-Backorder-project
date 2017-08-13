library(dplyr)
library(neuralnet)
library(randomForest)
library(xgboost)
library( nnet)

pt=read.csv("product_train.csv",stringsAsFactors = F,na.strings = c("Unknown",""," "))
str(pt)
glimpse(pt)
colSums(is.na(pt)) 
pt$national_inv[pt$national_inv<0]=median(pt$national_inv[pt$national_inv>0])

pt$sku=NULL

pt$potential_issue=ifelse(pt$potential_issue=="Yes",1,0) 
pt$deck_risk=ifelse(pt$deck_risk=="Yes",1,0) 
pt$oe_constraint=ifelse(pt$oe_constraint=="Yes",1,0) 
pt$ppap_risk=ifelse(pt$ppap_risk=="Yes",1,0) 
pt$stop_auto_buy=ifelse(pt$stop_auto_buy=="Yes",1,0) 
pt$rev_stop=ifelse(pt$rev_stop=="Yes",1,0) 
pt$went_on_backorder=ifelse(pt$went_on_backorder=="Yes",1,0) 

table(pt$lead_time)
length(unique(pt$in_transit_qty))

pt$lead_time[pt$lead_time==0] = median(pt$lead_time)

a = which(names(pt) %in% "went_on_backorder")
pt[-a]=scale(pt[-a])
glimpse(pt)

set.seed(100)
b=sample(1:nrow(pt),nrow(pt)*0.7)
pt_trn=pt[b,]
pt_tes=pt[-b,]



set.seed(1)
nn <- nnet(went_on_backorder~.,data=pt_trn,size = 10,decay=0.1, maxit=1000)#,subset = sampidx
nn$wts
summary(nn)
plot(nn)

bt=pt_trn
bt=pt_tes#[1:500,]

# View(b_test[2:5,])
set.seed(1)
bt$pre=predict(nn,bt)#raw,type = "raw"
hist(bt$pre)

library(pROC)
roccuv=roc(bt$went_on_backorder ,bt$pre)
plot(roccuv)

bt$pre=as.numeric(bt$pre>0.5)

a=table(bt$went_on_backorder ,bt$pre)
a
(a[2,2]/sum(a[2,]))-(a[1,2]/sum(a[1,]))
(a[2,2]/sum(a[2,]))-(a[2,2]/sum(a[,2]))
(a[2,2]+(a[1,1]))/(sum(a))



set.seed(2)
btr$pred=predict(bst,xgbdtest,ntreelimit = 1784)#,ntreelimit = 40
hist(btr$pred)
btr$pred=as.numeric(btr$pred>0.5)
a=table(btr$went_on_backorder,btr$pred)
a
(a[2,2]/sum(a[2,]))-(a[1,2]/sum(a[1,]))
(a[2,2]/sum(a[2,]))-(a[2,2]/sum(a[,2]))
(a[2,2]+(a[1,1]))/(sum(a))

confmatrix(btr$went_on_backorder,btr$pred,model="KS")

##=========Test============================

ptest=read.csv("product_test.csv",stringsAsFactors = F,na.strings = c("Unknown",""," "))
str(ptest)
glimpse(ptest)
colSums(is.na(ptest)) 

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

a = which(names(ptest) %in% "went_on_backorder")
ptest=as.data.frame(scale(ptest)) 
str(ptest)

b3=as.matrix(ptest)

xgbdtest_t=xgb.DMatrix(data=b3)

pre= predict(bst,xgbdtest_t,ntreelimit = 1784)
hist(pre)
pre=as.numeric(pre>0.5)
table(pre)
a=data.frame(zero=c(62286,408),one=c())
