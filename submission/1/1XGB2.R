library(dplyr)
library(neuralnet)
library(randomForest)
library(xgboost)

pt=read.csv("product_train.csv",stringsAsFactors = F,na.strings = c("Unknown",""," "))
str(pt)
glimpse(pt)
colSums(is.na(pt)) 
sum(pt1$local_bo_qty<0)
hist(pt1$local_bo_qty)
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

skl=function(x){
  y=(x-min(x))/(max(x)-min(x))
  return(y)
}

pt=apply(pt,2,skl)
pt=as.data.frame(pt)


a = which(names(pt) %in% "went_on_backorder")
pt[-a]=scale(pt[-a])
glimpse(pt)

set.seed(100)
b=sample(1:nrow(pt),nrow(pt)*0.7)
pt_trn=pt[b,]
pt_tes=pt[-b,]
# 
# str(pt_trn1)
# 
# pt_trn1=pt_trn[-22]
# 
# table(pt_tes$went_on_backorder)
# prin_comp <- prcomp(pt_trn1, scale. = T)
# prin_comp$center
# prin_comp$scale
# prin_comp$rotation
# prin_comp$rotation[1:5,1:4]
# summary(prin_comp)
# plot(prin_comp$rotation[,1],prin_comp$rotation[,2])
# # biplot(prin_comp, scale = 0)
# std_dev <- prin_comp$sdev
# pr_var <- std_dev^2
# prop_varex <- pr_var/sum(pr_var)
# plot(prop_varex, xlab = "Principal Component",
#      ylab = "Proportion of Variance Explained",
#      type = "b")
# plot(cumsum(prop_varex), xlab = "Principal Component",
#      ylab = "Cumulative Proportion of Variance Explained",
#      type = "b")
# 
# 
# train.data <- data.frame(went_on_backorder = (pt_trn$went_on_backorder), prin_comp$x)
# str(train.data)
# train.data=train.data[1:16]
# 
# str(pt_tes)
# pt_tes1=pt_tes[-22]
# 
# pt_tes1 =predict(prin_comp,pt_tes1)
# test.data=as.data.frame(cbind(went_on_backorder = (pt_tes$went_on_backorder),pt_tes1)) 
# str(test.data)
# test.data=test.data[1:16]

# a=which(names(test.data) %in% "went_on_backorder")
# b1=as.matrix(train.data[a])
# b2=as.matrix(train.data[-a])
# 
# b3=as.matrix(test.data[a])
# b4=as.matrix(test.data[-a])

a=which(names(pt_trn) %in% "went_on_backorder")
b1=as.matrix(pt_trn[a])
b2=as.matrix(pt_trn[-a])

b3=as.matrix(pt_tes[a])
b4=as.matrix(pt_tes[-a])

xgbd=xgb.DMatrix(data=b2,label=b1)
xgbdtest=xgb.DMatrix(data=b4,label=b3)



param <- list("eta" =0.1,
              
              "max.depth"=4,
              "min_child_weight"=33,
              "gamma"=0,
              # "lambda"=10,"lambda_bias"=0,"alpha"=2,
              "nthread"=32,
              # "subsample"=0.8,
              # "colsample_bytree"=0.9,
              "scale_pos_weight"=150,
              "eval_metric" = KS,
              
              "objective"="binary:logistic"
              # "booster"="dart",
              # "sample_type"="weighted",
              # "normalize_type"="forest"
) 

set.seed(2)
bst.cv = xgb.cv(param=param,  xgbd ,nrounds=10,early_stopping_rounds=50,nfold = 10,seed=2,maximize =T)#,maximize =T,"maximize"=F,niter=10,  ,niter=500, nrounds = 50,colsample_bytree =0.9,gamma =100,min_child_weight=10,subsample =0.9)#)#prediction=TRUE, verbose=FALSE,
bst.cv$evaluation_log

plot(bst.cv$evaluation_log[,iter],bst.cv$evaluation_log[,test_KS_mean])



set.seed(2)
bst <- xgboost(param=param,xgbd,nrounds=10 ,nfold = 10,seed=2)#,niter=299, early_stopping_rounds=50,min_child_weight=10,subsample =0.9)#
bst$params

names <- dimnames(b2)[[2]]
importance_matrix <- xgb.importance(names, model = bst)
xgb.plot.importance(importance_matrix[1:25])



btr=pt_trn

set.seed(2)
btr$pred=predict(bst,xgbd,ntreelimit = 60)#
hist(btr$pred)
btr$pred=as.numeric(btr$pred>0.5)
a=table(btr$went_on_backorder,btr$pred)
a
(a[2,2]/sum(a[2,]))-(a[1,2]/sum(a[1,]))
(a[2,2]/sum(a[2,]))-(a[2,2]/sum(a[,2]))
(a[2,2]+(a[1,1]))/(sum(a))

confmatrix(btr$went_on_backorder,btr$pred,model="KS")
mean(LogLossBinary(btr$went_on_backorder,btr$pred))



btr=pt_tes

set.seed(2)
btr$pred=predict(bst,xgbdtest,ntreelimit = 10)#,ntreelimit = 40
hist(btr$pred)
btr$pred=as.numeric(btr$pred>0.5)
a=table(btr$went_on_backorder,btr$pred)

a
(a[2,2]/sum(a[2,]))-(a[1,2]/sum(a[1,]))
(a[2,2]/sum(a[2,]))-(a[2,2]/sum(a[,2]))
(a[2,2]+(a[1,1]))/(sum(a))

confmatrix(btr$went_on_backorder,btr$pred,model="KS")
mean(LogLossBinary(btr$went_on_backorder,btr$pred))

LogLossBinary = function(actual, predicted, eps = 1e-15) {
predicted = pmin(pmax(predicted, eps), 1-eps) - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
return(predicted)
}

##=========Test============================

ptest=read.csv("product_test.csv",stringsAsFactors = F,na.strings = c("Unknown",""," "))
str(ptest)
glimpse(ptest)
colSums(is.na(ptest)) 

ptest$national_inv[ptest$national_inv<0]=17  #median(ptest$national_inv[ptest$national_inv>0])
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


# a = which(names(ptest) %in% "went_on_backorder")
# ptest=as.data.frame(scale(ptest)) 
# str(ptest)

b3=as.matrix(ptest)

xgbdtest_t=xgb.DMatrix(data=b3)

set.seed(2)
pre= predict(bst,xgbdtest_t,ntreelimit = 10)
hist(pre)
pre=as.numeric(pre>0.5)
table(pre)
 # a=data.frame(zero=c(62286,408),one=c())

write.csv(x = pre,file = "xgb.csv",row.names = F)


