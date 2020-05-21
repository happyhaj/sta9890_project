rm(list=ls())
library(ggplot2)
library(dplyr)
library(glmnet)
library(ISLR)
library(randomForest)
library(gridExtra)


### code to sample the original larger dataset ###
### don't need to run this if you download dataset from here ###

##data= read.csv('financial-indicators/2014_Financial_Data.csv')
##data = data[sample(1:nrow(data),1000),]
##growths = c((221-34):221) #34 growths features
##ratios = c(147:156) #10 ratios features
##data = data[,c(ratios,growths,224)]
##write.csv(data,'financial.csv')






### only use 1000 datapoints and 45 features ###

data= read.csv('financial.csv')
data=data.matrix(data)

# The first column sample index of the original data.
# remove index column
data=data[,-1]

### assigning X and y
y        =    data[,46]
X        =    data.matrix(data[,-46])


### filling NAs with col means
for(i in 1:ncol(X)){
  X[is.na(X[,i]), i] = mean(X[,i], na.rm = TRUE)
  }

### standardizing ratio features
standardize =function(x){x/sqrt(mean((x-mean(x))^2))}
X = apply(X,2,standardize)

summary(X)

### since data is standarized, we can check the highest range, which is the highest std dev in original data
X.range = apply(X,2,range)[2,] - apply(X,2,range)[1,]
X.range = X.range[order(X.range,decreasing=T)]



n        =    dim(X)[1]
p        =    dim(X)[2]



###split datasets into testing set and training set.

n.train          =     floor(0.8*n)
n.test           =     n-n.train
M                =     100

Rsq.train.rid    =     rep(0,M)
Rsq.test.rid     =     rep(0,M)
Rsq.train.las    =     rep(0,M)
Rsq.test.las     =     rep(0,M) 
Rsq.train.el     =     rep(0,M)
Rsq.test.el      =     rep(0,M)  #el = elastic net
Rsq.train.rf     =     rep(0,M)
Rsq.test.rf      =     rep(0,M)  # rf= randomForest

time.rid         =     rep(0,M)
time.las         =     rep(0,M)
time.rf          =     rep(0,M)
time.el          =     rep(0,M)

re.train.rid     =     rep(0,p)
re.test.rid      =     rep(0,p)
re.train.las     =     rep(0,p)
re.test.las      =     rep(0,p)
re.train.el      =     rep(0,p)
re.test.el       =     rep(0,p)
re.train.rf      =     rep(0,p)
re.test.rf       =     rep(0,p)


############################################################
## Repeat each model 100 times and calculate running time ##
############################################################

esq_total = mean((y - mean(y))^2) 

rid = ''
las = ''
el = ''

for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  
  # fit ridge and calculate and record the train and test R squares 
  #ridge
  start.time=Sys.time()
  cv.fit.rid          =     cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
  fit.rid              =     glmnet(X.train, y.train, alpha = 0, lambda = cv.fit.rid$lambda.min)
  y.train.hat.rid      =     predict(fit.rid, X.train) # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat.rid      =     predict(fit.rid, X.test) # y.test.hat=X.test %*% fit$beta  + fit$a0

  Rsq.test.rid[m]   =     1-mean((y.test - y.test.hat.rid)^2) /esq_total 
  Rsq.train.rid[m]  =     1-mean((y.train - y.train.hat.rid)^2) /esq_total  
  end.time=Sys.time()
  time.rid[m]=end.time-start.time


  # fit lasso and calculate and record the train and test R squares 
  #lasso
  start.time=Sys.time()
  cv.fit.las           =     cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
  fit.las              =     glmnet(X.train, y.train, alpha = 1, lambda = cv.fit.las$lambda.min)
  y.train.hat.las      =     predict(fit.las, X.train) # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat.las       =     predict(fit.las, X.test) # y.test.hat=X.test %*% fit$beta  + fit$a0

  Rsq.test.las[m]   =     1-mean((y.test - y.test.hat.las)^2) /esq_total 
  Rsq.train.las[m]  =     1-mean((y.train - y.train.hat.las)^2) /esq_total     
  end.time=Sys.time()
  time.las[m]=end.time-start.time
  
  
  # fit elastic-net and calculate and record the train and test R squares 
  #el
  start.time=Sys.time()
  cv.fit.el         =     cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
  fit.el              =     glmnet(X.train, y.train, alpha = 0.5, lambda = cv.fit.el$lambda.min)
  y.train.hat.el      =     predict(fit.el, X.train) # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat.el       =     predict(fit.el, X.test) # y.test.hat=X.test %*% fit$beta  + fit$a0

  Rsq.test.el[m]   =     1-mean((y.test - y.test.hat.el)^2) /esq_total 
  Rsq.train.el[m]  =     1-mean((y.train - y.train.hat.el)^2) /esq_total  
  end.time=Sys.time()
  time.el[m]=end.time-start.time
  
  
  # fit RF and calculate and record the train and test R squares
  start.time=Sys.time()
  rf               =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.test.hat.rf    =     predict(rf, X.test)
  y.train.hat.rf   =     predict(rf, X.train)

  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat.rf)^2) /esq_total 
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat.rf)^2) /esq_total  
  end.time=Sys.time()
  time.rf[m]=end.time-start.time
  
  cat(sprintf("m=%3.f| Rsq.train.rid=%.2f, Rsq.train.las=%.2f, Rsq.train.el=%.2f, Rsq.train.rf=%.2f| \n       Rsq.test.rid=%.2f, Rsq.test.las=%.2f, Rsq.test.el=%.2f, Rsq.test.rf=%.2f| \n", 
              m, Rsq.train.rid[m], Rsq.train.las[m], Rsq.train.el[m], Rsq.train.rf[m], Rsq.test.rid[m], Rsq.test.las[m], 
               Rsq.test.el[m],Rsq.test.rf[m]))
  
  ## record the last sample for CV plots and Residual plots
  if(m == 100){
    rid = cv.fit.rid
    las = cv.fit.las
    el = cv.fit.el
    
    re.train.rid = y.train - y.train.hat.rid
    re.test.rid = y.test - y.test.hat.rid
    
    re.train.las = y.train - y.train.hat.las
    re.test.las = y.test - y.test.hat.las
    
    re.train.el = y.train - y.train.hat.el
    re.test.el = y.test - y.test.hat.el
    
    re.train.rf = y.train - y.train.hat.rf
    re.test.rf = y.test - y.test.hat.rf
    
  }
}

cat(sprintf('Ridge regression runing time:%4f \nLass running time:%4f \nElastic-net running time:%4f \nRandom forest running time:%4f \n', sum(time.rid),sum(time.las),sum(time.el),sum(time.rf)))


######################
## 10-fold CV curve ##
######################
par(mfrow=c(1,1))
plot(rid,main='ridge 10-fold CV')
plot(las,main='lasso 10-fold CV')
plot(el,main='elastic-net 10-fold CV')


###########################################################
## Boxplots of training rsq and test rsq of each model ##
###########################################################

#boxplot rid
par(mfrow=c(1,2))
boxplot(Rsq.train.rid, main='ridge train Rsq boxplot')
boxplot(Rsq.test.rid, main='ridge test Rsq boxplot')

#boxplot lasso
par(mfrow=c(1,2))
boxplot(Rsq.train.las, main='lasso train Rsq boxplot')
boxplot(Rsq.test.las, main='lasso test Rsq boxplot')

#boxplot el
par(mfrow=c(1,2))
boxplot(Rsq.train.el, main='elstisc-net train Rsq boxplot')
boxplot(Rsq.test.el, main='elstisc-net test Rsq boxplot')

#boxplot rf
par(mfrow=c(1,2))
boxplot(Rsq.train.rf, main='random forest train Rsq boxplot')
boxplot(Rsq.test.rf, main='random forest test Rsq boxplot')


###########################################################
## Boxplots of training rsq and test residuals of each model ##
###########################################################

#boxplot rid
par(mfrow=c(1,2))
boxplot(re.train.rid, main='ridge train residual boxplot')
boxplot(re.test.rid, main='ridge test residual boxplot')

#boxplot lasso
par(mfrow=c(1,2))
boxplot(re.train.las, main='lasso train residual boxplot')
boxplot(re.test.las, main='lasso test residual boxplot')

#boxplot el
par(mfrow=c(1,2))
boxplot(re.train.el, main='elstisc-net train residual boxplot')
boxplot(re.test.el, main='elstisc-net test residual boxplot')

#boxplot rf
par(mfrow=c(1,2))
boxplot(re.train.rf, main='random forest train residual boxplot')
boxplot(re.test.rf, main='random forest test residual boxplot')

par(mfrow=c(1,1))
############################
## Barplot with bootstrap ##
############################

bootstrapSamples =     100
beta.rf.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.el.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)         
beta.rid.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.lasso.bs    =     matrix(0, nrow = p, ncol = bootstrapSamples)         

        

for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # fit bs rid
  cv.rid           =     cv.glmnet(X.bs, y.bs, alpha = 0, nfolds = 10)
  rid.fit          =     glmnet(X.bs, y.bs, alpha = 0, lambda = cv.rid$lambda.min)
  beta.rid.bs[,m]  =     as.vector(rid.fit$beta)
  
  # fit bs lasso
  cv.lasso         =     cv.glmnet(X.bs, y.bs, alpha = 1, nfolds = 10)
  lasso.fit        =     glmnet(X.bs, y.bs, alpha = 1, lambda = cv.lasso$lambda.min)
  beta.lasso.bs[,m]=     as.vector(lasso.fit$beta)
  
  # fit bs el
  cv.el            =     cv.glmnet(X.bs, y.bs, alpha = 0.5, nfolds = 10)
  el.fit           =     glmnet(X.bs, y.bs, alpha = 0.5, lambda = cv.el$lambda.min)  
  beta.el.bs[,m]   =     as.vector(el.fit$beta)
  
  # fit bs rf
  rf               =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]   =     as.vector(rf$importance[,1])
  
  cat(sprintf("Bootstrap Sample %3.f \n", m))
  
}

# calculate bootstrapped standard errors / alternatively you could use qunatiles to find upper and lower bounds

rf.bs.sd    = apply(beta.rf.bs, 1, sd)
el.bs.sd    = apply(beta.el.bs, 1, sd)
rid.bs.sd   = apply(beta.rid.bs, 1, sd)
las.bs.sd   = apply(beta.lasso.bs, 1, sd)


# fit rf to the whole data
rf               =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)

# fit el to the whole data
cv.el            =     cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
el.fit           =     glmnet(X, y, alpha = a, lambda = cv.el$lambda.min)

# fit rid to the whole data
cv.rid           =     cv.glmnet(X, y, alpha = 0, nfolds = 10)
rid.fit          =     glmnet(X, y, alpha = 0, lambda = cv.rid$lambda.min)
# fit lasso to the whole data
cv.lasso         =     cv.glmnet(X, y, alpha = 1, nfolds = 10)
lasso.fit        =     glmnet(X, y, alpha = 1, lambda = cv.lasso$lambda.min)


betaS.rf               =     data.frame(names(X[1,]), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")

betaS.el               =     data.frame(names(X[1,]), as.vector(el.fit$beta), 2*el.bs.sd)
colnames(betaS.el)     =     c( "feature", "value", "err")

betaS.rid              =     data.frame(names(X[1,]), as.vector(rid.fit$beta), 2*rid.bs.sd)
colnames(betaS.rid)    =     c( "feature", "value", "err")

betaS.lasso            =     data.frame(names(X[1,]), as.vector(lasso.fit$beta), 2*las.bs.sd)
colnames(betaS.lasso)  =     c( "feature", "value", "err")



# need to change the order of factor levels by specifying the order explicitly.
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.el$feature     =  factor(betaS.el$feature, levels = betaS.el$feature[order(abs(betaS.el$value), decreasing = TRUE)])
betaS.rid$feature    =  factor(betaS.rid$feature, levels = betaS.rid$feature[order(abs(betaS.rid$value), decreasing = TRUE)])
betaS.lasso$feature  =  factor(betaS.lasso$feature, levels = betaS.lasso$feature[order(abs(betaS.lasso$value), decreasing = TRUE)])

ridPlot =  ggplot(betaS.rid[1:10,], aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle('Rid importance of variables')

lassoPlot =  ggplot(betaS.lasso[1:10,], aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle('Lasso importance of variables')

elPlot =  ggplot(betaS.el[1:10,], aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle('El importance of variables')

rfPlot =  ggplot(betaS.rf[1:10,], aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle('Rf importance of variables')

grid.arrange(ridPlot,lassoPlot,nrow = 2)

grid.arrange(rfPlot, elPlot,nrow = 2)



