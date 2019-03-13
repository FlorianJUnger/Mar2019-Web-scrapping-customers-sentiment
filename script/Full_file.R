# Main file
# Florian UNGER
# 07/03/2019
# Amazon Web Scraping

### Preparing Training sets (Small Matrix for Iphone and Galaxy)

## Packages 

pacman::p_load(plot3Drgl, rgl, car, ggplot2,
               plotly, rstudioapi, corrplot, 
               rgl, manipulateWidget, reshape, 
               reshape2, Rfast, randomForest, 
               esquisse, doParallel, kknn, C50,
               e1071)

## Github

current_path = rstudioapi::getActiveDocumentContext()$path # save working directory
setwd(dirname(current_path))
setwd("..")

## Data files 

galaxy_matr <- read.csv(
  "C:/Users/Dell/Desktop/Ubiqum Data Analytics/AWS Web Scrapping/Web-scrapping-customers-sentiment/datasets/galaxy_smallmatrix_labeled_8d.csv",
  header = TRUE)

iphone_matr <- read.csv(
  "C:/Users/Dell/Desktop/Ubiqum Data Analytics/AWS Web Scrapping/Web-scrapping-customers-sentiment/datasets/iphone_smallmatrix_labeled_8d.csv",
  header = TRUE)

large_matrix <- read.csv(
  "C:/Users/Dell/Desktop/Ubiqum Data Analytics/AWS Web Scrapping/Web-scrapping-customers-sentiment/datasets/Unified_Monstrous_LM.csv",
  header = TRUE)

## Setup Paralell Programming 

detectCores() # 4 cores available
cluster <- makeCluster(3)
registerDoParallel(cluster) # Register cluster
getDoParWorkers() # check if there are now 2 cores working
# stopCluster(cluster) !do not forget to stop your cluster 


#### Exploring the data (Galaxy and Iphone) ####

summary(iphone_matr$iphone)
str(iphone_matr$iphone)
summary(galaxy_matr$samsunggalaxy)
str(galaxy_matr$samsunggalaxy)

# Plotting it

plot_ly(iphone_matr, x= ~iphone_matr$iphonesentiment, type='histogram')
plot_ly(iphone_matr, x= ~iphone_matr$iphone, type='histogram')
plot_ly(galaxy_matr, x= ~galaxy_matr$galaxysentiment, type='histogram')
plot_ly(galaxy_matr, x= ~galaxy_matr$samsunggalaxy, type='histogram')

# Check for NAs

apply(iphone_matr, 2, function(x) any(is.na(x))) # no NA in iPhone data
apply(galaxy_matr, 2, function(x) any(is.na(x))) # no NA in Galaxy data

# Delete duplicates

iphone_unique_m <- unique(iphone_matr) # from 12973 obs to 2582
galaxy_unique_m <- unique(galaxy_matr) # from 12973 obs to 2566


## Correlation Matrix with solely iPhone data  

iphone_vars <- c("iphone", "ios", "iphonecampos", "iphonecamneg", "iphonecamunc", 
                 "iphonedispos", "iphonedisneg", "iphonedisunc", "iphoneperpos", 
                 "iphoneperneg", "iphoneperunc", "iosperpos", "iosperneg", "iosperunc",
                 "iphonesentiment")

pure_iphone_df <- iphone_matr[,iphone_vars]

# delete rows that do not include terms
pure_iphone_df <- pure_iphone_df[!(pure_iphone_df$iphone==0 & pure_iphone_df$ios==0),] 

# delete duplicates as we are interested in the unique data 
pure_iphone_df <- unique(pure_iphone_df)

# Create a correlation matrix 
corr_iphone <- cor(pure_iphone_df)
corrplot(corr_iphone, method = "number",tl.cex= 0.7, number.cex = 0.8)

# Create decision tree
Iphone.tree <- ctree_control(maxdepth = 10)
IphoneDT <- ctree (iphonesentiment ~ ., data = pure_iphone_df)
plot(IphoneDT) 

# Linear Relationships are fairly weak as they are not strongly correlated
# Decision Tree highlights: iOs neg, iOs pos


## Correlation Matrix with solely Galaxy data 

galaxy_vars <- c("samsunggalaxy", "googleandroid", "samsungcampos", "samsungcamneg", "samsungcamunc", 
                 "samsungdispos", "samsungdisneg", "samsungdisunc", "samsungperpos", 
                 "samsungperneg", "samsungperunc", "googleperpos", "googleperneg", "googleperunc",
                 "galaxysentiment")

pure_galaxy_df <- galaxy_matr[,galaxy_vars]

# delete rows that do not include terms
pure_galaxy_df <- pure_galaxy_df[!(pure_galaxy_df$samsunggalaxy==0 & pure_galaxy_df$googleandroid==0),] 

# delete duplicates as we are interested in the unique data 
pure_galaxy_df <- unique(pure_galaxy_df)

# Create a correlation matrix 
corr_galaxy <- cor(pure_galaxy_df)
corrplot(corr_galaxy, method = "number",tl.cex= 0.7, number.cex = 0.8)

# Create decision tree
Galaxy.tree <- ctree_control(maxdepth = 10)
GalaxyDT <- ctree (galaxysentiment ~ ., data = pure_galaxy_df)
plot(GalaxyDT) 
varImp(GalaxyDT)
# No significant relationship predicts sentiment towards galaxy


#### Creating data sets #### 

#### NearZeroVariance of columns ####

# Iphone dataset without duplicates
iphone_nzv_metr <- nearZeroVar(iphone_unique_m, saveMetrics = T)
sum(iphone_nzv_metr$nzv=="TRUE") # 45 are near 0 Variance

iphone_nzv <- nearZeroVar(iphone_unique_m, saveMetrics = F) # all NZV columns

iphone_un_nozv <- iphone_unique_m[,-iphone_nzv] # final set

# Galaxy dataset without duplicates
galaxy_nzv_metr <- nearZeroVar(galaxy_unique_m, saveMetrics = T)
sum(galaxy_nzv_metr$nzv=="TRUE") # 45 are near 0 Variance

galaxy_nzv <- nearZeroVar(galaxy_unique_m, saveMetrics = F)
galaxy_un_nozv <- galaxy_unique_m[, -galaxy_nzv] # final galaxy set


#### Recursive Feature Elimination ####

set.seed(123)
RFE_ctrl <- rfeControl(functions = rfFuncs, method = "repeatedcv", # RF, cross-validation 
                   repeats = 5, verbose = FALSE)

## iPhone (all variabels)
start_rfe_58var <- Sys.time()
rfe_resu_58var <- rfe(iphone_unique_m[,1:58], 
                      iphone_unique_m$iphonesentiment, 
                      sizes=(1:58), 
                      rfeControl=RFE_ctrl)
stop_rfe_58var <- Sys.time()
time_rfe_58var <- stop_rfe_58var - start_rfe_58var

saveRDS(rfe_resu_58var, "RFE58variPhone.rds")
readRDS("RFE58variPhone.rds")

rfe_resu_58var # results = 17 variables (similar to nzv)
plot(rfe_resu_58var, type=c("g", "o"))

# create new df with variables from RFE
iphone_real_rfe <- iphone_unique_m[, predictors(rfe_resu_58var)]


##  Galaxy (all variabels)

galaxySample <- galaxy_unique_m[sample(1:nrow(galaxy_unique_m), 400, replace=FALSE),]

start_rfe_58var_ga <- Sys.time()
rfe_resu_58var_ga <- rfe(galaxySample[,1:58], 
                         galaxySample$galaxysentiment, 
                      sizes=(1:58), 
                      rfeControl=RFE_ctrl)
stop_rfe_58var_ga <- Sys.time()
time_rfe_58var_ga <- stop_rfe_58var_ga - start_rfe_58var_ga

saveRDS(rfe_resu_58var_ga, "RFE58varGalaxy.rds")
readRDS("RFE58varGalaxy.rds")

rfe_resu_58var_ga # results = 17 variables (similar to nzv)
plot(rfe_resu_58var_ga, type=c("g", "o"))

# create new df with variables from RFE
galaxy_real <- galaxy_unique_m[, predictors(rfe_resu_58var_ga)]


#### Principal Component Analysis ####

# removes all of your features and replaces them with mathematical representations of their variance
# data = training and testing from iphone_unique_m (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95

preprocessParams <- preProcess(iphone_unique_m[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams) # need 29 Components to capture 95% of output 

# split the data 
iphone.unique.partition <- createDataPartition(iphone_unique_m$iphonesentiment, times = 1, p = .7, 
                                            list = FALSE)
iphone_unique_train <- iphone_unique_m[iphone.unique.partition,]
iphone_unique_test <- iphone_unique_m[-iphone.unique.partition,]
iphone_unique_train$iphonesentiment <- as.factor(iphone_unique_train$iphonesentiment) # change to factor for classification
iphone_unique_test$iphonesentiment <- as.factor(iphone_unique_test$iphonesentiment)

# use predict to apply pca parameters, create training set, excluding dependent
train.iphone.pca <- predict(preprocessParams, iphone_unique_train[,-59])
train.iphone.pca$iphonesentiment <- as.factor(iphone_unique_train$iphonesentiment)
str(train.iphone.pca$iphonesentiment) 

# use predict to apply pca parameters, create test set, excluding dependent
test.iphone.pca <- predict(preprocessParams, iphone_unique_test[,-59])
test.iphone.pca$iphonesentiment <- as.factor(iphone_unique_test$iphonesentiment)
str(test.iphone.pca$iphonesentiment)

# inspect results
str(test.iphone.pca)
str(train.iphone.pca)


#### Model development ####
set.seed(123)

#### APPROACH: PCA data ####

### Train the models

## Model 1: RF with RF package
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2, returnData = T)
mtry_rf_pca_iphone <- tuneRF(train.iphone.pca[,-30], train.iphone.pca[,30], 
                             ntreeTry = 100, stepFactor = 2, improve = 0.05, trace = TRUE, plot = TRUE)
rf_pca_iphone_mdl <- randomForest(y = train.iphone.pca[,30], x = train.iphone.pca[,-30], 
                                  importance = T, ntree = 100, mtry = 3, trControl = control)
## Model 2: RF with caret package
rf_pca_iphone_mdl_car <- caret::train(iphonesentiment~., data = train.iphone.pca, 
                                      method = "rf", trControl=control, tuneLength = 2)
## Model 3: Kknn with train.kknn
kknn_pca_iphone_mdl <- train.kknn(formula = iphonesentiment~., data = train.iphone.pca, kmax = 11,
                                  distance = 2, kernel = "optimal", trControl = control)
# Model 4: SVM
svm_pca_iphone_mdl <- svm(formula = iphonesentiment~., data = train.iphone.pca, trControl = control, scale = T)

### Test the models 

## predict 
Pred_rf_pca_iphone <- predict(rf_pca_iphone_mdl,test.iphone.pca)
Pred_rf_pca_iphone_car <- predict(rf_pca_iphone_mdl_car,test.iphone.pca)
Pred_kknn_pca_iphone <- predict(kknn_pca_iphone_mdl,test.iphone.pca)
Pred_svm_cv_pca_iphone <- predict(svm_pca_iphone_mdl,test.iphone.pca)

## Postresamples
PR_rf_pca_rfmdl <- as.data.frame(postResample(pred = Pred_rf_pca_iphone, obs = test.iphone.pca$iphonesentiment))
PR_rf_pca_caret <- as.data.frame(postResample(pred = Pred_rf_pca_iphone_car, obs = test.iphone.pca$iphonesentiment))
PR_knn_pca <- as.data.frame(postResample(pred = Pred_kknn_pca_iphone, obs = test.iphone.pca$iphonesentiment))
PR_svm_pca <- as.data.frame(postResample(pred = Pred_svm_cv_pca_iphone, obs = test.iphone.pca$iphonesentiment))

Results_PCA_class <- cbind(PR_rf_pca_rfmdl, PR_rf_pca_caret, PR_knn_pca, PR_svm_pca)
setnames(Results_PCA_class, 
         old = c("postResample(pred = Pred_rf_pca_iphone, obs = test.iphone.pca$iphonesentiment)",
                 "postResample(pred = Pred_rf_pca_iphone_car, obs = test.iphone.pca$iphonesentiment)",
                 "postResample(pred = Pred_kknn_pca_iphone, obs = test.iphone.pca$iphonesentiment)",
                 "postResample(pred = Pred_svm_cv_pca_iphone, obs = test.iphone.pca$iphonesentiment)"), 
         new = c("RF_RFPack", "RF_Caret", "KKNN", "SVM"))
            

#### APPROACH: NZV set ####

# split the data 
iphone.nzv.partition <- createDataPartition(iphone_un_nozv$iphonesentiment, times = 1, p = .7, list = FALSE)
iphone_nzv_train <- iphone_un_nozv[iphone.nzv.partition,]
iphone_nzv_test <- iphone_un_nozv[-iphone.nzv.partition,]
iphone_nzv_train$iphonesentiment <- as.factor(iphone_nzv_train$iphonesentiment)
iphone_nzv_test$iphonesentiment <- as.factor(iphone_nzv_test$iphonesentiment)

## Model 1: RF with RF package
mtry_rf_nzv_iphone <- tuneRF(iphone_nzv_train[,-14], iphone_nzv_train[,14], 
                             ntreeTry = 100, stepFactor = 2, improve = 0.05, trace = TRUE, plot = TRUE)

rf_nzv_iphone_mdl <- randomForest(y = iphone_nzv_train[,14], x = iphone_nzv_train[,-14], 
                                  importance = T, ntree = 100, mtry = 3, trControl = control)
## Model 2: RF with caret package
rf_nzv_iphone_mdl_car <- caret::train(iphonesentiment~., data = iphone_nzv_train, 
                                      method = "rf", trControl=control, tuneLength = 2)
## Model 3: RF with train.knn 
kknn_nzv_iphone_mdl <- train.kknn(formula = iphonesentiment~., data = iphone_nzv_train, kmax = 11,
                                 distance = 2, kernel = "optimal", trControl = control)
## Model 4: RF with train.knn 
svm_nzv_iphone_mdl <- svm(formula = iphonesentiment~., data = iphone_nzv_train, trControl = control, scale = T)

## predict 
pred_rf_iph_nzv_rfmdl <- predict(rf_nzv_iphone_mdl,iphone_nzv_test)
pred_rf_iph_nzv_caret <- predict(rf_nzv_iphone_mdl_car,iphone_nzv_test)
pred_kknn_iph_nzv <- predict(kknn_nzv_iphone_mdl,iphone_nzv_test)
pred_svm_nzv_iphone_mdl <- predict(svm_nzv_iphone_mdl,iphone_nzv_test)

# postresamples 
PR_rf_iph_nzv_caret <- as.data.frame(postResample(pred = pred_rf_iph_nzv_caret, obs = iphone_nzv_test$iphonesentiment))
PR_rf_iph_nzv_rfmdl <- as.data.frame(postResample(pred = pred_rf_iph_nzv_rfmdl, obs = iphone_nzv_test$iphonesentiment))
PR_kknn_iph_nzv <- as.data.frame(postResample(pred = pred_kknn_iph_nzv, obs = iphone_nzv_test$iphonesentiment))
PR_svm_nzv_iphone_mdl <- as.data.frame(postResample(pred = pred_svm_nzv_iphone_mdl, obs = iphone_nzv_test$iphonesentiment))

# Results
Results_NZV_class <- cbind(PR_rf_iph_nzv_rfmdl, PR_rf_iph_nzv_caret, PR_kknn_iph_nzv, PR_svm_nzv_iphone_mdl)
setnames(Results_NZV_class, 
         old = c("postResample(pred = pred_rf_iph_nzv_rfmdl, obs = iphone_nzv_test$iphonesentiment)",
                 "postResample(pred = pred_rf_iph_nzv_caret, obs = iphone_nzv_test$iphonesentiment)",
                 "postResample(pred = pred_kknn_iph_nzv, obs = iphone_nzv_test$iphonesentiment)",
                 "postResample(pred = pred_svm_nzv_iphone_mdl, obs = iphone_nzv_test$iphonesentiment)"), 
         new = c("RF_RFPack", "RF_Caret", "KKNN", "SVM"))

# Combine Results
Results_NZV_class$Approach <- "Near_Zero_Var"
Results_PCA_class$Approach <- "PCA"

Results_overall <- rbind(Results_NZV_class, Results_PCA_class)
Results_Kappa <- Results_overall[-c(1,3),]
Results_Accuracy <- Results_overall[-c(2,4),]

Res_Kappa_melt <- melt(Results_Kappa)
Res_Acc_melt <- melt(Results_Accuracy)

# Near-Zero-Variance approach delivers better results
ggplot(Res_Kappa_melt, aes(x = variable,y=value,fill=variable)) + geom_bar(stat = "identity")+facet_wrap(~Approach)+
  coord_flip()+ggtitle("Kappa Comparison between NZV and PCA Approach")
ggplot(Res_Acc_melt, aes(x = variable,y=value,fill=variable)) + geom_bar(stat = "identity")+facet_wrap(~Approach)+
  coord_flip()+ggtitle("Accuracy Comparison between NZV and PCA Approach")

# RF package in the NZV Approach offers the best Kappa and Accuracy 
large_matrix$iphonesentiment_RF <- predict(rf_nzv_iphone_mdl, large_matrix)
histogram(large_matrix$iphonesentiment_RF) #class imbalance
# where is the class imbalance coming from?
# need to check the results for the class imbalance before training the model 

#### Class imbalance & tactics ####

class_imb <- melt(summary(iphone_nzv_train$iphonesentiment)) # class imbalance in big matrix stems from 
class_imb$variable <- c(0:5)

ggplot(class_imb, aes(x = variable, y = value, fill = variable))+ geom_bar(stat = "identity")+
  ggtitle("Class imbalance of training set")+xlab("Sentiment category from 0-5")+ylab("Count")

### Approach 1: trainControl up/downsapling
## train models
# upSample 
upcontrol <- trainControl(method = "repeatedcv", number = 10, repeats = 2, returnData = T, sampling = "up")

rf_nzv_iph_up <- randomForest(y = iphone_nzv_train[,14], x = iphone_nzv_train[,-14], 
                                  importance = T, ntree = 100, mtry = 3, trControl = upcontrol)

# downSample 
downcontrol <- trainControl(method = "repeatedcv", number = 10, repeats = 2, returnData = T, sampling = "down")

rf_nzv_iph_down <- randomForest(y = iphone_nzv_train[,14], x = iphone_nzv_train[,-14], 
                              importance = T, ntree = 100, mtry = 3, trControl = downcontrol)

## test models

# upSample 
pre_rf_test_up <- predict(rf_nzv_iph_up, iphone_nzv_test)
up_test <- summary(pre_rf_test_up)

# downSample
pre_rf_test_down <- predict(rf_nzv_iph_down, iphone_nzv_test)
down_test <- summary(pre_rf_test_down)

# normal 
pre_rf_test_neutral <- predict(rf_nzv_iphone_mdl, iphone_nzv_test)
neutral_test <- summary(pre_rf_test_neutral)

# Visualise impact of class imbalance tactics 
class_imb_test <- melt(rbind(up_test, down_test, neutral_test))
ggplot(class_imb_test, aes(x = Var2, y = value, fill = Var2))+ geom_bar(stat = "identity")+
  facet_wrap(~Var1)+ggtitle("Tactics on class imbalance problem")+xlab("Sentiment category from 0-5")+ylab("Count")
 # it appeats to be not working 

### Approach 2: upSample/Downsample function 

## DownSampling manually  
# change to factor
iphone_un_nozv$iphonesentiment <- as.factor(iphone_un_nozv$iphonesentiment) 

# downsample whole set
iphone_nzv_full_d <- downSample(x = iphone_un_nozv, y = iphone_un_nozv$iphonesentiment)
table(iphone_nzv_full_d$iphonesentiment)

# split the data
set.seed(123)
iph.nzv.down.partition <- createDataPartition(iphone_nzv_full_d$iphonesentiment, times = 1, p = .7, list = FALSE)
iph.nzv.down.train <- iphone_nzv_full_d[iph.nzv.down.partition,]
iph.nzv.down.test <- iphone_nzv_full_d[-iph.nzv.down.partition,]
iph.nzv.down.train$Class <- NULL
iph.nzv.down.test$Class <- NULL

# train
rf_nzv_i_A2_down <- randomForest(y = iph.nzv.down.train[,14], x = iph.nzv.down.train[,-14], 
                              importance = T, ntree = 100, mtry = 3, trControl = control)
# test
pre_rf_A2_test_down <- predict(rf_nzv_i_A2_down ,iph.nzv.down.test)
A2_test_down <- summary(pre_rf_A2_test_down)

## UpSampling manually  

# downsample whole set
iphone_nzv_full_up <- upSample(x = iphone_un_nozv, y = iphone_un_nozv$iphonesentiment)
table(iphone_nzv_full_up$iphonesentiment)

# split the data
set.seed(123)
iph.nzv.up.partition <- createDataPartition(iphone_nzv_full_up$iphonesentiment, times = 1, p = .7, list = FALSE)
iph.nzv.up.train <- iphone_nzv_full_up[iph.nzv.up.partition,]
iph.nzv.up.test <- iphone_nzv_full_up[-iph.nzv.up.partition,]
iph.nzv.up.train$Class <- NULL
iph.nzv.up.test$Class <- NULL

# train
rf_nzv_i_A2_up <- randomForest(y = iph.nzv.up.train[,14], x = iph.nzv.up.train[,-14], 
                                 importance = T, ntree = 100, mtry = 3, trControl = control)
# test
pre_rf_A2_test_up <- predict(rf_nzv_i_A2_up ,iph.nzv.up.test)
A2_test_up <- summary(pre_rf_A2_test_up)

# Visualise impact of class imbalance tactics Approach 2 
class_imb_test2down <- melt(rbind(A2_test_down))
class_imb_test2up <- melt(rbind(A2_test_up))

ggplot(class_imb_test2down, aes(x = Var2, y = value/sum(value), fill = Var2))+ geom_bar(stat = "identity")+
  facet_wrap(~Var1)+ggtitle("Tactics 2 on class imbalance problem")+xlab("Sentiment category from 0-5")+ylab("Count")

ggplot(class_imb_test2up, aes(x = Var2, y = value/sum(value), fill = Var2))+ geom_bar(stat = "identity")+
  facet_wrap(~Var1)+ggtitle("Tactics 2 on class imbalance problem")+xlab("Sentiment category from 0-5")+ylab("Count")

# Approach 2 appears to be working the best















