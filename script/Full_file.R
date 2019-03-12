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

## NearZeroVariance of columns 

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


### Recursive Feature Elimination 

set.seed(123)
RFE_ctrl <- rfeControl(functions = rfFuncs, method = "repeatedcv", # RF, cross-validation 
                   repeats = 5, verbose = FALSE)
# iPhone (all variabels)
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


# Galaxy (all variabels)
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


#### Model development ####
set.seed(123)

### NZV set 

# split the data 
iphone.nzv.partition <- createDataPartition(iphone_un_nozv$iphonesentiment, times = 1, p = .7, 
                                        list = FALSE)
iphone_nzv_train <- iphone_un_nozv[iphone.nzv.partition,]
iphone_nzv_test <- iphone_un_nozv[-iphone.nzv.partition,]
iphone_nzv_train$iphonesentiment <- as.factor(iphone_nzv_train$iphonesentiment)
iphone_nzv_test$iphonesentiment <- as.factor(iphone_nzv_test$iphonesentiment)

# RF
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2, returnData = T)

# best mtry 
iph_nzv_vector <- names(iphone_nzv_train)
iph_nzv_vec_sen <- iph_nzv_vector[14] # vector name sentiment
iph_nzv_vec_rest <- iph_nzv_vector[1:13]

mtry_rf_nzv_iphone <- tuneRF(iphone_nzv_train[,iph_nzv_vec_rest], 
                             iphone_nzv_train[,iph_nzv_vec_sen], ntreeTry = 100, stepFactor = 2,
                             improve = 0.05, trace = TRUE, plot = TRUE)

rf_nzv_iphone_mdl <- randomForest(y = iphone_nzv_train[,iph_nzv_vec_sen], 
                                  x = iphone_nzv_train[,iph_nzv_vec_rest], importance = T, ntree = 100,
                                  mtry = 3, trControl = control)

rf_nzv_iphone_mdl_car <- caret::train(iphonesentiment~.,
                         data = iphone_nzv_train, method = "rf", trControl=control,
                         tuneLength = 2)

# Kknn
kknn_nzv_iphone_mdl <- train.kknn(formula = iphonesentiment~., data = iphone_nzv_train, kmax = 11,
                                 distance = 2, kernel = "optimal", trControl = control)

kknn_nzv_iphone_mdl_cv <- cv.kknn(formula = iphonesentiment~., data = iphone_nzv_train,
                                  kcv = 10)

# C5.0
C5iphone_nzv_train <- iphone_nzv_train
C5iphone_nzv_train$iphonesentiment <- as.factor( #change to factor
  C5iphone_nzv_train$iphonesentiment)

C5_nzv_iphone_mdl <- C50::C5.0(x = C5iphone_nzv_train[,iph_nzv_vec_rest], 
                               y = C5iphone_nzv_train$iphonesentiment, trControl = control)

# SVM
svm_nzv_iphone_mdl <- svm(formula = iphonesentiment~., data = iphone_nzv_train, 
                            trControl = control, scale = T)

# Models
svm_nzv_iphone_mdl
C5_nzv_iphone_mdl
kknn_nzv_iphone_mdl_cv
kknn_nzv_iphone_mdl
rf_nzv_iphone_mdl
rf_nzv_iphone_mdl_car

## predict 
pred_rf_iph_nzv_caret <- predict(rf_nzv_iphone_mdl_car,iphone_nzv_test)
pred_rf_iph_nzv_rfmdl <- predict(rf_nzv_iphone_mdl,iphone_nzv_test)
pred_kknn_iph_nzv <- predict(kknn_nzv_iphone_mdl,iphone_nzv_test)
pred_C5_nzv_iphone_mdl <- predict(C5_nzv_iphone_mdl,iphone_nzv_test)
pred_svm_nzv_iphone_mdl <- predict(svm_nzv_iphone_mdl,iphone_nzv_test)

# postresamples 
PR_rf_iph_nzv_caret <- postResample(pred = pred_rf_iph_nzv_caret, obs = iphone_nzv_test$iphonesentiment)
PR_rf_iph_nzv_rfmdl <- postResample(pred = pred_rf_iph_nzv_rfmdl, obs = iphone_nzv_test$iphonesentiment)
PR_kknn_iph_nzv <- postResample(pred = pred_kknn_iph_nzv, obs = iphone_nzv_test$iphonesentiment)
PR_svm_nzv_iphone_mdl <- postResample(pred = pred_svm_nzv_iphone_mdl, obs = iphone_nzv_test$iphonesentiment)

C5iphone_nzv_test <- iphone_nzv_test
C5iphone_nzv_test$predictsentiment <- pred_C5_nzv_iphone_mdl
CM_C5_nzv_iphone_mdl <- confusionMatrix(pred_C5_nzv_iphone_mdl, C5iphone_nzv_test$predictsentiment)
PR_C5_nzv_iphone_mdl <- postResample(pred_C5_nzv_iphone_mdl, C5iphone_nzv_test$predictsentiment)

# Regression/Classification results
PR_rf_iph_nzv_caret <- as.data.frame(PR_rf_iph_nzv_caret)
PR_rf_iph_nzv_rfmdl <- as.data.frame(PR_rf_iph_nzv_rfmdl)
PR_kknn_iph_nzv <- as.data.frame(PR_kknn_iph_nzv)
PR_svm_nzv_iphone_mdl <- as.data.frame(PR_svm_nzv_iphone_mdl)

# Results
Regression_results <- cbind(PR_rf_iph_nzv_caret, PR_rf_iph_nzv_rfmdl, PR_kknn_iph_nzv, PR_svm_nzv_iphone_mdl)
Classification_results <- cbind(PR_rf_iph_nzv_caret, PR_rf_iph_nzv_rfmdl, PR_kknn_iph_nzv, PR_svm_nzv_iphone_mdl)









