Wi-Fi Fingerprinting
================
Fernanda N.
9 October 2018

The main goal of this task is to evaluate the application of machine learning techniques to the problem of indoor localization via Wi-Fi fingerprinting.

#### **CODE:**

``` r
#Loading libraries
library("readr")
library("dplyr")
library("tidyr")
library("lubridate")
library("ggplot2")
library("caret")
library("ggthemes")
library("reshape2")
library("plotly")
library("lattice")
library("colorspace")
  
#loading the datasets
trainingData <- as.data.frame(read_csv("trainingData.csv"))
validationData <- as.data.frame(read_csv("validationData.csv"))
```

##### **Initial preprocessing**

``` r
#lower case column names
names(trainingData) <- tolower(names(trainingData))
names(validationData) <- tolower(names(validationData))

#joining both datasets for preprocessing
#creating partition column
trainingData$partition <- "train"
validationData$partition <- "validation"

#combining dataframes
dataCombined <- rbind(trainingData, validationData)

#converting time
dataCombined$timestamp <-
  as.POSIXct(dataCombined$timestamp, origin = "1970-01-01", tz = "CET")

#factorizing columns
dataCombined$floor <- as.factor(dataCombined$floor)
dataCombined$buildingid <- as.factor(dataCombined$buildingid)
```

##### **Preprocessing**

``` r
dataClean <- dataCombined

#replacing WAP signal < -90 dBm (unusable) to -100 (no signal)
dataClean[, 1:(ncol(dataClean) - 10)] <-
  apply(dataClean[, 1:(ncol(dataClean) - 10)], 2, function(x)
    ifelse(x < -90,-100, x))

#replacing WAP signal = 100 to -100 (both = no signal at all)
dataClean[, 1:(ncol(dataClean) - 10)] <-
  apply(dataClean[, 1:(ncol(dataClean) - 10)], 2, function(x)
    ifelse(x == 100,-100, x))

#splitting df
trClean <- dataClean %>% filter(partition == "train")
valClean <- dataClean %>% filter(partition == "validation")

#removing columns with 0 variance
trClean <- trClean[-which(apply(trClean[, 1:(ncol(trClean) - 10)], 2, var) == 0)]

#keeping only unique rows
trClean <- unique(trClean)

#creating df to normalize without the rows with standard deviation = 0, so it doesn't get NaNs
trClean <-
  trClean[which(apply(trClean[, 1:(ncol(trClean) - 10)], 1, sd) != 0),]

valClean <-
  valClean[which(apply(valClean[, 1:(ncol(valClean) - 10)], 1, sd) != 0),]

#keeping same columns in training and validation df
valClean <- valClean[, which(names(valClean) %in% names(trClean))]
```

##### **Normalizing the data**

``` r
#normalizing by rows
#training
trNormRow <- trClean
trNormRow[, 1:(ncol(trNormRow) - 10)]  <-
  t(apply(trClean[, 1:(ncol(trClean) - 10)], 1, function(x)
    (x - min(x)) / (max(x) - min(x))))

#validation
valNormRow <- valClean
valNormRow[, 1:(ncol(valNormRow) - 10)]  <-
  t(apply(valClean[, 1:(ncol(valClean) - 10)], 1, function(x)
    (x - min(x)) / (max(x) - min(x))))
```

------------------------------------------------------------------------

##### **Data Visualization**

``` r
#colorspace palette
palette <- choose_palette()
colors <- rainbow_hcl(18)

#creating copy of the df to use with plotly
dataPlot <- trClean
dataPlot$floor <- as.numeric(dataPlot$floor)
dataPlot$buildingid <- as.numeric(dataPlot$buildingid)

#range of z axis (floor)
axz <- list(nticks = 6, range = c(0, 4), title = 'Floor')
```

``` r
#location of users
plotlyPhoneLocation <-
  plot_ly(
    dataPlot,
    x = dataPlot$latitude,
    y = dataPlot$longitude,
    z = dataPlot$floor,
    color = ~as.factor(dataPlot$userid),
    colors = colors,
    marker = list(size = 3)
  ) %>% add_markers() %>% layout(scene = list(
    xaxis = list(title = 'Latitude'),
    yaxis = list(title = 'Longitude'),
    zaxis = axz
  ))

plotlyPhoneLocation
```

![](plots/plotuserlocation.png)

``` r
#wifi signals strength
#creating column showing which is the max wap signal
dataPlot <- dataPlot %>% mutate(rowmax = apply(dataPlot[, 1:404],1,max))

#creating column with wifi signal strength
dataPlot <- dataPlot %>%
  mutate(sig_strength = case_when(rowmax < -75 ~ "low", rowmax >= -75 &
                                    rowmax <= -50 ~ "med", rowmax > -50 ~ "high"))

plotlySignal <-
  plot_ly(
    dataPlot,
    x = dataPlot$latitude,
    y = dataPlot$longitude,
    z = dataPlot$floor,
    color = dataPlot$sig_strength,
    colors = c("green", "red", "yellow"),
    marker = list(size = 3)
  ) %>% add_markers() %>% layout(scene = list(
    xaxis = list(title = 'Latitude'),
    yaxis = list(title = 'Longitude'),
    zaxis = axz
  ))

plotlySignal
```

![](plotwifistrength.png)

------------------------------------------------------------------------

##### **Training and Validation**

###### **Predicting Building**

``` r
set.seed(123)

#sampling the data (3000 rows) - normalized waps
sampleTrBuild <- trNormRow[sample(nrow(trNormRow), 3000),]

#creating data partition to predict BUILDING ID
inTrainBuild <- createDataPartition(
  sampleTrBuild$buildingid,
  p = .75,
  list = FALSE
)

#creating training and testing df
trainBuild <- sampleTrBuild[inTrainBuild,]
testBuild <-  sampleTrBuild[-inTrainBuild,]

#cross validation
ctrl <- trainControl(
  method = "repeatedcv",
  number = 3,
  repeats = 3
)
```

``` r
#models
#SVM 3 (best result)
SVM3_Building <- train(
  buildingid ~ .,
  data = trainBuild[, c(1:404,408)],
  method = "svmLinear3",
  scale = FALSE,
  trControl = ctrl
)

SVM3_Building
```

    ## L2 Regularized Support Vector Machine (dual) with Linear Kernel 
    ## 
    ## 2251 samples
    ##  404 predictor
    ##    3 classes: '0', '1', '2' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold, repeated 3 times) 
    ## Summary of sample sizes: 1499, 1501, 1502, 1501, 1500, 1501, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cost  Loss  Accuracy   Kappa    
    ##   0.25  L1    0.9991115  0.9985940
    ##   0.25  L2    0.9991115  0.9985940
    ##   0.50  L1    0.9991115  0.9985940
    ##   0.50  L2    0.9989632  0.9983592
    ##   1.00  L1    0.9991115  0.9985940
    ##   1.00  L2    0.9986671  0.9978910
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were cost = 0.25 and Loss = L1.

``` r
#testing
Test_PredSVM3_Build <- predict(
  SVM3_Building,
  newdata = testBuild)

postResample(Test_PredSVM3_Build, testBuild$buildingid)
```

    ## Accuracy    Kappa 
    ##        1        1

``` r
#validation
Val_PredSVM3_Build <- predict(
  SVM3_Building,
  newdata = valNormRow)

postResample(Val_PredSVM3_Build, valNormRow$buildingid)
```

    ##  Accuracy     Kappa 
    ## 0.9990983 0.9985736

###### **Predicting Floor**

``` r
#creating a new column with the predicted building
valNormRow$predbuildingid <- factor(Val_PredSVM3_Build)

#subsetting buildings (training)
building0 <- trNormRow[which(trNormRow$buildingid == '0'), ]
building1 <- trNormRow[which(trNormRow$buildingid == '1'), ]
building2 <- trNormRow[which(trNormRow$buildingid == '2'), ]

#replacing the actual buildings with the predicted ones
valNormRow$buildingid <- valNormRow$predbuildingid

#subsetting buildings (validation)
vbuilding0 <- valNormRow[which(valNormRow$buildingid == '0'), ]
vbuilding1 <- valNormRow[which(valNormRow$buildingid == '1'), ]
vbuilding2 <- valNormRow[which(valNormRow$buildingid == '2'), ]

#refactoring 'floor' to drop unused levels
building0$floor <- factor(building0$floor)
building1$floor <- factor(building1$floor)
building2$floor <- factor(building2$floor)

vbuilding0$floor <- factor(vbuilding0$floor)
vbuilding1$floor <- factor(vbuilding1$floor)
vbuilding2$floor <- factor(vbuilding2$floor)

#sampling the data (3000 rows each) - normalized rows
sampleTr0 <- building0[sample(nrow(building0), 3000),]
sampleTr1 <- building1[sample(nrow(building1), 3000),]
sampleTr2 <- building2[sample(nrow(building2), 3000),]

# View(head(sampleTr))

#predicting
#creating a separate data partition for each building
inTrain0 <- createDataPartition(
  sampleTr0$floor,
  p = .75,
  list = FALSE
)

inTrain1 <- createDataPartition(
  sampleTr1$floor,
  p = .75,
  list = FALSE
)

inTrain2 <- createDataPartition(
  sampleTr2$floor,
  p = .75,
  list = FALSE
)

#creating training and testing df
trainNormRow0 <- sampleTr0[inTrain1,]
testNormRow0 <-  sampleTr0[-inTrain1,]

trainNormRow1 <- sampleTr1[inTrain1,]
testNormRow1 <-  sampleTr1[-inTrain1,]

trainNormRow2 <- sampleTr2[inTrain2,]
testNormRow2 <-  sampleTr2[-inTrain2,]

#overwriting with predictions for building
testNormRow_all <- rbind(testNormRow0,testNormRow1,testNormRow2)
testNormRow_all$buildingid <- predict(SVM3_Building,newdata = testNormRow_all)

#subsetting
testNormRow0 <- testNormRow_all[which(testNormRow_all$buildingid == '0'), ]
testNormRow1 <- testNormRow_all[which(testNormRow_all$buildingid == '1'), ]
testNormRow2 <- testNormRow_all[which(testNormRow_all$buildingid == '2'), ]
```

``` r
#training models
#> SVM Linear
SVM_Floor0 <- train(
  floor ~ .,
  data = trainNormRow0[, c(1:404,407)],
  method = "svmLinear",
  scale = FALSE,
  trControl = ctrl
)

SVM_Floor0
```

    ## Support Vector Machines with Linear Kernel 
    ## 
    ## 2252 samples
    ##  404 predictor
    ##    4 classes: '0', '1', '2', '3' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold, repeated 3 times) 
    ## Summary of sample sizes: 1502, 1501, 1501, 1501, 1501, 1502, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.9928954  0.9904857
    ## 
    ## Tuning parameter 'C' was held constant at a value of 1

``` r
SVM_Floor1 <- train(
  floor ~ .,
  data = trainNormRow1[, c(1:404,407)],
  method = "svmLinear",
  scale = FALSE,
  trControl = ctrl
)

SVM_Floor1
```

    ## Support Vector Machines with Linear Kernel 
    ## 
    ## 2252 samples
    ##  404 predictor
    ##    4 classes: '0', '1', '2', '3' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold, repeated 3 times) 
    ## Summary of sample sizes: 1502, 1501, 1501, 1500, 1502, 1502, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.9902323  0.9868299
    ## 
    ## Tuning parameter 'C' was held constant at a value of 1

``` r
SVM_Floor2 <- train(
  floor ~ .,
  data = trainNormRow2[, c(1:404,407)],
  method = "svmLinear",
  scale = FALSE,
  trControl = ctrl
)

SVM_Floor2
```

    ## Support Vector Machines with Linear Kernel 
    ## 
    ## 2251 samples
    ##  404 predictor
    ##    5 classes: '0', '1', '2', '3', '4' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold, repeated 3 times) 
    ## Summary of sample sizes: 1501, 1500, 1501, 1500, 1501, 1501, ... 
    ## Resampling results:
    ## 
    ##   Accuracy  Kappa    
    ##   0.994077  0.9923324
    ## 
    ## Tuning parameter 'C' was held constant at a value of 1

``` r
#testing
#floors from building 0
Test_PredSVM_Floor0 <- predict(
  SVM_Floor0,
  newdata = testNormRow0)

postResample(Test_PredSVM_Floor0, testNormRow0$floor)
```

    ##  Accuracy     Kappa 
    ## 0.9919786 0.9892393

``` r
#building 1
Test_PredSVM_Floor1 <- predict(
  SVM_Floor1,
  newdata = testNormRow1)

postResample(Test_PredSVM_Floor1, testNormRow1$floor)
```

    ##  Accuracy     Kappa 
    ## 0.9906417 0.9873719

``` r
#building 2
Test_PredSVM_Floor2 <- predict(
  SVM_Floor2,
  newdata = testNormRow2)

postResample(Test_PredSVM_Floor2, testNormRow2$floor)
```

    ##  Accuracy     Kappa 
    ## 0.9933244 0.9913607

``` r
#validation
#building 0
Val_PredSVM_Floor0 <- predict(
  SVM_Floor0,
  newdata = vbuilding0)

postResample(Val_PredSVM_Floor0, vbuilding0$floor)
```

    ##  Accuracy     Kappa 
    ## 0.9590317 0.9420621

``` r
#building 1
Val_PredSVM_Floor1 <- predict(
  SVM_Floor1,
  newdata = vbuilding1)

postResample(Val_PredSVM_Floor1, vbuilding1$floor)
```

    ##  Accuracy     Kappa 
    ## 0.8355263 0.7654430

``` r
#building 2
Val_PredSVM_Floor2 <- predict(
  SVM_Floor2,
  newdata = vbuilding2)

postResample(Val_PredSVM_Floor2, vbuilding2$floor)
```

    ##  Accuracy     Kappa 
    ## 0.9477612 0.9290227

###### **Predicting Longitude & Latitude**

``` r
inTrainLAT <- createDataPartition(
  sampleTrBuild$latitude, #recycling sample from training building
  p = .75,
  list = FALSE
)

inTrainLONG <- createDataPartition(
  sampleTrBuild$longitude,
  p = .75,
  list = FALSE
)

#creating training and testing df
trainLat <- sampleTrBuild[inTrainLAT,]
testLat <-  sampleTrBuild[-inTrainLAT,]

trainLong <- sampleTrBuild[inTrainLONG,]
testLong <-  sampleTrBuild[-inTrainLONG,]
```

``` r
#training models
#longitude
KNN_Long <- train(
  longitude ~ .,
  data = trainLong[, c(1:404,405)],
  method = "knn",
  scale = FALSE,
  trControl = ctrl
)

KNN_Long
```

    ## k-Nearest Neighbors 
    ## 
    ## 2251 samples
    ##  404 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold, repeated 3 times) 
    ## Summary of sample sizes: 1501, 1501, 1500, 1501, 1501, 1500, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   k  RMSE       Rsquared   MAE     
    ##   5   9.408354  0.9942438  5.368835
    ##   7  10.026806  0.9934830  5.756202
    ##   9  10.410379  0.9929906  6.059476
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was k = 5.

``` r
#latitude
KNN_Lat <- train(
  latitude ~ .,
  data = trainLat[, c(1:404,406)],
  method = "knn",
  scale = FALSE,
  trControl = ctrl
)

KNN_Lat
```

    ## k-Nearest Neighbors 
    ## 
    ## 2252 samples
    ##  404 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold, repeated 3 times) 
    ## Summary of sample sizes: 1501, 1502, 1501, 1501, 1502, 1501, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   k  RMSE      Rsquared   MAE     
    ##   5  7.138665  0.9884772  4.649266
    ##   7  7.392660  0.9876779  4.917157
    ##   9  7.742436  0.9864607  5.180600
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was k = 5.

``` r
#testing
#longitude
Test_PredKNN_Long <- predict(
  KNN_Long,
  newdata = testLong)

postResample(Test_PredKNN_Long, testLong$longitude)
```

    ##      RMSE  Rsquared       MAE 
    ## 9.9947220 0.9936121 4.9572542

``` r
#latitude
Test_PredKNN_Lat <- predict(
  KNN_Lat,
  newdata = testLat)

postResample(Test_PredKNN_Lat, testLat$latitude)
```

    ##      RMSE  Rsquared       MAE 
    ## 5.8053214 0.9924051 3.8127933

``` r
#validation
#longitude
Val_PredKNN_Long <- predict(
  KNN_Long,
  newdata = valNormRow)

postResample(Val_PredKNN_Long, valNormRow$longitude)
```

    ##      RMSE  Rsquared       MAE 
    ## 12.039127  0.990010  5.907179

``` r
#latitude
Val_PredKNN_Lat <- predict(
  KNN_Lat,
  newdata = valNormRow)

postResample(Val_PredKNN_Lat, valNormRow$latitude)
```

    ##      RMSE  Rsquared       MAE 
    ## 8.2690800 0.9863706 5.3853085

``` r
#checking the error distance
valNormRow$error_distance <- sqrt((valNormRow$latitude - Val_PredKNN_Lat)^2 +
                                    (valNormRow$longitude - Val_PredKNN_Long)^2)
summary(valNormRow$error_distance)
```

    ##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
    ##   0.06434   3.65477   6.43402   8.88939  10.38253 207.42592

``` r
boxplot(valNormRow$error_distance)
```

![](wifi_markdown_files/figure-markdown_github/unnamed-chunk-30-1.png)
