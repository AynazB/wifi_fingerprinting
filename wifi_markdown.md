Wi-Fi Fingerprinting
================
Fernanda N.
9 September 2018

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

![](plotuserlocation.png)

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

###### **Building**

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
