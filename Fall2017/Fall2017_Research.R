# Author: Matthew Sears
# Advisor: Semere Habtemicael
# Data: Stock Portfolio Performance Data Set (UC Irvine ML Repo)

library("readxl") # To read excel file

# Set working directory and clear workspace
setwd("~/Desktop/Research/Data/")
rm(list = ls())

# 1st, 2nd periods will be training data and 3rd period will be testing data.
sheetList = c("1st period", "2nd period", "3rd period")

# Labels for table of training model results
tableNames <- c("DOF","Sum-of-Squares","MSE","F Ratio","Prob > F")

for(i in 1:length(sheetList)) {
  
  # Uncomment i and run line by line to view results of the model
  # before the removal of outliers. Also change to i to 2 to
  # view second training data set
  # i <- 1
  
  # Read in excel data
  rawData <- data.frame(read_excel("stock portfolio performance data set.xlsx",
                        sheet = sheetList[i]))
  rawData <- rawData[-1,]


  # Extract x1 - x6 (weighted concepts) and y (annual return) as num
  data <- data.frame(x1 = as.numeric(rawData[,c(2)]),
                     x2 = as.numeric(rawData[,c(3)]),
                     x3 = as.numeric(rawData[,c(4)]),
                     x4 = as.numeric(rawData[,c(5)]),
                     x5 = as.numeric(rawData[,c(6)]),
                     x6 = as.numeric(rawData[,c(7)]),
                     y  = as.numeric(rawData[,c(8)]))
 
  # Uncomment to view Cook's Distance info before removal of outliers.
  # Also, skip to the if statement (after running this line by line)
  # to view the results of model before removal of outliers.
  # linReg_1  <- lm(y ~ -1 + ., data)
  # quadReg_1 <- lm(y ~ -1 + . + .^2, data)
  # linReg_2  <- lm(y ~ -1 + ., data)
  # quadReg_2 <- lm(y ~ -1 + . + .^2, data)
  # plot(linReg_1)
  # plot(quadReg_1)
  # plot(linReg_2)
  # plot(quadReg_2)
  
  # Make data frames for each sheet, for testing purposes
  switch(i,
         sheet1 <- data,
         sheet2 <- data,
         sheet3 <- data)

  # For training data (1, 2) remove rows containing outliers determined by Cook's Dist
  # then perform regression. Only need to use predict() for one set of observations
  # due to the mixture-model nature of the data
  switch(i,
         {data      <- data[c(-1,-3:-6,-19),]
          linReg_1  <- lm(y ~ -1 + ., data)
          yHat_LR1  <- predict(linReg_1, sheet1[,c(1:6)])
          quadReg_1 <- lm(y ~ -1 + . + .^2, data)
          yHat_QR1  <- predict(quadReg_1, sheet1[,c(1:6)])},
         {data      <- data[c(-1:-6,-8),]
          linReg_2  <- lm(y ~ -1 + ., data)
          yHat_LR2  <- predict(linReg_2, sheet2[,c(1:6)])
          quadReg_2 <- lm(y ~ -1 + . + .^2, data)
          yHat_QR2  <- predict(quadReg_2, sheet2[,c(1:6)])})
  
  
  # For training data results
  if(i < 3) {
    
    # Analysis of Variance Table for linear and quadratic regression
    switch(i,
           {tableLR <- anova(linReg_1)
            tableQR <- anova(quadReg_1)},
           {tableLR <- anova(linReg_2)
            tableQR <- anova(quadReg_2)})
    
    # Model sum of squares and degrees of freedom, ignoring the residual
    mssLR <- sum(tableLR[1:length(tableLR$`Sum Sq`) - 1, 2])
    mssQR <- sum(tableQR[1:length(tableQR$`Sum Sq`) - 1, 2])
    mdfLR <- round(sum(tableLR[1:length(tableLR$`Sum Sq`) - 1, 1]),1)
    mdfQR <- round(sum(tableQR[1:length(tableQR$`Sum Sq`) - 1, 1]),1)
    
    # Mean square error of model
    mmseLR <- mssLR/mdfLR
    mmseQR <- mssQR/mdfQR
    
    # Residual sum of squares and degrees of freedom
    rssLR <- tableLR[length(tableLR$`Sum Sq`),2]
    rssQR <- tableQR[length(tableQR$`Sum Sq`),2]
    rdfLR <- tableLR[length(tableLR$`Sum Sq`),1]
    rdfQR <- tableQR[length(tableLR$`Sum Sq`),1]
    
    # Mean square errors of residual
    mseLR <- rssLR/rdfLR
    mseQR <- rssQR/rdfQR
    
    # Combine and print results
    residualLR <- c(rdfLR,rssLR,mseLR,NA,NA)
    modelLR <- c(mdfLR,mssLR,mmseLR,mmseLR/mseLR,df(mmseLR/mseLR,mdfLR,rdfLR))
    resultsLR <- data.frame(rbind(modelLR,residualLR))
    names(resultsLR) <- tableNames
    
    residualQR <- c(rdfQR,rssQR,mseQR,NA,NA)
    modelQR <- c(mdfQR,mssQR,mmseQR,mmseQR/mseQR,df(mmseQR/mseQR,mdfQR,rdfQR))
    resultsQR <- data.frame(rbind(modelQR,residualQR))
    names(resultsQR) <- tableNames
    
    cat("~~~~~ Training Set #", i, " ~~~~~\n")
    print(resultsLR)
    print(resultsQR)
  }
  
  # Clean up
  rm(data)
  rm(rawData)
}

# Use training data results and compare
# testCaseList <- c("1st vs 2nd", "1st vs 3rd", "2nd vs 3rd")

# cor(yHat_LR1, sheet2$y)
# cor(yHat_LR1, sheet3$y)
# cor(yHat_QR1, sheet2$y)
# cor(yHat_QR1, sheet3$y)
# cor(yHat_LR2, sheet3$y)
# cor(yHat_QR2, sheet3$y)



