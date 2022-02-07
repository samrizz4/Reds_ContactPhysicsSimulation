#Sam Rizzuto
#Cincinnati Reds
#Technical Assessment: 2.5
#6 February 2022


#load libraries
library(dplyr)
library(mgcv)
library(parallel)
library(visreg)
library(ggplot2)
library(randomForest)
library(e1071)
library(caret)
library(flexclust)
library(factoextra)
library(knitr)

#set working directory
setwd("~/Desktop/22-ds")

#load in datasets
trainDF <- read.csv("train.csv")
testDF <- read.csv("test.csv")



#backward tracking hit velo

#dist grouping each 3 vars based on pitch type from k means and TB hit from slg and 
# says this is hr launch angle pulls them randomly runs 1000 times says this is LA on tet
#first randomize tb
#randomize again
#take avg
# smallSmall <- trainDF[1:5,12:44]
# 
# store2 <- list()
# for (i in 1:nrow(smallSmall)) {
#   storeTest <- sample(smallSmall[i,30:33], 1, prob = abs(c(smallSmall[i,30], smallSmall[i,31], smallSmall[i,32], smallSmall[i,33])))
#   store2[[i]] <- storeTest
# }
# store2
#then calcluate probability of the 5 outcomes


# trainDF <- trainDF %>% mutate(swingSpeed = EXIT_SPEED - (0.2 * trainDF$RELEASE_SPEED) / (1 + 0.2))
# newDF <- trainDF %>% group_by(BATTER_UID) %>% summarise(avgSS = mean(swingSpeed),
#                                                         n = n())
# newDF %>% arrange(desc(avgSS))
# 
# hist(newDF$avgSS)
# 
# #0,2 wooden, avg 70mph blast motion
# exitVeloFormula <- (0.2 * trainDF$RELEASE_SPEED) + (1 + 0.2)*runif(nrow(trainDF), min = 40, max = 80)
# summary(exitVeloFormula)
# round(asin(trainDF$PLATE_X / exitVeloFormula*(21.922)),3)




#pitchers can predict launch angle the most
#not exit velo or spray angle

#slash line

#vaa down in zone, can predicr launch angle is gonna be low

#cant predict exit velo


#deviance explained, t value p val

#gam better for predicting constant

#same hand or diff
#low exit velo if same








######################Filtering Data and Removing Outliers through IQR
Q1_Angle <- quantile(trainDF$ANGLE, .25)
Q3_Angle <- quantile(trainDF$ANGLE, .75)
IQR_Angle <- IQR(trainDF$ANGLE)

Q1_EXIT_SPEED <- quantile(trainDF$EXIT_SPEED, .25)
Q3_EXIT_SPEED <- quantile(trainDF$EXIT_SPEED, .75)
IQR_EXIT_SPEED <- IQR(trainDF$EXIT_SPEED)

Q1_DIRECTION <- quantile(trainDF$DIRECTION, .25)
Q3_DIRECTION <- quantile(trainDF$DIRECTION, .75)
IQR_DIRECTION <- IQR(trainDF$DIRECTION)

Q1_ReleaseSpeed <- quantile(trainDF$RELEASE_SPEED, .25)
Q3_ReleaseSpeed <- quantile(trainDF$RELEASE_SPEED, .75)
IQR_ReleaseSpeed <- IQR(trainDF$RELEASE_SPEED)

Q1_PlateX <- quantile(trainDF$PLATE_X, .25)
Q3_PlateX <- quantile(trainDF$PLATE_X, .75)
IQR_PlateX <- IQR(trainDF$PLATE_X)

Q1_PlateZ <- quantile(trainDF$PLATE_Z, .25)
Q3_PlateZ <- quantile(trainDF$PLATE_Z, .75)
IQR_PlateZ <- IQR(trainDF$PLATE_Z)

Q1_InducedVertBreak <- quantile(trainDF$INDUCED_VERTICAL_BREAK, .25)
Q3_InducedVertBreak <- quantile(trainDF$INDUCED_VERTICAL_BREAK, .75)
IQR_InducedVertBreak <- IQR(trainDF$INDUCED_VERTICAL_BREAK)

Q1_HorizontalBreak <- quantile(trainDF$HORIZONTAL_BREAK, .25)
Q3_HorizontalBreak <- quantile(trainDF$HORIZONTAL_BREAK, .75)
IQR_HorizontalBreak <- IQR(trainDF$HORIZONTAL_BREAK)

Q1_VertApproachAngle <- quantile(trainDF$VERTICAL_APPROACH_ANGLE, .25)
Q3_VertApproachAngle <- quantile(trainDF$VERTICAL_APPROACH_ANGLE, .75)
IQR_VertApproachAngle <- IQR(trainDF$VERTICAL_APPROACH_ANGLE)

Q1_HorizApproachAngle <- quantile(trainDF$HORIZONTAL_APPROACH_ANGLE, .25)
Q3_HorizApproachAngle <- quantile(trainDF$HORIZONTAL_APPROACH_ANGLE, .75)
IQR_HorizApproachAngle <- IQR(trainDF$HORIZONTAL_APPROACH_ANGLE)


trainDF <- subset(trainDF, trainDF$ANGLE > (Q1_Angle - 1.5*IQR_Angle) & trainDF$ANGLE < (Q3_Angle + 1.5*IQR_Angle))
trainDF <- subset(trainDF, trainDF$EXIT_SPEED > (Q1_EXIT_SPEED - 1.5*IQR_EXIT_SPEED) & trainDF$EXIT_SPEED < (Q3_EXIT_SPEED + 1.5*IQR_EXIT_SPEED))
trainDF <- subset(trainDF, trainDF$DIRECTION > (Q1_DIRECTION - 1.5*IQR_DIRECTION) & trainDF$DIRECTION < (Q3_DIRECTION + 1.5*IQR_DIRECTION))

trainDF <- subset(trainDF, trainDF$RELEASE_SPEED > (Q1_ReleaseSpeed - 1.5*IQR_ReleaseSpeed) & trainDF$RELEASE_SPEED < (Q3_ReleaseSpeed + 1.5*IQR_ReleaseSpeed))
trainDF <- subset(trainDF, trainDF$PLATE_X > (Q1_PlateX - 1.5*IQR_PlateX) & trainDF$PLATE_X < (Q3_PlateX + 1.5*IQR_PlateX))
trainDF <- subset(trainDF, trainDF$PLATE_Z > (Q1_PlateZ - 1.5*IQR_PlateZ) & trainDF$PLATE_Z < (Q3_PlateZ + 1.5*IQR_PlateZ))
trainDF <- subset(trainDF, trainDF$INDUCED_VERTICAL_BREAK > (Q1_InducedVertBreak - 1.5*IQR_InducedVertBreak) & trainDF$INDUCED_VERTICAL_BREAK < (Q3_InducedVertBreak + 1.5*IQR_InducedVertBreak))
trainDF <- subset(trainDF, trainDF$HORIZONTAL_BREAK > (Q1_HorizontalBreak - 1.5*IQR_HorizontalBreak) & trainDF$HORIZONTAL_BREAK < (Q3_HorizontalBreak + 1.5*IQR_HorizontalBreak))
trainDF <- subset(trainDF, trainDF$VERTICAL_APPROACH_ANGLE > (Q1_VertApproachAngle - 1.5*IQR_VertApproachAngle) & trainDF$VERTICAL_APPROACH_ANGLE < (Q3_VertApproachAngle + 1.5*IQR_VertApproachAngle))
trainDF <- subset(trainDF, trainDF$HORIZONTAL_APPROACH_ANGLE > (Q1_HorizApproachAngle - 1.5*IQR_HorizApproachAngle) & trainDF$HORIZONTAL_APPROACH_ANGLE < (Q3_HorizApproachAngle + 1.5*IQR_HorizApproachAngle))

#disregard strikeouts, hbp, walks
trainDF <- trainDF %>% filter(PITCH_RESULT_KEY == "InPlay")
#goes from original 26417 rows to 24510 after cleaning

summary(trainDF)
######################

trainDF <- trainDF %>% mutate(OPS = OBP + SLG)
testDF <- testDF %>% mutate(OPS = OBP + SLG)
trainDF <- trainDF %>% mutate(Handedness = if_else(THROWS_LEFT == BATS_LEFT, 0, 1))
testDF <- testDF %>% mutate(Handedness = if_else(THROWS_LEFT == BATS_LEFT, 0, 1))

######################Random Forest model
#running random forest on 3 prediction variables
rfTrainExitVelo <- randomForest(EXIT_SPEED ~ RELEASE_SPEED + PLATE_X + PLATE_Z + 
                                  INDUCED_VERTICAL_BREAK + HORIZONTAL_BREAK +
                                  VERTICAL_APPROACH_ANGLE + HORIZONTAL_APPROACH_ANGLE + 
                                  OPS + Handedness, data = trainDF)
rfTrainAngle <- randomForest(ANGLE ~ RELEASE_SPEED + PLATE_X + PLATE_Z + 
                               INDUCED_VERTICAL_BREAK + HORIZONTAL_BREAK +
                               VERTICAL_APPROACH_ANGLE + HORIZONTAL_APPROACH_ANGLE + 
                               OPS + Handedness, data = trainDF)
rfTrainDirection <- randomForest(DIRECTION ~ RELEASE_SPEED + PLATE_X + PLATE_Z + 
                                   INDUCED_VERTICAL_BREAK + HORIZONTAL_BREAK +
                                   VERTICAL_APPROACH_ANGLE + HORIZONTAL_APPROACH_ANGLE + 
                                   OPS + Handedness, data = trainDF)
#viewing importance plots of each model type to determine most significant vars in model
varImpPlot(rfTrainExitVelo) #drop handedness
varImpPlot(rfTrainAngle) #drop ops, handedness
varImpPlot(rfTrainDirection) #drop handedness

#choose first grouping vars
rfUpdatedExitVelo <- randomForest(EXIT_SPEED ~ PLATE_X + PLATE_Z + HORIZONTAL_APPROACH_ANGLE +
                                    HORIZONTAL_BREAK + RELEASE_SPEED +
                                    VERTICAL_APPROACH_ANGLE + INDUCED_VERTICAL_BREAK + OPS, data = trainDF)
importance(rfUpdatedExitVelo) #importance vars of rf exit velo model
plot(rfUpdatedExitVelo) #error of rf exit velo model

rfUpdatedAngle <- randomForest(ANGLE ~ PLATE_Z + INDUCED_VERTICAL_BREAK + RELEASE_SPEED + VERTICAL_APPROACH_ANGLE +
                                PLATE_X + HORIZONTAL_BREAK + HORIZONTAL_APPROACH_ANGLE , data = trainDF)

rfUpdatedDirection <- randomForest(DIRECTION ~  PLATE_X + HORIZONTAL_APPROACH_ANGLE + RELEASE_SPEED + 
                                     HORIZONTAL_BREAK + VERTICAL_APPROACH_ANGLE +
                                     INDUCED_VERTICAL_BREAK + PLATE_Z + OPS, data = trainDF)



#plot randomForest results back in trainDF
trainDF <- trainDF %>% 
  mutate(exitVelo_RF = round(predict(rfUpdatedExitVelo, newdata = .),3))
trainDF <- trainDF %>% 
  mutate(angle_RF = round(predict(rfUpdatedAngle, newdata = .),3))
trainDF <- trainDF %>% 
  mutate(direction_RF = round(predict(rfUpdatedDirection, newdata = .),3))



######################SVM model
#running svm on exit velo
svmExitVelo <- svm(EXIT_SPEED ~ RELEASE_SPEED + PLATE_X + PLATE_Z + 
                     INDUCED_VERTICAL_BREAK + HORIZONTAL_BREAK +
                     VERTICAL_APPROACH_ANGLE + HORIZONTAL_APPROACH_ANGLE + 
                     OPS + Handedness, data = trainDF, cost = 100, gamma = 1)
#removing predictor variable of exit velo
svmExitVelo_Pred <- round(predict(svmExitVelo, trainDF[,-19]), 3)
#add svm exit velo into training df
trainDF <- trainDF %>% mutate(exitVelo_SVM = svmExitVelo_Pred)



#running svm on angle
svmAngle <- svm(ANGLE ~ RELEASE_SPEED + PLATE_X + PLATE_Z + 
                  INDUCED_VERTICAL_BREAK + HORIZONTAL_BREAK +
                  VERTICAL_APPROACH_ANGLE + HORIZONTAL_APPROACH_ANGLE + 
                  OPS + Handedness, data = trainDF, cost = 100, gamma = 1)
#removing predictor variable of angle
svmAngle_Pred <- round(predict(svmAngle, trainDF[,-20]), 3)
#adding prob of angle into training df
trainDF <- trainDF %>% mutate(angle_SVM = svmAngle_Pred)



#running svm model on direction
svmDirection <- svm(DIRECTION ~ RELEASE_SPEED + PLATE_X + PLATE_Z + 
                      INDUCED_VERTICAL_BREAK + HORIZONTAL_BREAK +
                      VERTICAL_APPROACH_ANGLE + HORIZONTAL_APPROACH_ANGLE + 
                      OPS + Handedness, data = trainDF, cost = 100, gamma = 1)
#removing predictor variable of direction
svmDirection_Pred <- round(predict(svmDirection, trainDF[,-21]), 3)
#adding direction into training df
trainDF <- trainDF %>% mutate(direction_SVM = svmDirection_Pred)





######################generalized additive model to include most important variables in training dataset
options(mc.cores = parallel::detectCores())#run model in parallel

gam_EV <- bam(EXIT_SPEED ~ RELEASE_SPEED + PLATE_X + PLATE_Z + INDUCED_VERTICAL_BREAK +
              HORIZONTAL_BREAK + VERTICAL_APPROACH_ANGLE + HORIZONTAL_APPROACH_ANGLE + OPS + Handedness, 
            data = trainDF, family = gaussian, method = "GCV.Cp")
summary(gam_EV) #drop platex, platez, vert approach angle
gam_EV_Upd <- bam(EXIT_SPEED ~ RELEASE_SPEED + INDUCED_VERTICAL_BREAK +
                HORIZONTAL_BREAK + HORIZONTAL_APPROACH_ANGLE + OPS + Handedness, 
              data = trainDF, family = gaussian, method = "GCV.Cp")

gam_Ang <- bam(ANGLE ~ RELEASE_SPEED + PLATE_X + PLATE_Z + INDUCED_VERTICAL_BREAK + 
              HORIZONTAL_BREAK + VERTICAL_APPROACH_ANGLE + HORIZONTAL_APPROACH_ANGLE + OPS + Handedness, 
            data = trainDF, family = gaussian, method = "GCV.Cp")
summary(gam_Ang) #no drop
gam_Ang_Upd <- gam_Ang

gam_Dir <- bam(DIRECTION ~ RELEASE_SPEED + PLATE_X + PLATE_Z + INDUCED_VERTICAL_BREAK +
              HORIZONTAL_BREAK + VERTICAL_APPROACH_ANGLE + HORIZONTAL_APPROACH_ANGLE + OPS + Handedness, 
            data = trainDF, family = gaussian, method = "GCV.Cp")
summary(gam_Dir) #drop platez
gam_Dir_Upd <- bam(DIRECTION ~ RELEASE_SPEED + PLATE_X + INDUCED_VERTICAL_BREAK +
                 HORIZONTAL_BREAK + VERTICAL_APPROACH_ANGLE + HORIZONTAL_APPROACH_ANGLE + OPS + Handedness, 
               data = trainDF, family = gaussian, method = "GCV.Cp")


gamEV <- gam_EV_Upd$coefficients
gamAng <- gam_Ang_Upd$coefficients
gamDir <- gam_Dir_Upd$coefficients

trainDF <- trainDF %>% mutate(exitVelo_GAM = round(gamEV[1] + gamEV[2]*RELEASE_SPEED + gamEV[3]*INDUCED_VERTICAL_BREAK +
                                                     gamEV[4]*HORIZONTAL_BREAK + gamEV[5]*HORIZONTAL_APPROACH_ANGLE +
                                                     gamEV[6]*OPS + gamEV[7]*Handedness, 3))

trainDF <- trainDF %>% mutate(angle_GAM = round(gamAng[1] + gamAng[2]*RELEASE_SPEED + gamAng[3]*PLATE_X +
                                                     gamAng[4]*PLATE_Z + gamAng[5]*INDUCED_VERTICAL_BREAK +
                                                     gamAng[6]*HORIZONTAL_BREAK + gamAng[7]*VERTICAL_APPROACH_ANGLE +
                                                     gamAng[8]*HORIZONTAL_APPROACH_ANGLE + 
                                                     gamAng[9]*OPS + gamAng[10]*Handedness, 3))

trainDF <- trainDF %>% mutate(direction_GAM = round(gamDir[1] + gamDir[2]*RELEASE_SPEED + gamDir[3]*PLATE_X +
                                                     gamDir[4]*INDUCED_VERTICAL_BREAK +
                                                     gamDir[5]*HORIZONTAL_BREAK + gamDir[6]*VERTICAL_APPROACH_ANGLE +
                                                     gamDir[7]*HORIZONTAL_APPROACH_ANGLE + 
                                                     gamDir[8]*OPS + gamDir[9]*Handedness, 3))



###smaller gam

gam_Small <- bam(EXIT_SPEED ~ s(PLATE_X, PLATE_Z), 
            data = trainDF, family = gaussian, method = "GCV.Cp")

xs <- matrix(data=seq(from=-2, to=2, length=50), nrow=50, ncol=50)    
ys <- t(matrix(data=seq(from=0,to=5, length=50), nrow=50, ncol=50))

gamSmallFit <- data.frame(PLATE_X = as.vector(xs), PLATE_Z = as.vector(ys))
exitVeloPred <- predict(gam_Small, gamSmallFit, types = "response")
exitVeloPred <- matrix(exitVeloPred, nrow = 50, ncol = 50)
density(exitVeloPred)
#width of HP is 23in #according to baseball prospectus
#height is 25.79
summary(trainDF$EXIT_SPEED)
summary(trainDF$PLATE_X) #-1.5 to 1.5
summary(trainDF$PLATE_Z) #0.8 to 3.9
filled.contour(x=seq(from=-2, to=2, length=50), y=seq(from=0, to=5, length=50), z = exitVeloPred, zlim=c(60,110), 
               color.palette = colorRampPalette(c("lightblue","yellow","orange", "red", "darkred")),
               plot.axes = { rect(-0.71, 1.52, 0.71, 3.67, border="black", lty="dashed", lwd=3)
                 axis(1, at=c(-2,-1,0,1,2), pos=0, labels=c(-2,-1,0,1,2), las=0, col="black")
                 axis(2, at=c(0,1,2,3,4,5), pos=-2, labels=c(0,1,2,3,4,5), las=0, col="black")
                 }, main = "Heat Map for Exit Velo Based on X/Y Location of Strikezone", ylab = "Vertical (ft)", xlab = "Horizontal Location of Pitch at Plate (ft)")

gam_Small2 <- bam(ANGLE ~ s(PLATE_X, PLATE_Z), 
                 data = trainDF, family = gaussian, method = "GCV.Cp")
gamSmallFit2 <- data.frame(PLATE_X = as.vector(xs), PLATE_Z = as.vector(ys))
anglePred <- predict(gam_Small2, gamSmallFit2, types = "response")
anglePred <- matrix(anglePred, nrow = 50, ncol = 50)
summary(trainDF$ANGLE)
filled.contour(x=seq(from=-2, to=2, length=50), y=seq(from=0, to=5, length=50), z = anglePred, zlim=c(-15,35), 
               color.palette = colorRampPalette(c("lightblue","yellow","orange", "red", "darkred")),
               plot.axes = { rect(-0.71, 1.52, 0.71, 3.67, border="black", lty="dashed", lwd=3)
                 axis(1, at=c(-2,-1,0,1,2), pos=0, labels=c(-2,-1,0,1,2), las=0, col="black")
                 axis(2, at=c(0,1,2,3,4,5), pos=-2, labels=c(0,1,2,3,4,5), las=0, col="black")
               }, main = "Heat Map for Launch Angle Based on X/Y Location of Strikezone", ylab = "Vertical (ft)", xlab = "Horizontal Location of Pitch at Plate (ft)")


gam_Small3 <- bam(DIRECTION ~ s(PLATE_X, PLATE_Z), 
                  data = trainDF, family = gaussian, method = "GCV.Cp")
gamSmallFit3 <- data.frame(PLATE_X = as.vector(xs), PLATE_Z = as.vector(ys))
directionPred <- predict(gam_Small3, gamSmallFit3, types = "response")
directionPred <- matrix(directionPred, nrow = 50, ncol = 50)
summary(trainDF$DIRECTION)
#not taking into account handedness (l/r are combined here)
filled.contour(x=seq(from=-2, to=2, length=50), y=seq(from=0, to=5, length=50), z = directionPred, zlim=c(-25,25), 
               color.palette = colorRampPalette(c("lightblue","yellow","orange", "red", "darkred")),
               plot.axes = { rect(-0.71, 1.52, 0.71, 3.67, border="black", lty="dashed", lwd=3)
                 axis(1, at=c(-2,-1,0,1,2), pos=0, labels=c(-2,-1,0,1,2), las=0, col="black")
                 axis(2, at=c(0,1,2,3,4,5), pos=-2, labels=c(0,1,2,3,4,5), las=0, col="black")
               }, main = "Heat Map for Direction Based on X/Y Location of Strikezone", ylab = "Vertical (ft)", xlab = "Horizontal Location of Pitch at Plate (ft)")







exitVeloPredConf <- predict(gam_Small, gamSmallFit, types = "response", se = TRUE)
predVals <- data.frame(gamSmallFit, exitVeloPredConf) %>% 
  mutate(lower = exitVeloPredConf$fit - 1.96*exitVeloPredConf$se.fit,
         upper = exitVeloPredConf$fit + 1.96*exitVeloPredConf$se.fit)

ggplot(aes(x=PLATE_X,y=exitVeloPredConf$fit), data=predVals) +
  geom_ribbon(aes(ymin = lower, ymax=upper), fill='gray90') +
  geom_line(color='#00aaff') + ylab("Exit Velocity Prediction") + 
  ggtitle("Prediction of Exit Velocity Based on Horizontal Location of Pitch")
ggplot(aes(x=PLATE_Z,y=exitVeloPredConf$fit), data=predVals) +
  geom_ribbon(aes(ymin = lower, ymax=upper), fill='gray90') +
  geom_line(color='#00aaff') + ylab("Exit Velocity Prediction") + 
  ggtitle("Prediction of Exit Velocity Based on Vertical Location of Pitch")


# ggsave("name.png")

vis.gam(gam_Small, type='response', plot.type='contour')
visreg2d(gam_Small, xvar='PLATE_X', yvar='PLATE_Z', scale='response')

visreg2d(gam_EV_Upd, xvar='RELEASE_SPEED', yvar='OPS', scale='response')

anova(gam_EV_Upd, gam_Small, test="Chisq")
anova(gam_EV_Upd)
anova(gam_Ang_Upd)
anova(gam_Dir_Upd)




######################GLM model
############exit velo
exitVeloCalc <- glm(EXIT_SPEED ~ RELEASE_SPEED + PLATE_X + PLATE_Z + INDUCED_VERTICAL_BREAK + HORIZONTAL_BREAK +
                      VERTICAL_APPROACH_ANGLE + HORIZONTAL_APPROACH_ANGLE + OPS + Handedness, 
                  data = trainDF, 
                  family = gaussian)
summary(exitVeloCalc) 
exitVeloCalc$coefficients #coefficients used to determine exit velo
anova(exitVeloCalc, test = "Chisq")

############angle
angleCalc <- glm(ANGLE ~ RELEASE_SPEED + PLATE_X + PLATE_Z + INDUCED_VERTICAL_BREAK + HORIZONTAL_BREAK +
                      VERTICAL_APPROACH_ANGLE + HORIZONTAL_APPROACH_ANGLE + OPS + Handedness, 
                    data = trainDF, 
                    family = gaussian)
anova(angleCalc, test = "Chisq")


############direction
directionCalc <- glm(DIRECTION ~ RELEASE_SPEED + PLATE_X + PLATE_Z + INDUCED_VERTICAL_BREAK + HORIZONTAL_BREAK +
                      VERTICAL_APPROACH_ANGLE + HORIZONTAL_APPROACH_ANGLE + OPS + Handedness, 
                    data = trainDF, 
                    family = gaussian)
anova(directionCalc, test = "Chisq")

evc <- exitVeloCalc$coefficients
ac <- angleCalc$coefficients
dc <- directionCalc$coefficients


trainDF <- trainDF %>%
  mutate(exitVelo_GLM = round(evc[1] + evc[2]*RELEASE_SPEED + evc[3]*PLATE_X + evc[4]*PLATE_Z +
                                evc[5]*INDUCED_VERTICAL_BREAK + evc[6]*HORIZONTAL_BREAK +
                                evc[7]*VERTICAL_APPROACH_ANGLE + 
                                evc[8]*HORIZONTAL_APPROACH_ANGLE + evc[9]*OPS + evc[10]*Handedness, 3))


trainDF <- trainDF %>%
  mutate(angle_GLM = round(ac[1] + ac[2]*RELEASE_SPEED + ac[3]*PLATE_X + ac[4]*PLATE_Z +
                             ac[5]*INDUCED_VERTICAL_BREAK + ac[6]*HORIZONTAL_BREAK +
                             ac[7]*VERTICAL_APPROACH_ANGLE + 
                             ac[8]*HORIZONTAL_APPROACH_ANGLE + ac[9]*OPS + ac[10]*Handedness, 3))

trainDF <- trainDF %>%
  mutate(direction_GLM = round(dc[1] + dc[2]*RELEASE_SPEED + dc[3]*PLATE_X + dc[4]*PLATE_Z +
                                 dc[5]*INDUCED_VERTICAL_BREAK + dc[6]*HORIZONTAL_BREAK +
                                 dc[7]*VERTICAL_APPROACH_ANGLE + 
                                 dc[8]*HORIZONTAL_APPROACH_ANGLE + dc[9]*OPS + dc[10]*Handedness, 3))




###################################################Kmeans

subsetTrain <- trainDF[,c(19, 12:18, 29:30)]
km1 <- kmeans(subsetTrain[,2:10], 3, iter.max = 100)

ggplot(trainDF, aes(x = EXIT_SPEED, y = RELEASE_SPEED, color = factor(km1$cluster))) + geom_point() +
  ggtitle("Release Speed vs Exit Velocity Grouped By Clusters")


km2 <- kmeans(scale(trainDF[, c(12:18, 29:30)]), 3, nstart = 25, iter.max = 100)
fviz_cluster(km2, data = trainDF[, c(12:18, 29:30)])


# Dimension reduction PCA
pca <- prcomp(trainDF[, c(12:18, 29:30)],  scale = TRUE)
cords <- as.data.frame(get_pca_ind(pca)$coord)
cords$cluster1 <- factor(km3$cluster)
cords$EXIT_SPEED <- trainDF$EXIT_SPEED
head(cords) #see first few rows dimensions and clusters


###############KNN
#######knn on exit velo
trainKNN_x_EV <- trainDF[,c(12:18,29:30)]
trainKNN_y_EV <- trainDF[,19]
knnModel_EV <- knnreg(trainKNN_x_EV, trainKNN_y_EV)
testKNN_x_EV <- testDF[,c(12:20)]

knnpred_y_EV <- predict(knnModel_EV, data.frame(testKNN_x_EV))

knnpred_xTconf_EV <- predict(knnModel_EV, data.frame(trainKNN_x_EV), interval = "confidence", level = 0.9)
ggplot(trainDF, aes(EXIT_SPEED, knnpred_xTconf_EV)) + geom_smooth() +
  ggtitle("Error Band on KNN Model Predicting Exit Velocity") + ylab("Exit Velo Prediction")
  

mse <- mean((trainKNN_y_EV - knnpred_xTconf_EV)^2)
mae <- MAE(trainKNN_y_EV, knnpred_xTconf_EV)
rmse <- RMSE(trainKNN_y_EV, knnpred_xTconf_EV)
mse
mae
rmse



#####knn on angle
trainKNN_x_ang <- trainDF[,c(12:18,29:30)]
trainKNN_y_ang <- trainDF[,20]
knnModel_ang <- knnreg(trainKNN_x_ang, trainKNN_y_ang)
testKNN_x_ang <- testDF[,c(12:20)]

knnpred_y_ang <- predict(knnModel_ang, data.frame(testKNN_x_ang))

knnpred_xTconf_ang <- predict(knnModel_ang, data.frame(trainKNN_x_ang), interval = "confidence", level = 0.9)
ggplot(trainDF, aes(ANGLE, knnpred_xTconf_ang)) + geom_smooth() +
  ggtitle("Error Band on KNN Model Predicting Launch Angle") + ylab("Angle Prediction")

#####knn on direction
trainKNN_x_dir <- trainDF[,c(12:18,29:30)]
trainKNN_y_dir <- trainDF[,21]
knnModel_dir <- knnreg(trainKNN_x_dir, trainKNN_y_dir)
testKNN_x_dir <- testDF[,c(12:20)]

knnpred_y_dir <- predict(knnModel_dir, data.frame(testKNN_x_dir))

knnpred_xTconf_dir <- predict(knnModel_dir, data.frame(trainKNN_x_dir), interval = "confidence", level = 0.9)
ggplot(trainDF, aes(DIRECTION, knnpred_xTconf_dir)) + geom_smooth() +
  ggtitle("Error Band on KNN Model Predicting Direction") + ylab("Direction Prediction")


trainDF <- trainDF %>% mutate(exitVelo_KNN = round(knnpred_xTconf_EV,3))
trainDF <- trainDF %>% mutate(angle_KNN = round(knnpred_xTconf_ang,3))
trainDF <- trainDF %>% mutate(direction_KNN = round(knnpred_xTconf_dir,3))



#######################finding absolute errors on training df models
mean(abs(trainDF$EXIT_SPEED - trainDF$exitVelo_RF))
mean(abs(trainDF$EXIT_SPEED - trainDF$exitVelo_SVM))
mean(abs(trainDF$EXIT_SPEED - trainDF$exitVelo_GAM))
mean(abs(trainDF$EXIT_SPEED - trainDF$exitVelo_GLM))
mean(abs(trainDF$EXIT_SPEED - trainDF$exitVelo_KNN))

mean(abs(trainDF$ANGLE - trainDF$angle_RF))
mean(abs(trainDF$ANGLE - trainDF$angle_SVM))
mean(abs(trainDF$ANGLE - trainDF$angle_GAM))
mean(abs(trainDF$ANGLE - trainDF$angle_GLM))
mean(abs(trainDF$ANGLE - trainDF$angle_KNN))

mean(abs(trainDF$DIRECTION - trainDF$direction_RF))
mean(abs(trainDF$DIRECTION - trainDF$direction_SVM))
mean(abs(trainDF$DIRECTION - trainDF$direction_GAM))
mean(abs(trainDF$DIRECTION - trainDF$direction_GLM))
mean(abs(trainDF$DIRECTION - trainDF$direction_KNN))

######################











##################Testing Data
####gam
testDF <- testDF %>% mutate(exitVelo_GAM = round(gamEV[1] + gamEV[2]*RELEASE_SPEED + gamEV[3]*INDUCED_VERTICAL_BREAK +
                                                     gamEV[4]*HORIZONTAL_BREAK + gamEV[5]*HORIZONTAL_APPROACH_ANGLE +
                                                     gamEV[6]*OPS + gamEV[7]*Handedness, 3))

testDF <- testDF %>% mutate(angle_GAM = round(gamAng[1] + gamAng[2]*RELEASE_SPEED + gamAng[3]*PLATE_X +
                                                  gamAng[4]*PLATE_Z + gamAng[5]*INDUCED_VERTICAL_BREAK +
                                                  gamAng[6]*HORIZONTAL_BREAK + gamAng[7]*VERTICAL_APPROACH_ANGLE +
                                                  gamAng[8]*HORIZONTAL_APPROACH_ANGLE + 
                                                  gamAng[9]*OPS + gamAng[10]*Handedness, 3))

testDF <- testDF %>% mutate(direction_GAM = round(gamDir[1] + gamDir[2]*RELEASE_SPEED + gamDir[3]*PLATE_X +
                                                      gamDir[4]*INDUCED_VERTICAL_BREAK +
                                                      gamDir[5]*HORIZONTAL_BREAK + gamDir[6]*VERTICAL_APPROACH_ANGLE +
                                                      gamDir[7]*HORIZONTAL_APPROACH_ANGLE + 
                                                      gamDir[8]*OPS + gamDir[9]*Handedness, 3))

####glm
testDF <- testDF %>% mutate(exitVelo_GLM = round(evc[1] + evc[2]*RELEASE_SPEED + evc[3]*PLATE_X + evc[4]*PLATE_Z +
                                evc[5]*INDUCED_VERTICAL_BREAK + evc[6]*HORIZONTAL_BREAK +
                                evc[7]*VERTICAL_APPROACH_ANGLE + 
                                evc[8]*HORIZONTAL_APPROACH_ANGLE + evc[9]*OPS + evc[10]*Handedness, 3))


testDF <- testDF %>% mutate(angle_GLM = round(ac[1] + ac[2]*RELEASE_SPEED + ac[3]*PLATE_X + ac[4]*PLATE_Z +
                             ac[5]*INDUCED_VERTICAL_BREAK + ac[6]*HORIZONTAL_BREAK +
                             ac[7]*VERTICAL_APPROACH_ANGLE + 
                             ac[8]*HORIZONTAL_APPROACH_ANGLE + ac[9]*OPS + ac[10]*Handedness, 3))

testDF <- testDF %>% mutate(direction_GLM = round(dc[1] + dc[2]*RELEASE_SPEED + dc[3]*PLATE_X + dc[4]*PLATE_Z +
                                 dc[5]*INDUCED_VERTICAL_BREAK + dc[6]*HORIZONTAL_BREAK +
                                 dc[7]*VERTICAL_APPROACH_ANGLE + 
                                 dc[8]*HORIZONTAL_APPROACH_ANGLE + dc[9]*OPS + dc[10]*Handedness, 3))

###knn
testDF <- testDF %>% mutate(exitVelo_KNN = round(knnpred_y_EV,3))
testDF <- testDF %>% mutate(angle_KNN = round(knnpred_y_ang,3))
testDF <- testDF %>% mutate(direction_KNN = round(knnpred_y_dir,3))




write.csv(trainDF, "myTrainDF.csv")
write.csv(testDF, "myTestDF.csv")


