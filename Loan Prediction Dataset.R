#================================================
#Loan Prediction Problem III
#================================================

#In this, we need to predict the Loan_status i.e. to give loan or not based on the given variables.

#================================================
#Cleaning Memory & Decimal Places
#================================================

rm(list=ls(all=TRUE))
options(digits = 4)

#================================================
#Importing Libraries
#================================================

library(data.table)           # extension of data.frame
library(caret)                # used for classification & Regression Training
library(e1071)                # used for SVM
library(rpart)                # used for Decision Tree
library(rpart.plot)           # used for Decision Tree Plots
library(ggplot2)              # used for Plots
library(randomForest)         # used for Random Forest
library(corrplot)             # used for correlation
library(dplyr)                # used for manipulating data
library(cowplot)              # used for multiple plots

#================================================
#Importing Data & Replacing Blanks by NA
#================================================

train = fread('C:/Users/DELL/Desktop/Analytics Vidhya/Loan Prediction - III/train_ctrUa4K.csv', 
              na.strings = c("", " ", NA))

test = fread('C:/Users/DELL/Desktop/Analytics Vidhya/Loan Prediction - III/test_lAUu6dG.csv',
             na.strings = c("", " ", NA))

#================================================
#Checking Dimensions, Names, Structure, Summary & Head
#================================================

dim(train)
dim(test)

names(train)
names(test)

str(train)
str(test)

summary(train)
summary(test)

head(train)
head(test)

#================================================
# Adding 13th Column in Test Dataset
#================================================

test[,Loan_Status := NA]

#================================================
# Combine The Two Dataset
#================================================
#Concatenate the data for easy handling of missing values/null values.

data = rbind(train,test)

#================================================
#Uni-variate Analysis
#================================================

#Target Variable
ggplot(train) + geom_bar(aes(Loan_Status), fill = 'dark green')  

# 31% = N and 69% = Y, Unbalanced Data

#Independent Numeric Variable

ggplot(data) + geom_histogram(aes(ApplicantIncome), binwidth = 10000, fill = 'blue')
# Positively Skewed
boxplot(data$ApplicantIncome)

ggplot(data) + geom_histogram(aes(CoapplicantIncome), binwidth = 1000, fill = 'blue')
# Positively Skewed
boxplot(data$CoapplicantIncome)

ggplot(data) + geom_histogram(aes(LoanAmount), binwidth = 100, fill = 'blue')
# Positively Skewed
boxplot(data$LoanAmount)

#All variables are skewed ie non-normal as well as contains outliers.
#We need to normalized it. Normalizing may reduces the effects of ouliers.

ggplot(data) + geom_bar(aes(as.factor(Loan_Amount_Term)), fill = 'blue')
# Here, we need to change 6 to 60 and 350 to 360

#Independent Categorical Variable
ggplot(data %>% group_by(Gender) %>% summarise(Count = n())) +
        geom_bar(aes(Gender, Count), stat = "identity", fill = 'coral1') +
        geom_label(aes(Gender, Count, label = Count), vjust = 0.5)

ggplot(data %>% group_by(Married) %>% summarise(Count = n())) +
        geom_bar(aes(Married, Count), stat = "identity", fill = 'coral1') +
        geom_label(aes(Married, Count, label = Count), vjust = 0.5)

ggplot(data %>% group_by(Self_Employed) %>% summarise(Count = n())) +
        geom_bar(aes(Self_Employed, Count), stat = "identity", fill = 'coral1') +
        geom_label(aes(Self_Employed, Count, label = Count), vjust = 0.5)

ggplot(data %>% group_by(Credit_History) %>% summarise(Count = n())) +
        geom_bar(aes(Credit_History, Count), stat = "identity", fill = 'coral1') +
        geom_label(aes(Credit_History, Count, label = Count), vjust = 0.5)

#All the variables contains missing values which needs to be imputed.

# Independent Ordinal Variable
ggplot(data %>% group_by(Dependents) %>% summarise(Count = n())) +
        geom_bar(aes(Dependents, Count), stat = "identity", fill = 'coral1') +
        geom_label(aes(Dependents, Count, label = Count), vjust = 0.5)

#Dependents contains missing value.

ggplot(data %>% group_by(Education) %>% summarise(Count = n())) +
        geom_bar(aes(Education, Count), stat = "identity", fill = 'coral1') +
        geom_label(aes(Education, Count, label = Count), vjust = 0.5)

ggplot(data %>% group_by(Property_Area) %>% summarise(Count = n())) +
        geom_bar(aes(Property_Area, Count), stat = "identity", fill = 'coral1') +
        geom_label(aes(Property_Area, Count, label = Count), vjust = 0.5)

#================================================
#Changing Loan_Amount Term : 350 to 360 & 6 to 60
#================================================

data[, Loan_Amount_Term := ifelse(Loan_Amount_Term == 350, 360,
                                  ifelse(Loan_Amount_Term == 6, 60, Loan_Amount_Term))]

#================================================
#Bi-variate Analysis
#================================================

#Target Variable vs Independent Categorical Variable
ggplot(train, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Gender)+
        ggtitle("Loan Status by Gender of Applicant")

ggplot(train, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Married)+
        ggtitle("Loan Status by Marital Status of Applicant")

ggplot(train, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Dependents)+
        ggtitle("Loan Status by number of Dependents of Applicant")

ggplot(train, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Education)+
        ggtitle("Loan Status by Education of Applicant")

ggplot(train, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Self_Employed)+
        ggtitle("Loan Status by Employment status of Applicant")

ggplot(train, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Loan_Amount_Term)+
        ggtitle("Loan Status by terms  of loan")

ggplot(train, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Credit_History)+
        ggtitle("Loan Status by credit history of Applicant")

ggplot(train, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Property_Area)+
        ggtitle("Loan Status by property area")

#Target Variable vs Independent Numeric Variable
ggplot(train, aes(x=Loan_Status,y=ApplicantIncome))+geom_boxplot()+
        ggtitle("Loan Status by Applicant income")

ggplot(train, aes(x=Loan_Status,y=CoapplicantIncome))+geom_boxplot()+
        ggtitle("Loan Status by coapplicant income")

ggplot(train, aes(x=Loan_Status,y=LoanAmount))+geom_boxplot()+
        ggtitle("Loan Status by Loan Amount")

#================================================
#Imputing Missing Value
#================================================

#Replacing Categorical and Ordinal Missing Value by Mode
data$Gender[is.na(data$Gender)] <- "Male"
data$Married[is.na(data$Married)] <- "Yes"
data$Dependents[is.na(data$Dependents)] <- "0"
data$Self_Employed[is.na(data$Self_Employed)] <- "No"
data$Credit_History[is.na(data$Credit_History)] <- 1
data$Loan_Amount_Term[is.na(data$Loan_Amount_Term)] <- 360

#Replacing Numerical Missing Value by Mean
data$LoanAmount[is.na(data$LoanAmount)] <- mean(data$LoanAmount, na.rm = T)

#================================================
#Checking Missing Value
#================================================

colSums(is.na(data))

#================================================
#Feature Engineering - Creating New Variables
#================================================

data$Total_Income <- data$ApplicantIncome + data$CoapplicantIncome
data$EMI <- ceiling((data$LoanAmount/as.numeric(data$Loan_Amount_Term))*1000)
data$Balance_Income <- data$Total_Income - data$EMI

#================================================
#Checking Association Among Categorical Variables
#================================================

xtabs(~ Loan_Status + Gender , train)
chisq.test(train$Gender, train$Loan_Status, correct = F)
#p-value > 0.05, association present

xtabs(~ Loan_Status + Married , train)
chisq.test(train$Married, train$Loan_Status, correct = F)
#p-value < 0.05, no association

xtabs(~ Loan_Status + Dependents , train)
chisq.test(train$Dependents, train$Loan_Status, correct = F)
#p-value > 0.05, association present

xtabs(~ Loan_Status + Education , train)
chisq.test(train$Education, train$Loan_Status, correct = F)
#p-value < 0.05, no association

xtabs(~ Loan_Status + Self_Employed , train)
chisq.test(train$Self_Employed, train$Loan_Status, correct = F)
#p-value > 0.05, association present

xtabs(~ Loan_Status + Credit_History , train)
chisq.test(train$Credit_History, train$Loan_Status, correct = F)
#p-value < 0.05, no association

xtabs(~ Loan_Status + Property_Area , train)
chisq.test(train$Property_Area, train$Loan_Status, correct = F)
#p-value < 0.05, no association

#Here, Gender, Dependents and Self_Employed have an association with Loan_Status
#ie. these variables are significant in predicting Loan_Status

#================================================
#Checking Correlation Among Numeric Variable
#================================================

df = data[, c("LoanAmount", "Loan_Amount_Term", "Credit_History", 
               "Total_Income", "EMI", "Balance_Income")]
M <- cor(df)
corrplot(M, method = "number")

#Removed Applicant Income and Co-applicant Income as it is highly correlated 
#with newly created variables.

#Balance Income-Total Income, Balance Income-Loan Amount, Total Income-Loan Amount
#are highly correlated with each other.

#================================================
#Converting to Factor
#================================================

data$Credit_History = factor(data$Credit_History, levels = c(0,1), labels = c("Unmet", "Met"))
data$Loan_Amount_Term = as.factor(data$Loan_Amount_Term)

#================================================
#Transformation of variable
#================================================

data$LoanAmount = log(data$LoanAmount)
data$Total_Income = log(data$Total_Income)

#================================================
#Splitting The Data
#================================================

train = data[1:614, c(2:6,11:16)]
test = data[615:981, c(2:6,11:16)]
test = test[, Loan_Status := NULL]

#================================================
#Converting Dependent Variable To Factor
#================================================

#train[, Loan_Status := ifelse(Loan_Status == "N", 0, 1)]
train$Loan_Status = as.factor(train$Loan_Status)

#================================================
#Model Building - Binary Logistics
#================================================

reg_mod = glm(Loan_Status ~., family = 'binomial', data = train)
summary(reg_mod)

reg = step(reg_mod)
summary(reg)

#Misclassification Error for train
pred1 = ifelse(predict(reg, train , type = 'response')>0.5 , 1, 0)
tab1 = table(Predicted = pred1 , Actual = train$Loan_Status)
1 - sum(diag(tab1)/sum(tab1))   # error = 18.4%

#Predicting for test
p1 = ifelse(predict(reg_mod, test, type = 'response')>0.5, "Y", "N")

#Creating csv file of predicted values
write.csv(p1, 'C:/Users/DELL/Desktop/Analytics Vidhya/Loan Prediction - III/BinaryLogistics - Predicted.csv', 
          row.names = F)

#================================================
#Model Building - Decision Tree
#================================================

dtree <- rpart(Loan_Status ~., data = train, 
               control = rpart.control(minsplit = 15, xval = 5, maxdepth = 5))
summary(dtree)

#Prediction for train
p2 = predict(dtree, train , type = 'class')

#Evaluating for train
confusionMatrix(p2, train$Loan_Status) #error = 18.2%

#Predicting for test
pred2 = predict(dtree, test, type = 'class')

#Visualization
plot(dtree)
text(dtree)

prp(dtree)

rpart.plot(dtree)

#Creating csv file of predicted values
write.csv(pred2, 'C:/Users/DELL/Desktop/Analytics Vidhya/Loan Prediction - III/DecisionTree - Predicted.csv', 
          row.names = F)

#================================================
#Model Building - Random Forest
#================================================

set.seed(123)
rf_mod <- randomForest(Loan_Status ~., data = train, ntree = 300, importance = T)

#Prediction for train
p3 = predict(rf_mod, train , type = 'class')

#Evaluating for train
confusionMatrix(p3, train$Loan_Status) #error = 0%

#Predicting for test
pred3 = predict(rf_mod, test, type = 'class')

#Creating csv file of predicted values
write.csv(pred3, 'C:/Users/DELL/Desktop/Analytics Vidhya/Loan Prediction - III/RandomForest - Predicted.csv', 
          row.names = F)

plot(rf_mod)
varImp(rf_mod)
