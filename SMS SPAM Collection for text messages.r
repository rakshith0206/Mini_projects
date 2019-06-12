## 1. installing the packages

install.packages("dplyr")
install.packages("ggplot2")
install.packages("caret")
install.packages("readr")
install.packages("stringr")
install.packages("caTools")
install.packages("tm")
install.packages("ROCR")
install.packages("SnowballC")
install.packages("pROC")
install.packages("rpart")
install.packages("randomForest")
install.packages("rpart.plot")
install.packages("NLP")
install.packages("wordcloud")
install.packages("e1071")

## 2. Loading the packages

library("dplyr","ggplot2","caret","readr","stringr","caTools","tm","ROCR","SnowballC",
        "pROC","rpart","randomForest","rpart.plot","NLIP","wordcloud","e1071")

## 3. Retrieving the data

data<-read.csv("E:\\DMA\\spam.csv")

str(data)

head(data)

## 3.1 removing the unnecessary variables from the data
data$X <- NULL
data$X.1 <- NULL
data$X.2 <- NULL

str(data)

head(data)

## 3.2 naming the variables

names(data)<- c("index","text")
str(data)

data$index <- as.factor(data$index)

## 4 visualizing the index data

ggplot(data,aes(x = index,fill=index))+geom_bar(stat = "count")


## 5 Preprocessing steps

## 5.1 balancing the data

prop.table(table(data$index))

data$index<-as.character(data$index)
str(data)

data$index[data$index=='ham']<- "1"
data$index[data$index=='spam']<- "0"
data$index<-as.factor(data$index)

str(data)
head(data)

data$index<-as.factor(data$index)

data_1<-data[data$index==1,]
data_0<-data[data$index==0,]

ids_1=sample(nrow(data_1),nrow(data_1)*0.6)
View(ids_1)
data_1_samp=data_1[ids_1,]
View(data_1_samp)

data1 <- rbind(data_1_samp,data_0) 
prop.table(table(data1$index))

## 5.2 Preprocess the text data

# 5.2.1 creating the corpus for the word in the text file

corpus <- Corpus(VectorSource(data$text))
corpus
corpus[[77]]$content

# 5.2.2 Changing the uppercase to lowercase

corpus <- tm_map(corpus, tolower)
corpus[[77]]$content

# 5.2.3 Removing the punctuations

corpus <- tm_map(corpus,removePunctuation)
corpus[[77]]$content

# 5.2.4 Removing the stopwords

corpus <- tm_map(corpus,removeWords,c(stopwords("english")))
corpus[[77]]$content

# 5.2.5 Removing the white spaces

corpus <- tm_map(corpus,stripWhitespace)
corpus[[77]]$content

# 5.2.6 Stemming the document

corpus <- tm_map(corpus, stemDocument)
corpus[[77]]$content

## 6 creating the document term matrix

dtm <- DocumentTermMatrix(corpus) 
dtm

# 6.1 Removing the sparse words

sparsewords <- removeSparseTerms(dtm,0.97)
sparsewords

SparseWords <- as.data.frame(as.matrix(sparsewords)) 
SparseWords

colnames(SparseWords) <- make.names(colnames(SparseWords))
SparseWords$index <- data$index
SparseWords$index

# 6.2 creating the word cloud

wordcloud(corpus,min.freq = 50,random.order = FALSE,colors = 'violet')

set.seed(1234)

## 7 splitting the data into train and test

split <- sample.int(n=nrow(SparseWords),size = floor(0.75*nrow(SparseWords)),replace = F) 

train <- SparseWords[split,]
test <- SparseWords[-split,]

dim(SparseWords)
dim(train)
dim(test)

## 8 Modeling the data

# 8.1 Logistic Regression

Model <- glm(index~.,data = train,family = "binomial")
Predictions <- predict(Model,newdata = test,type = "response")

confusionMatrix<-table(test$index,Predictions>0.7)
confusionMatrix

ROCR_LR <- prediction(Predictions,test$index)
Auc_LR <- as.numeric(performance(ROCR_LR,"auc")@y.values) 
print(Auc_LR*100)

Prediction_LR <- prediction(Predictions,test$index)
performance_LR <- performance(Prediction_LR,"tpr","fpr")
plot(performance_LR)


# 8.2 Decision Tree

Model_Tree <- rpart(index~.,data = train,method = "class",minbucket=50)
Prediction_tree <- predict(Model_Tree,newdata = test,type = "class")

confusionMatrix_tree <- table(test$index,Prediction_tree)
confusionMatrix_tree

prp(Model_Tree)

TreeAccuracy <- as.data.frame(table(test$index,Prediction_tree))
print(paste("Decision tree Accuracy:",round((TreeAccuracy$Freq[1]+TreeAccuracy$Freq[4])/nrow(test),4)*100,"%"))

# 8.3 Random Forest

Model_RF <- randomForest(index~.,data = train,ntree=200,nodesize=15)
Prediction_RF <- predict(Model_RF,newdata = test,type = "class")
table(test$index,Prediction_RF)

plot(Model_RF)

RFAccuracy <- as.data.frame(table(test$index,Prediction_RF))
print(paste("Random Forest Accuracy:",round((RFAccuracy$Freq[1]+RFAccuracy$Freq[4])/nrow(test),4)*100,"%"))

sensitivity(Prediction_RF,test$index)

specificity(Prediction_RF,test$index)


## 8.4 SVM

?svm

Model_SVM <- svm(index~.,data = train)
Prediction_SVM <- predict(Model_SVM,newdata = test,type = "class")
table(test$index,Prediction_SVM)

SVMAccuracy <- as.data.frame(table(test$index,Prediction_SVM))
print(paste("SVM Accuracy:",round((SVMAccuracy$Freq[1]+SVMAccuracy$Freq[4])/nrow(test),4)*100,"%"))

#### Sensitivity, specificity 

sensitivity(Prediction_SVM,test$index)

specificity(Prediction_SVM,test$index)

posPredValue(Prediction_SVM,test$index)

negPredValue(Prediction_SVM,test$index)
