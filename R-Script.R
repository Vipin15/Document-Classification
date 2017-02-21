# Initiate
libs <- c('tm','caret','SnowballC','class','e1071','dplyr','randomForest')
lapply(libs, require,character.only=T)
rm(libs)


# Set seed for reproducible results
set.seed(100)

# Read the raw file
raw_doc <- read.table(url('https://raw.githubusercontent.com/Loktra/Data-Scientist/master/trainingdata.txt'),sep="\t", 
             fill=FALSE, 
             strip.white=TRUE,
             header = T,
             stringsAsFactors = F)


# remove 1st col & rearrange to the desired format; character & target seprated
colnames(raw_doc) <- 'text'
raw_doc$target <- substr(raw_doc$text, 1, 1)
raw_doc$text <- substring(raw_doc$text,3)


# create corpus (large and structured set of texts)
corpus <- Corpus(VectorSource(raw_doc$text))

# Clean corpus
clean_corpus <- function(corpus){
  corpus.temp <- tm_map(corpus, removeNumbers)
  corpus.temp <- tm_map(corpus.temp, removeWords, stopwords("english"))
  corpus.temp <- tm_map(corpus.temp, removePunctuation)
  corpus.temp <- tm_map(corpus.temp, stripWhitespace)
  corpus.temp <- tm_map(corpus.temp,content_transformer(tolower))
  corpus.temp <- tm_map(corpus.temp, stemDocument, language = "english")
  return(corpus.temp)
}

corpus <- clean_corpus(corpus)


# Create dtm
dtm <- DocumentTermMatrix(corpus) %>% removeSparseTerms(sparse=0.95)


# Transform dtm to matrix to data frame - df is easier to work with
mat.df <- as.data.frame(data.matrix(dtm), stringsAsfactors = FALSE)

# Column bind category (known classification)
mat.df <- cbind(mat.df, raw_doc$target)

# Change name of new column to "category"
colnames(mat.df)[ncol(mat.df)] <- "category"

set.seed(300)

#Spliting data as training and test set. Using createDataPartition() function from caret

indxTrain <- createDataPartition(y = mat.df$category,p = 0.8,list = FALSE)
training <- mat.df[indxTrain,]
testing <- mat.df[-indxTrain,]

#Checking distibution in origanl data and partitioned data
round(prop.table(table(training$category)) * 100,1)
round(prop.table(table(testing$category)) * 100,1)
round(prop.table(table(mat.df$category)) * 100,1)



set.seed(400)
control <- trainControl(method = "repeatedcv",
                        number = 3,
                        repeats = 1,
                        search = "grid")

tunegrid <- expand.grid(.mtry = c(1:25))

# Random forrest
rfFit <- train(category ~ .,
               data = training,
               method = "rf",
               metric = 'Accuracy',
               tuneGrid = tunegrid,
               trControl = control,
               preProcess = c("center", "scale"))


plot(rfFit)

rfPredict <- predict(rfFit,newdata = testing[,-ncol(testing)] )
confusionMatrix(rfPredict, testing$category )

mean(rfPredict == testing$category)


#Selecting random forest algo.

# SELECTING THE TEST_SET YOU WANT TO PREDICT
input <- read.table(file.choose(),sep="\t", 
                    fill=FALSE, 
                    strip.white=TRUE,
                    header = T,
                    stringsAsFactors = F)

colnames(input) <- 'category'

corpus.t <- Corpus(VectorSource(input$category))

# Clean corpus
corpus <- clean_corpus(corpus.t)


# Create dtm
dtm.t <- DocumentTermMatrix(corpus.t)


# Transform dtm to matrix to data frame - df is easier to work with
mat.df.t <- as.data.frame(data.matrix(dtm.t), stringsAsfactors = FALSE)

mat.df.t <- mat.df.t[names(mat.df.t) %in% names(mat.df)]

temp <- names(mat.df)[!(names(mat.df) %in% names(mat.df.t))]

missing_vars <- matrix(0, nrow = nrow(mat.df.t), ncol = length(temp))
missing_vars <- as.data.frame(missing_vars)

names(missing_vars) <- temp

unseen_set <- bind_cols(mat.df.t,missing_vars)

#Output
Output <- predict(rfFit,newdata = unseen_set)

Output





-----------------------------------------------------------------
# Trying KNN, but it gave lower accuracy
set.seed(400)
ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 4) 
knnFit <- train(category ~ .,
                data = training,
                method = "knn",
                trControl = ctrl,
                preProcess = c("center", "scale"),
                tuneLength = 7)

#Output of kNN fit
knnFit

#Plotting yields Number of Neighbours Vs accuracy (based on repeated cross validation)
plot(knnFit)


knnPredict <- predict(knnFit,newdata = testing[,-ncol(testing)] )
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knnPredict, testing$category )

mean(knnPredict == testing$category)
