# Natural Language Processing

# Importing the dataset
dataset = read.delim('TSP Dataset.tsv', quote = '', stringsAsFactors = FALSE)
                      
# Cleaning the text to make it ready for ML algorithms
library(NLP)
library(SnowballC)
library(tm)
library(RColorBrewer)
library(wordcloud)
corpus = VCorpus(VectorSource(dataset$Comments))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating Word Cloud
dtm <- TermDocumentMatrix(corpus)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 25)

# Code for Word Cloud
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 5,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))
# Frequent Words Used
findFreqTerms(dtm, lowfreq = 4)

# Finding associated words
findAssocs(dtm, terms = "smoke", corlimit = 0.3)

barplot(d[1:25,]$freq, min.freq = 10, las = 3, names.arg = d[1:25,]$word,
        col ="blue", main ="Most frequent words used by patient particiapting in Tobacco Quit Program(Facebook)",
        ylab = "Word frequencies", ylim = c(0, 100))

# Creating the Bag of words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.99)
dataset = as.data.frame(as.matrix(dtm))
dataset$Response = dataset_original$Response

# Applying Classification (Random Forest Algorithm)

# Encoding the target feature as factor
dataset$Response = factor(dataset$Response, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Response, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-269],
                          y = training_set$Response,
                          ntree = 500)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-269])

# Making the Confusion Matrix
cm = table(test_set[, 269], y_pred)

# 5 Correct Predictions for Positive comment
# 23 Correct Predictions for Negative comments
# 9 In correct predictions for Negative comments
# 6 In correct Predictions for positive comments
# Accuracy
# (5+23)/43
# 0.6511628
# Precision
# 5/(5+9)
# 0.3571429
# Recall
# 5/(5+6)
# 0.4545455

# F1 Score
# 2*0.357*0.454/(0.357+0.454)
# 0.3996991

#precision is "how useful the search results are", and recall is "how complete the results are".
#high precision means that an algorithm returned substantially more relevant results than irrelevant ones, while high recall means that an algorithm returned most of the relevant results.