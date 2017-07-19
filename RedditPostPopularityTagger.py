
# coding: utf-8

# # Reddit Post Popularity Tagger

# *This assignment attempts to use text classification on reddit posts to see how accurate the classifier is in determining whether a given post on the subreddit "Today I Learned (TIL)" is popular or not (with popular meaning the post has more than 10 comments).*

# **Data:** existing dataset of 10,000 reddit posts in the subreddit "Today I Learned"
# 
# **Document:** each post
# 
# **Two classes of data:** Popular vs. Not Popular (a popular post has more than 10 comments)
# 
# **Tokens/features of each document:** 50 most frequently occurring words across all posts (posts are then classified according to whether these common words appear in the document)
# 
# **Eliminating tokens:** eliminate stopwords, punctuation, low frequency tokens, and some part-of-speech

# ### 1. Fetch data

# In[1]:

import nltk
import pandas as pd


# In[2]:

# import Reddit dataset to Pandas dataframe
reddit = pd.read_csv("data/reddit_til_2500.csv", encoding="utf-8", dtype="unicode")
reddit.drop(labels = ['Column'], axis=1, inplace=True)
reddit


# ### 2. Split data into classes & process text

# In[3]:

# a function for tagging each post with a label for whether
# it is popular or not

def popular_tag(comments):
    # define a popular post as a post with more than 10 comments
    if int(comments.strip()) >= 10:
        return 'popular'
    else:
        return 'not popular'


# In[4]:

# get a list of labeled reddit posts

labeled_posts = []
for row in reddit.itertuples():
    # tokenize each document
    labeled_posts += [(row[2], popular_tag(row[3]))]


# In[5]:

# a function for further processing each post by
# filtering words according to part of speech

def POS_processor(text):
    features = []
    tokens = nltk.word_tokenize(text)
    pos_tagged_tokens = nltk.pos_tag(tokens) # POS tagging
    for (token, pos_tag) in pos_tagged_tokens:
        # keep only words that are tagged as adjective, verb, or noun
        if (pos_tag.startswith("N") or pos_tag.startswith("V") or pos_tag.startswith("J")):
            features.append(token)
    return features


# In[6]:

# grab all posts and store them in a list
all_posts = []
for row in reddit.itertuples():
    all_posts.append(row[2])

# further process posts by removing words by
# performing part of speech tagging
all_words = []
for post in all_posts:
    all_words.append(POS_processor(post))


# ### 3. Create tokens

# In[7]:

# create a list of the most common words across all posts

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.append("til") #remove TIL

temp = []
all_words_processed = [item for post in all_words for item in post]

# filter out stopwords and non-alpha words
words_freq = nltk.FreqDist(word.lower() 
                          for word in all_words_processed
                          if word.isalpha() and word.lower() not in stop_words)

# consists of 50 most common words
word_features = [word for (word, feature) in words_freq.most_common(50)]

text = nltk.Text(word_features)
fdist = text.vocab()
# print(fdist)
# words_freq.most_common(100)


# ### 4. Featurize posts & create labeled documents

# In[8]:

# create one feature per word in each post, with a binary value, 
# indicating whether the document contains the word or not

def post_features(post):
    features = dict()
    for word in word_features:
        features[word] = (word.lower() in post)
    return features # boolean for whether word appears in document or not


# In[9]:

# generate a list of documents that contain a 
# dictionary of features for each unique post
labeled_documents = [(post_features(post), category) for (post, category) in labeled_posts]


# In[10]:

# randomize the post order
import random
random.shuffle(labeled_documents)


# In[11]:

labeled_documents # a list of all the featurized posts


# ### 5. Train classifier to classify data & evaluate accuracy

# In[12]:

# divide the list of labeled posts into training set and test set
# where the former will be used to train the classifier
train_set, test_set = labeled_documents[:-500], labeled_documents[-500:]

# build Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# classify test set
cnt = 1
for row in test_set:
    print("test_post", cnt, ":", classifier.classify(row[0]))
    cnt += 1


# In[13]:

# get accuracy of classifier
accuracy = nltk.classify.accuracy(classifier, test_set)
print(accuracy)


# In[14]:

# conduct the classification process for multiple trials,
# shuffling the training and test data every time
train_set, test_set = [], []
trials = 50
psum = 0;
cnt = 0;
for i in range(trials):
    random.shuffle(labeled_documents)
    #keep 500 examples for testing and the remaining for training
    train_set, test_set = labeled_documents[:-500], labeled_documents[-500:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    accuracy = nltk.classify.accuracy(classifier, test_set)
    print("Trial:", cnt, " Accuracy:", accuracy)
    psum += accuracy
    cnt += 1
    
print("Avg Accuracy: ", (psum/cnt))


# In[15]:

# get confusion matrix
# shows the number of times that a classifier 
# classifies a specific instance into a particular class

ans = [category for (features, category) in test_set]
guess = [classifier.classify(features) for (features, category) in test_set]

cm = nltk.ConfusionMatrix(ans, guess)

print(cm.pretty_format(sort_by_count=True, show_percents=False))
print(cm.pretty_format(sort_by_count=True, show_percents=True))


# ### 6. Create a wordcloud & visualize most important features

# In[16]:

# get the most important features for determining post popularity
classifier.show_most_informative_features(50)


# In[17]:

# now get important features for each class separately
import math
f  = classifier._feature_probdist
mif = classifier.most_informative_features(50)
pos_features = []
neg_features = []
for (w,t) in mif:
    if t != True:
        continue
    p = f[("popular", w)]
    n = f[("not popular", w)]
    l = p.logprob(t) - n.logprob(t)
    s = l/abs(l)
    word = w.split("/")[0]
    
    # print w, math.exp(abs(l)), s
    if s>0:
        pos_features.append(word)
    else:
        neg_features.append(word)

set(neg_features) & set(pos_features)


# In[18]:

# create a wordcloud that visualizes the most important features
# of each class of posts (popular vs not popular)

from wordcloud import WordCloud
popular_wordcloud = WordCloud().generate(" ".join(pos_features))
notpopular_wordcloud = WordCloud().generate(" ".join(neg_features))


# In[19]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 15)


# In[20]:

plt.figure()
# two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(15,10))
ax1.imshow(popular_wordcloud)
ax1.axis("off")
ax1.figsize=(15,10)
ax2.imshow(notpopular_wordcloud)
ax2.axis("off")

plt.axis("off")
plt.show()


# In[ ]:



