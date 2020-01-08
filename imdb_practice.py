
# coding: utf-8

# In[1]:


import pandas as pd
from bs4 import BeautifulSoup
import re


# In[2]:


# load stop words from english.txt
stopwords = []
with open('english.txt','r',encoding='utf8')as f:
    lines = f.readlines()
    for line in lines:
        stopwords.append(line.replace('\n', ''))
stopwords = set(stopwords)


# In[3]:


# concat positive comments with negative ones
data_list = []
with open('imdb_train_pos.txt','r',encoding='utf8')as f:
    lines = f.readlines()
    for line in lines:
        line = re.sub("[^a-zA-Z]", " ",BeautifulSoup(line).get_text()).lower()
        data_list.append((line, 1))

with open('imdb_train_neg.txt','r',encoding='utf8')as f:
    lines = f.readlines()
    for line in lines:
        line = re.sub("[^a-zA-Z]", " ",BeautifulSoup(line).get_text()).lower()
        data_list.append((line, 0))


df = pd.DataFrame(data_list, columns=['text', 'label'])


# In[4]:


# preprocess the text
df['words'] = df['text'].apply(lambda x: [item for item in x.split() if item not in stopwords])
df['processed_text'] = df['words'].apply(lambda x: ' '.join(x))


# In[5]:


# extract features from long text
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 500) 
train_data_features = vectorizer.fit_transform(list(df['processed_text']))
train_data_features = train_data_features.toarray()
vocab = vectorizer.get_feature_names()


# In[6]:


# train random forest model
from sklearn.ensemble import RandomForestClassifier
 
forest = RandomForestClassifier(n_estimators = 100) 
forest = forest.fit(train_data_features, df["label"] )
importance = forest.feature_importances_


# In[7]:


importance_list = []
for i in range(len(vocab)):
    importance_list.append((vocab[i], importance[i], i))
a = pd.DataFrame(importance_list, columns=['word', 'importance', 'index'])
a.sort_values(by='importance', ascending=False, inplace=True)
selected_features = list(a['index'])[:25]


# In[8]:


# retrain by selected features
from sklearn.ensemble import RandomForestClassifier
 
forest = RandomForestClassifier(n_estimators = 100) 
forest = forest.fit(train_data_features[:, selected_features], df["label"] )


# In[10]:


# preprocess dev data
dev_data_list = []
with open('imdb_dev_pos.txt','r',encoding='utf8')as f:
    lines = f.readlines()
    for line in lines:
        line = re.sub("[^a-zA-Z]", " ",BeautifulSoup(line).get_text()).lower()
        dev_data_list.append((line, 1))

with open('imdb_dev_neg.txt','r',encoding='utf8')as f:
    lines = f.readlines()
    for line in lines:
        line = re.sub("[^a-zA-Z]", " ",BeautifulSoup(line).get_text()).lower()
        dev_data_list.append((line, 0))


dev_df = pd.DataFrame(dev_data_list, columns=['text', 'label'])
dev_df['words'] = dev_df['text'].apply(lambda x: [item for item in x.split() if item not in stopwords])
dev_df['processed_text'] = dev_df['words'].apply(lambda x: ' '.join(x))
dev_data_features = vectorizer.fit_transform(list(dev_df['processed_text']))
dev_data_features = dev_data_features.toarray()
dev_df['predict'] = forest.predict(dev_data_features[:, selected_features])


# In[11]:


# predict and print the performance on develep set
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
accuracy = accuracy_score(dev_df['label'], dev_df['predict'])
precision = precision_score(dev_df['label'], dev_df['predict'])
recall = recall_score(dev_df['label'], dev_df['predict'])
f_measure = f1_score(dev_df['label'], dev_df['predict'])
print(accuracy, precision, recall, f_measure)


# In[12]:


# preprocess test data
test_data_list = []
with open('imdb_test_pos.txt','r',encoding='utf8')as f:
    lines = f.readlines()
    for line in lines:
        line = re.sub("[^a-zA-Z]", " ",BeautifulSoup(line).get_text()).lower()
        test_data_list.append((line, 1))

with open('imdb_test_neg.txt','r',encoding='utf8')as f:
    lines = f.readlines()
    for line in lines:
        line = re.sub("[^a-zA-Z]", " ",BeautifulSoup(line).get_text()).lower()
        test_data_list.append((line, 0))


test_df = pd.DataFrame(test_data_list, columns=['text', 'label'])
test_df['words'] = test_df['text'].apply(lambda x: [item for item in x.split() if item not in stopwords])
test_df['processed_text'] = test_df['words'].apply(lambda x: ' '.join(x))
test_data_features = vectorizer.fit_transform(list(test_df['processed_text']))
test_data_features = test_data_features.toarray()
test_df['predict'] = forest.predict(test_data_features[:, selected_features])


# In[13]:


# predict and print the performance
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
accuracy = accuracy_score(test_df['label'], test_df['predict'])
precision = precision_score(test_df['label'], test_df['predict'])
recall = recall_score(test_df['label'], test_df['predict'])
f_measure = f1_score(test_df['label'], test_df['predict'])
print(accuracy, precision, recall, f_measure)

