#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis of President Trump's Tweets

# ### The following work is divided into two phases: the first is the analysis of a dataset of all Donald Trump's tweets; the second is mainly concerned with visualisation. For the analysis, I tried to perform a sentimental analysis using the NRC Emotional Lexicon, a list of English words in which each term is associated with an emotion. The intention is to understand how many tweets contain a feeling of anger and to see if there is a readable trend. 

# ### We start exploring the NRC Emotional Lexicon dataset with Pandas.

# In[2]:


#Import the library
import pandas as pd
get_ipython().system('pip install session-info')


# In[4]:


filepath = "/Users/macssd/Desktop/NRCEMOTION.txt"
emolex_df = pd.read_csv(filepath,  names=["word", "emotion", "association"], skiprows=45, sep='\t', keep_default_na=False)
emolex_df.head(12)


# In[5]:


#check whether only 8 emotions are counted and which ones.
emolex_df.emotion.unique()


# In[6]:


#how many terms are associated with each emotion
emolex_df.emotion.value_counts()


# In[7]:


emolex_df[emolex_df.association == 1].emotion.value_counts()


# In[8]:


#Words associated with the feeling of anger
emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'anger')].word


# In[9]:


emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()
emolex_words.head()


# In[10]:


emolex_words[emolex_words.anger == 1].head()


# In[11]:


emolex_words[(emolex_words.joy == 1) & (emolex_words.negative == 1)].head()


# In[12]:


# Angry words
emolex_words[emolex_words.anger == 1].word


# ### We import my two datasets, which contain all of Trump's tweets from 2009 to the day the account was disabled. (Source: Github repository: https://github.com/MarkHershey/CompleteTrumpTweetsArchive)
# 

# In[13]:


#We import them and join them
df1=pd.read_csv("/Users/macssd/Desktop/Beforeoffice.csv", on_bad_lines='skip')
df2=pd.read_csv("/Users/macssd/Desktop/Inoffice.csv",on_bad_lines='skip')
frames = [df1, df2]
tweets_df= pd.concat(frames)
tweets_df


# In[14]:


##I check the size of each one to see how many tweets were written during his presidential term.
print(df1.shape) #before election
print(df2.shape) #after election


# In[15]:


#I rename the columns
tweets_df.columns = ['id', 'time', 'url', 'content']


# In[16]:


#I clean the dataset
tweets_df.content = tweets_df.content.str.replace("[^A-Za-z ]", " ")


# In[17]:


tweets_df


# ### Using Machine Learning techniques, I begin to vectorize only the words that are within the NRC from Trump's tweets

# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer

# I only want you to look for words in the emotional lexicon
# because we don't know what's up with the other words
vec = TfidfVectorizer(vocabulary=emolex_words.word,
                      use_idf=False, 
                      norm='l1') # ELL - ONE
matrix = vec.fit_transform(tweets_df.content)
vocab = vec.get_feature_names_out()
wordcount_df = pd.DataFrame(matrix.toarray(), columns=vocab)
wordcount_df.head()


# In[19]:


# Get your list of angry words
angry_words = emolex_words[emolex_words.anger == 1]['word']
angry_words.head()


# In[20]:


# Only give me the columns of angry words
tweets_df['anger'] = wordcount_df[angry_words].sum(axis=1)
tweets_df.head(3)


# In[21]:


tweets_df = tweets_df.drop(columns=['id', 'url'])
tweets_df.head()


# In[22]:


tweets_df.info()


# In[23]:


tweets_df['time']= pd.to_datetime(tweets_df['time'])
tweets_df.info()


# In[24]:


#I create a column with just the year
import datetime
tweets_df['year'] = tweets_df['time'].dt.year


# In[25]:


#tweets with angry connotation
newtweets=tweets_df[tweets_df.anger > 0]
newtweets


# In[26]:


#I create a Pivot Table in which I count the occurrences of tweets distributed over the years.

import numpy as np
newtweets.pivot_table(index=["year"],values=["anger"],aggfunc=np.count_nonzero)


# In[27]:


#I store it in the variable "example"
example=newtweets.pivot_table(index=["year"],values=["anger"],aggfunc=np.count_nonzero)
example


# In[28]:


#I display the graph to see their distribution over the years.

import matplotlib.pyplot as plt
plt.plot(example['anger'])
plt.title('A campaign of wrath')
plt.xlabel('years')
plt.ylabel('number of tweets with an angry connotation')
plt.show()


# ### What emerges from a first analysis is that tweets with angry connotations are mainly distributed over two periods: 2012 to 2016 and 2019 to 2020. During his presidential term, the tones are more moderate, but they escalate as he approaches his second presidential run. Therefore, I try to select only the tweets going from 2017.

# In[29]:


example1=newtweets[newtweets.year>2016]


# In[30]:


#number of tweets with angry connotation during the time as President
example1.shape


# In[31]:


tweets=example1.pivot_table(index=["year"],values=["anger"],aggfunc=np.count_nonzero)
tweets


# In[32]:


plt.plot(tweets['anger'])
plt.title('A campaign of wrath')
plt.xlabel('years')
plt.ylabel('number of tweets with an angry connotation')
plt.show()


# In[33]:


#Insert an additional column in which I also distribute the occurrences for each month.
example1['month_year'] = example1['time'].dt.to_period('M')


# In[34]:


example1.head()


# In[35]:


#Definitive dataset
tweetsdef=example1.pivot_table(index=["month_year"],values=["anger"],aggfunc=np.count_nonzero)


# In[103]:


tweetsdef


# In[104]:


#I save it
tweetsdef.to_csv("/Users/macssd/Desktop/DAV_21110444/tweets.csv")


# ### Topic models with Gensim. I do a further analysis to see what the topics and themes of Trump's tweets are, and assess whether anything interesting emerges.

# In[36]:


speeches_df= pd.concat(frames)


# In[37]:


speeches_df


# In[38]:


speeches_df.columns = ['id', 'time', 'url', 'content']


# In[39]:


#I clean my dataset

import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
speeches_df.content = speeches_df.content.str.replace("[^A-Za-z ]", " ")
stop = stopwords.words('english')
speeches_df['content1'] = speeches_df['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[40]:


speeches_df


# In[41]:


# I delete tweets that only contain references to other links
speeches_df=speeches_df[speeches_df[ 'content1' ].str.contains( 'http | run | com |pic' )==False ]


# In[42]:


speeches_df


# In[43]:


from gensim.utils import simple_preprocess

texts = speeches_df.content1.apply(simple_preprocess)


# In[44]:


from gensim import corpora

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=5, no_above=0.5)

corpus = [dictionary.doc2bow(text) for text in texts]


# In[45]:


from gensim import models

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]


# In[46]:


n_topics = 15

# Build an LSI model
lsi_model = models.LsiModel(corpus_tfidf,
                            id2word=dictionary,
                            num_topics=n_topics)


# In[67]:


lsi_model.print_topics()


# In[47]:


n_words = 10

topic_words = pd.DataFrame({})

for i, topic in enumerate(lsi_model.get_topics()):
    top_feature_ids = topic.argsort()[-n_words:][::-1]
    feature_values = topic[top_feature_ids]
    words = [dictionary[id] for id in top_feature_ids]
    topic_df = pd.DataFrame({'value': feature_values, 'word': words, 'topic': i})
    topic_words = pd.concat([topic_words, topic_df], ignore_index=True)

topic_words.head()


# In[48]:


#using seaborn in matplotlib to display topic distribution
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.FacetGrid(topic_words, col="topic", col_wrap=3, sharey=False)
g.map(plt.barh, "word", "value")


# In[49]:


from gensim.utils import simple_preprocess

texts = speeches_df.content1.apply(simple_preprocess)


# In[50]:


from gensim import corpora

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=2000)
corpus = [dictionary.doc2bow(text) for text in texts]


# In[51]:


from gensim import models

n_topics = 15

lda_model = models.LdaModel(corpus=corpus, num_topics=n_topics)


# In[73]:


lda_model.print_topics()


# In[74]:


import pyLDAvis
import pyLDAvis.gensim_models

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
vis


# ## Nothing particularly interesting was found. I therefore proceed to make the final changes to the dataset I downloaded on Google Sheets (tweets.xlsl https://docs.google.com/spreadsheets/d/1ikvdzE-uYmI1HcBG8BWR2naoZ5ClwKhvMGFe1nT2t3k/edit?usp=sharing) and complete the display on Datawrapper(https://www.datawrapper.de/_/wRD2W/)

# In[52]:


import session_info
session_info.show()


# In[ ]:




