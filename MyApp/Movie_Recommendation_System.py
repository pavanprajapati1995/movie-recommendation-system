#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation System

# In[48]:


# Import all the neceesary Packages 
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import TruncatedSVD
from fuzzywuzzy import process


# ### Read Movie and Rating CSV Files using Pandas

# In[2]:


# get_my_movie_name
movie_dataFrame = pd.read_csv('movies.csv')



# In[3]:


rating_dataFrame = pd.read_csv('ratings.csv')


# ### Data Analysis

# In[4]:


### we will first check the null values present in our data




# ### Merge both the data frame and delete those columns which are not required

# In[5]:


overall_movie_rating = pd.merge(rating_dataFrame, movie_dataFrame, on = 'movieId')
overall_movie_rating.head()


# In[6]:


columns = ['timestamp', 'genres']
overall_movie_rating = overall_movie_rating.drop(columns, axis = 1)
overall_movie_rating.head()


# In[7]:


overall_movie_rating['title'].isnull().sum()
overall_ratingCount = (overall_movie_rating.groupby(by = ['title'])['rating'].count().reset_index().rename(columns = {'rating': 'totalRatingCount'}))
print(overall_ratingCount.shape)
overall_ratingCount.head()


# In[11]:


rating_with_totalRatingCount = overall_movie_rating.merge(overall_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
print(rating_with_totalRatingCount.shape)
rating_with_totalRatingCount.head(20)


# ### Remove Duplicate Records

# In[12]:


user_rating = rating_with_totalRatingCount.drop_duplicates(['userId','title'])
user_rating.head()


# ## Matrix Factorization using SVD

# In[13]:


#### Create matrix of the user_rating data frame 
movie_user_rating_pivot = user_rating.pivot(index = 'userId', columns = 'title', values = 'rating')
movie_user_rating_pivot = movie_user_rating_pivot.fillna(0)
movie_user_rating_pivot.head()


# In[14]:


## Transpose the above  matrix so that the column (movies) becomes rows(userId) and the userId comes to the column

X = movie_user_rating_pivot.T


# ### Fit the Model

# In[59]:


## Fit the model on using X data
## so we will import sckit learn


SVD = TruncatedSVD(n_components=17, random_state=17)
matrix = SVD.fit_transform(X)
print(SVD.explained_variance_ratio_.sum()*100)
matrix.shape


# ### Pearson’s R correlation

# In[60]:


import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
corr = np.corrcoef(matrix)
corr.shape


# ### Testing

# In[63]:

print('Enter the movie name')
movie_name = input()
all_movies_name = movie_user_rating_pivot.columns
movieList = list(all_movies_name)
idx = process.extractOne(movie_name, movie_dataFrame['title'])[0]
movie_index = movieList.index(idx)



# In[64]:


myPrediction = corr[movie_index]
output = list(all_movies_name[(myPrediction >= 0.9)])
for ans in output:
    print(ans)

