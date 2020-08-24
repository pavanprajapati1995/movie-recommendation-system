from django.shortcuts import render
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import TruncatedSVD
from fuzzywuzzy import process
from tmdbv3api import Movie
from tmdbv3api import TMDb
import re
import os


tmdb = TMDb()
tmdb.api_key = "39713e8796c347e6f6b0be42da4ae724" #Insert your own API key
def main_code(movieName):
    movie_dataFrame = pd.read_csv(r'C:\Users\PAVAN\Desktop\python project\python_project\movie_recommendation-master\MyApp\movies.csv')
    rating_dataFrame = pd.read_csv(r'C:\Users\PAVAN\Desktop\python project\python_project\movie_recommendation-master\MyApp\ratings.csv')
    overall_movie_rating = pd.merge(rating_dataFrame, movie_dataFrame, on = 'movieId')
    columns = ['timestamp', 'genres']
    overall_movie_rating = overall_movie_rating.drop(columns, axis = 1)
    overall_ratingCount = (overall_movie_rating.groupby(by = ['title'])['rating'].count().reset_index().rename(columns = {'rating': 'totalRatingCount'}))
    rating_with_totalRatingCount = overall_movie_rating.merge(overall_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
    user_rating = rating_with_totalRatingCount.drop_duplicates(['userId','title'])
    movie_user_rating_pivot = user_rating.pivot(index = 'userId', columns = 'title', values = 'rating')
    movie_user_rating_pivot = movie_user_rating_pivot.fillna(0)
    X = movie_user_rating_pivot.T
    SVD = TruncatedSVD(n_components=17, random_state=17)
    matrix = SVD.fit_transform(X)
    corr = np.corrcoef(matrix)
    movie_name = movieName 
    all_movies_name = movie_user_rating_pivot.columns
    movieList = list(all_movies_name)
    idx = process.extractOne(movie_name, movie_dataFrame['title'])[0]
    movie_index = movieList.index(idx)
    myPrediction = corr[movie_index]
    output = list(all_movies_name[(myPrediction >= 0.9)])
    return output

def index(request):
    return render(request, 'index.html')




def findMovies(request):  
    baseURL = "https://image.tmdb.org/t/p/w300/"
    movieName = request.GET['search field']
    recommended_movies = main_code(movieName)
    movie = Movie()
    movie_poster = []
    new_movie_list = []
    for title in recommended_movies:
        m = re.match(r'^(.*) \((19\d\d|20\d\d)\)$', title)
        name, year = m.groups()
        new_movie_list.append(name)

    for i in new_movie_list:
        print(i)
    count = 0
    flag = True
    for i in new_movie_list:
        if count <= 6:
            try:
                search = movie.search(i)
                item = search[0].poster_path
            except IndexError: 
                return render(request, "error.html")
           

            movie_poster_path = str(item)
            final = baseURL + movie_poster_path
            movie_poster.append(final)
            print(final)
            count += 1
        else:
            break    

      
    return render(request, "output.html", {'movie_poster' : movie_poster})
