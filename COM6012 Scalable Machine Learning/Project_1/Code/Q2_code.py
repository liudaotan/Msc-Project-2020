from pyspark.sql import SparkSession

spark = SparkSession.builder \
        .master("local[2]") \
        .appName("assignment1_Q2") \
        .config("spark.local.dir","/fastdata/acp20dl") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

df_ratings = spark.read.load("../Data/ml-latest/ratings.csv", format="csv", inferSchema="true", header="true")
# (df_ratings, test) = df_ratings.randomSplit([0.001, 0.999], 1234)
df_ratings.cache()

# sort the timestamp column
df_ratings_timeordered = df_ratings.sort(df_ratings.timestamp.asc()) 

from pyspark.sql.window import Window
from pyspark.sql.functions import lit,row_number,monotonically_increasing_id,col,format_number

# df: the dataframe need to be splitted; percent: the percent of train set 
def split_by_row_index(df, percent):
    df_num = df.count()
    split_index = int(df_num*percent)
    df = df.withColumn('row_id', row_number().over(Window.orderBy(monotonically_increasing_id())))
    train = df.filter(col("row_id").between(0,split_index))
    test = df.filter(col("row_id").between(split_index+1,df_num))
    return train.cache(),test.cache()

percent_50_train,percent_50_test= split_by_row_index(df_ratings_timeordered, 0.5)
percent_65_train,percent_35_test = split_by_row_index(df_ratings_timeordered, 0.65)
percent_80_train,percent_20_test = split_by_row_index(df_ratings_timeordered, 0.8)

print("The size of percent_50_train: %d; the size of percent_50_test %d"%(percent_50_train.count(),percent_50_test.count()))
print("The size of percent_65_train: %d; the size of percent_35_test %d"%(percent_65_train.count(),percent_35_test.count()))
print("The size of percent_80_train: %d; the size of percent_20_test %d"%(percent_80_train.count(),percent_20_test.count()))

# 2) For each of the three splits above, study three versions of ALS using your student number as the seed as the following [2 marks]
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

# model_name: the als model name;
def als_fit_pred_losses(model_name,train_data,test_data):
	model = model_name.fit(train_data)
	predictions = model.transform(test_data)
	rmes_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
	mse_evaluator = RegressionEvaluator(metricName="mse", labelCol="rating",predictionCol="prediction")
	mae_evaluator = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")
	rmse = rmes_evaluator.evaluate(predictions)
	mse = mse_evaluator.evaluate(predictions)
	mae = mae_evaluator.evaluate(predictions)
	return model,rmse,mse,mae

myseed = 200206596
# orginal model with different data split
als1 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
model1_50_50,model1_50_50_rmse,model1_50_50_mse,model1_50_50_mae = als_fit_pred_losses(als1,percent_50_train,percent_50_test)
model1_65_35,model1_65_35_rmse,model1_65_35_mse,model1_65_35_mae = als_fit_pred_losses(als1,percent_65_train,percent_35_test)
model1_80_20,model1_80_20_rmse,model1_80_20_mse,model1_80_20_mae = als_fit_pred_losses(als1,percent_80_train,percent_20_test)

# model2 with different data split
als2 = ALS(userCol="userId", itemCol="movieId", seed=myseed, regParam = 0.01, coldStartStrategy="drop")  
_,model2_50_50_rmse,model2_50_50_mse,model2_50_50_mae = als_fit_pred_losses(als2,percent_50_train,percent_50_test)
_,model2_65_35_rmse,model2_65_35_mse,model2_65_35_mae = als_fit_pred_losses(als2,percent_65_train,percent_35_test)
_,model2_80_20_rmse,model2_80_20_mse,model2_80_20_mae = als_fit_pred_losses(als2,percent_80_train,percent_20_test)

# model3 with different data split
als3 = ALS(userCol="userId", itemCol="movieId", seed=myseed, maxIter=15, nonnegative = True, coldStartStrategy="drop")
_,model3_50_50_rmse,model3_50_50_mse,model3_50_50_mae = als_fit_pred_losses(als3,percent_50_train,percent_50_test)
_,model3_65_35_rmse,model3_65_35_mse,model3_65_35_mae = als_fit_pred_losses(als3,percent_65_train,percent_35_test)
_,model3_80_20_rmse,model3_80_20_mse,model3_80_20_mae = als_fit_pred_losses(als3,percent_80_train,percent_20_test)

print("als1 = ALS(userCol=\"userId\", itemCol=\"movieId\", seed=myseed, coldStartStrategy=\"drop\")")
print("als2 = ALS(userCol=\"userId\", itemCol=\"movieId\", seed=myseed, regParam = 0.01, coldStartStrategy=\"drop\")")
print("als3 = ALS(userCol=\"userId\", itemCol=\"movieId\", seed=myseed, maxIter=15, nonnegative = True, coldStartStrategy=\"drop\")")

row = Row("MODEL", "SPLIT","RMSE","MSE","MAE")
c1 = ['ALS1','ALS1','ALS1','ALS2','ALS2','ALS2','ALS3','ALS3','ALS3']
c2 = ['50-50','65-35','80-20','50-50','65-35','80-20','50-50','65-35','80-20']
c3 = [model1_50_50_rmse,model1_65_35_rmse,model1_80_20_rmse,model2_50_50_rmse,model2_65_35_rmse,model2_80_20_rmse,model3_50_50_rmse,model3_65_35_rmse,model3_80_20_rmse]
c4 = [model1_50_50_mse,model1_65_35_mse,model1_80_20_mse,model2_50_50_mse,model2_65_35_mse,model2_80_20_mse,model3_50_50_mse,model3_65_35_mse,model3_80_20_mse]
c5 = [model1_50_50_mae,model1_65_35_mae,model1_80_20_mae,model2_50_50_mae,model2_65_35_mae,model2_80_20_mae,model3_50_50_mae,model3_65_35_mae,model3_80_20_mae]

# contruct a dataframe 
loss_results = sc.parallelize([row(c1[i], c2[i],c3[i], c4[i],c5[i]) for i in range(9)]).toDF()
loss_results = loss_results.withColumn("RMSE", format_number(loss_results.RMSE, 3))\
                 .withColumn("MSE", format_number(loss_results.MSE, 3))\
                 .withColumn("MAE", format_number(loss_results.MAE, 3))
loss_results.show()

# 1) After ALS, each user is modelled with some factors. For each of the three time-splits, use k-means in pyspark with k=20 to cluster the user factors learned with the ALS setting in Lab 3 but with your student number as the seed (as above in A.2 first bullet), and find the top three largest user clusters. Report the size of each cluster (number of users) in one Table, in total 3 splits x 3 clusters = 9 numbers. [2 marks]

from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors

# model: the model learned with the ALS setting in Lab 3
# return1: user-id--features--prediction(cluster id) dataframe
# return2: top 3 predictions and corresponding quantity dataframe
def top_cluster_size(model, k=20, seed=myseed):
	model_factors = model.userFactors.cache()
	kmeans = KMeans(k = k, seed = seed)
	kmeansmodel = kmeans.fit(model_factors)
	transformed = kmeansmodel.transform(model_factors)
	rank = transformed.groupBy(['prediction']).count().sort("count", ascending=False)
	return transformed.cache(),rank.limit(3).cache() 

# use models with same ALS setting(Lab 3), but fit them from 3 different split dataset
user_cluster_predictions_50_50,cluster_size_rank_50_50 = top_cluster_size(model1_50_50)
user_cluster_predictions_65_35,cluster_size_rank_65_35 = top_cluster_size(model1_65_35)
user_cluster_predictions_80_20,cluster_size_rank_80_20 = top_cluster_size(model1_80_20)

# for result dataframe construct
result1 = [row[0] for row in cluster_size_rank_50_50.select('count').distinct().collect()]
result2 = [row[0] for row in cluster_size_rank_65_35.select('count').distinct().collect()]
result3 = [row[0] for row in cluster_size_rank_80_20.select('count').distinct().collect()]

row = Row("SPLIT","Top1-size","Top2-size","Top3-size")
col1 = ['Data-split: 50-50','Data-split: 65-35','Data-split: 80-20']
col2 = [result1[0]] + [result2[0]] + [result3[0]]
col3 = [result1[1]] + [result2[1]] + [result3[1]]
col4 = [result1[2]] + [result2[2]] + [result3[2]]

cluser_size_table = sc.parallelize([row(col1[i], col2[i],col3[i],col4[i]) for i in range(3)]).toDF()
cluser_size_table.show()

# 2) Report 3 splits x 5 genres x 2 sets = 30 genres in one Table. [3 marks]

# get the largest user cluster ID form the top 3
# based on different dataset, here are three numbers.
largest_cluster_id_50_50 = cluster_size_rank_50_50.collect()[0][0]
largest_cluster_id_65_35 = cluster_size_rank_65_35.collect()[0][0]
largest_cluster_id_80_20 = cluster_size_rank_80_20.collect()[0][0]

import pyspark.sql.functions as F

# clusterID: a number, biggest cluster's id
# cluster_Pred_DF: df contains userID and clusterID
# DataSet: the 3 split and 2 sets
# return: a list with required movie id
def getMovieId(clusterID,cluster_Pred_DF,DataSet): # clusterID is the ID of the biggest cluster;cluster_Pred_DF is a cluster DF;Dataset is the splited DF like half of  50-50, 65-35, 80-20 data set.
	userIdDf = cluster_Pred_DF.where(cluster_Pred_DF.prediction == clusterID).cache() # get the userIDs of user who are cluseter in the clusterID
	userIdList = [user_id[0] for user_id in userIdDf.collect()]
	df_over4star = DataSet.filter(F.col('userId').isin(userIdList)).filter(F.col('rating') >= 4).cache()
	movieIdList = [movie_id[0] for movie_id in df_over4star.select('movieId').distinct().collect()]
	return movieIdList

train_50_movieId_list = getMovieId(largest_cluster_id_50_50,user_cluster_predictions_50_50,percent_50_train)
test_50_movieId_list = getMovieId(largest_cluster_id_50_50,user_cluster_predictions_50_50,percent_50_test)
train_65_movieId_list = getMovieId(largest_cluster_id_65_35,user_cluster_predictions_65_35,percent_65_train)
test_35_movieId_list = getMovieId(largest_cluster_id_65_35,user_cluster_predictions_65_35,percent_35_test)
train_80_movieId_list = getMovieId(largest_cluster_id_80_20,user_cluster_predictions_80_20,percent_80_train)
test_20_movieId_list = getMovieId(largest_cluster_id_80_20,user_cluster_predictions_80_20,percent_20_test)


df_movies = spark.read.load("../Data/ml-latest/movies.csv", format="csv", inferSchema="true", header="true")
df_movies.cache()

from pyspark.sql.functions import split

# movie_DF: the dataframe above
# movie_list: the movie id list
# return: the top 5 gennres
def getTopGenres(movie_DF,movie_list): # movieDF is the df read from .csv;
	movieDF_in_list = movie_DF.filter(F.col('movieId').isin(movie_list)).cache()
	split_genres_list = movieDF_in_list.select(split(movieDF_in_list.genres, "[|]", -1).alias('genres')).collect() # split_genres_list: the list of row which contains a list of genres like: Row(genres=['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy'])
	genre_dict = {}
	for row in split_genres_list:
		for genre_list in row:
			for genre in genre_list:
				if genre in genre_dict:
					genre_dict[genre] += 1
				else:
					genre_dict[genre] = 0
	result = list(dict(sorted(genre_dict.items(), key=lambda item:item[1], reverse=True)[0:5]).keys()) # get the top 5 as type of list
	return result

train50_genres = ["Train 50"] + getTopGenres(df_movies,train_50_movieId_list)
test50_genres = ["Test 50"] + getTopGenres(df_movies,test_50_movieId_list)
train65_genres = ["Train 65"] + getTopGenres(df_movies,train_65_movieId_list)
test35_genres = ["Test 35"] + getTopGenres(df_movies,test_35_movieId_list)
train80_genres = ["Train 80"] + getTopGenres(df_movies,train_80_movieId_list)
test20_genres = ["Test 20"] + getTopGenres(df_movies,test_20_movieId_list)

genres_list = []
genres_list.append(train50_genres)
genres_list.append(test50_genres)
genres_list.append(train65_genres)
genres_list.append(test35_genres)
genres_list.append(train80_genres)
genres_list.append(test20_genres)

genre_df = spark.createDataFrame(genres_list,['Data','1', '2','3', '4', '5'])
genre_df.show()