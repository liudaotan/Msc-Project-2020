from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from pyspark.sql.types import StringType,DoubleType,IntegerType
from pyspark.sql.functions import col
from pyspark.sql.functions import avg
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, explode, array, lit
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier,LogisticRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from datetime import datetime 

spark = SparkSession.builder \
        .master("local[10]") \
        .appName("assignment2") \
        .config("spark.local.dir","/fastdata/acp20dl") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 


myseed = 406

#df,_ = spark.read.csv('../Data/train_set.csv',header = True,inferSchema=True).randomSplit([0.0001, 0.9999], seed=myseed)
df = spark.read.csv('../Data/train_set.csv',header = True,inferSchema=True)
df.cache()

# deal the missing data
# replace ？as NaN
df = df.replace('?', None).cache()
# ID、Calendar_Year、Blind_Make、Blind_Submodel,etc are not related to the target value
df = df.drop('Row_ID','Household_ID','Blind_Submodel','Calendar_Year','Blind_Make')

# calculate the proportion of NaN
col_names = df.columns
row_num = df.count()
percent_of_null_list = np.array([])
for column in col_names:
    num_none = df.filter(df[column].isNull()).count()
    percent_of_null = num_none/row_num
    if percent_of_null != 0:
        print('The percent of Null in',column,':',percent_of_null*100 ,"%")

# drop Cat2 Cat4 Cat5 Cat7
df = df.drop('Cat2','Cat4','Cat5','Cat7')
# drop all na row
df = df.na.drop(how='any')

# find the str and int col
str_col = [x.name for x in df.schema.fields if x.dataType == StringType()]
int_col = [x.name for x in df.schema.fields if x.dataType == IntegerType()]

str2index_col = []
for col in str_col:
    str2index_col.append(col+"_num")

# the converted str col
from pyspark.ml.feature import StringIndexer
for col in str_col:
    indexer = StringIndexer(inputCol=col, outputCol=col+"_num")
    df = indexer.fit(df).transform(df)

numeric_col = [x.name for x in df.schema.fields if x.dataType == DoubleType()]
numeric_col.remove('Claim_Amount') # the list of col without the label col: Claim_Amount
df = df.select(int_col + numeric_col+['Claim_Amount']) # now the df is a DF with all cols are numerical
train_data,test_data = df.randomSplit([0.7,0.3])
cols_name = df.columns

major_train = train_data.filter(train_data.Claim_Amount == 0)
minor_train = train_data.filter(train_data.Claim_Amount != 0)
ratio = int(major_train.count()/minor_train.count())

# duplicate the minority rows
oversampled_train = minor_train.withColumn("dummy", explode(array([lit(x) for x in range(ratio)]))).drop('dummy')
# combine both oversampled minority rows and previous majority rows 
balanced_train = major_train.unionAll(oversampled_train)

# result of oversampling
claim_num = balanced_train.where(balanced_train.Claim_Amount != 0).count()
unclaim_num = balanced_train.where(balanced_train.Claim_Amount == 0).count()
print("The claim/unclaim: ",claim_num/unclaim_num)

#from datetime import datetime 
start_time = datetime.now()

vecAssembler = VectorAssembler(inputCols = cols_name[:-1],outputCol="features")
# Create a Linear Regression Model object
lr = LinearRegression(labelCol='Claim_Amount')
stages = [vecAssembler,lr]
pipeline = Pipeline(stages=stages)

# Fit the model to the data and call this model pipelineModel
pipelineModel = pipeline.fit(balanced_train)
test_results = pipelineModel.transform(test_data)
train_results = pipelineModel.transform(balanced_train)

MAE_evaluator = RegressionEvaluator(labelCol="Claim_Amount", predictionCol="prediction", metricName="mae")
MSE_evaluator = RegressionEvaluator(labelCol="Claim_Amount", predictionCol="prediction", metricName="mse")

print("MAE on test data: {}".format(MAE_evaluator.evaluate(test_results)))
print("MSE on test data: {}".format(MSE_evaluator.evaluate(test_results)))

end_time = datetime.now()
cost_time=(end_time-start_time).seconds
print("Total time of LinearRegression: ",cost_time/60," minutes.")


start_time = datetime.now()
# to modified the train set label to 0 1 for binary training
binary_train = balanced_train.withColumn('isClaim',(balanced_train['Claim_Amount']!=0).cast('int')).cache()

# first model
Lr = LogisticRegression(labelCol="isClaim", featuresCol="features",maxIter=50,regParam=0.01,tol=1,family="binomial")
stages = [vecAssembler, Lr]
pipeline_1 = Pipeline(stages=stages)
pipelineModel_1 = pipeline_1.fit(binary_train)

# non zero train set for the training of second model
non_zero_binary_train = balanced_train.where(balanced_train.Claim_Amount != 0)

# second model 
GLR = GeneralizedLinearRegression(featuresCol='features',labelCol='Claim_Amount',
                                  family="gamma", link="identity", maxIter=50, regParam=0.01)
stages = [vecAssembler, GLR]
pipeline_2 = Pipeline(stages=stages)
pipelineModel_2 = pipeline_2.fit(non_zero_binary_train)

# the prediction(0-1) from first model 
model1_result = pipelineModel_1.transform(test_data)

# get features from the data that were predicted will claim money  
model2_data = model1_result.where(model1_result.prediction != 0).select(cols_name)
# using the result of first model to get the prediction from model2
model2_result = pipelineModel_2.transform(model2_data)

# the model 1 result's col is different from model2's
# select the common cols
selected_model1_result = model1_result.where(model1_result.prediction == 0).select('Claim_Amount','prediction')
selected_model2_result = model2_result.select('Claim_Amount','prediction')

# concate two result
final_result = selected_model1_result.unionAll(selected_model2_result)

print("MAE on test data: {}".format(MAE_evaluator.evaluate(final_result)))
print("MSE on test data: {}".format(MSE_evaluator.evaluate(final_result)))

end_time = datetime.now()
cost_time=(end_time-start_time).seconds
print("Total time of combination of two models: ",cost_time/60," minutes.")