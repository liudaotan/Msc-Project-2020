from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import GBTClassifier
from datetime import datetime 
import json

spark = SparkSession.builder \
        .master("local[10]") \
        .appName("assignment2_Q1") \
        .config("spark.local.dir","/fastdata/acp20dl") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# train and test with partial data
myseed = 406
df,_ = spark.read.csv('../Data/HIGGS.csv.gz',inferSchema=True).randomSplit([0.01, 0.99], seed=myseed)
df.cache()

# The first column is the class label (1 for signal, 0 for background),
#     followed by the 28 features (21 low-level features then 7 high-level features)
df = df.withColumnRenamed('_c0', 'labels').cache() # _c0
schemaNames = df.schema.names
feature_columns = schemaNames[1:] # _c1 ~ _c28

# Use the same splits of training and test data when comparing performances among the algorithms
train, test = df.randomSplit([0.7, 0.3], seed=myseed)
train.cache()
test.cache()

# bigger data
df_bigger= spark.read.csv('../Data/HIGGS.csv.gz',inferSchema=True)
df_bigger.cache()

df_bigger = df_bigger.withColumnRenamed('_c0', 'labels').cache() # _c0

# Use the same splits of whole training and test data when comparing performances among the algorithms
train_bigger, test_bigger = df_bigger.randomSplit([0.8, 0.2], seed=myseed)
train_bigger.cache()
test_bigger.cache()

vecAssembler = VectorAssembler(inputCols = feature_columns, outputCol = 'features')

####### Random forest GridSearch ############
rf = RandomForestClassifier(labelCol="labels", featuresCol="features", seed=myseed)
stages = [vecAssembler, rf]
pipeline = Pipeline(stages=stages)

# configurations of parameters
paramGrid = ParamGridBuilder() \
    .addGrid(rf.featureSubsetStrategy, ['all','sqrt','log2']) \
    .addGrid(rf.maxDepth, [1, 10, 15]) \
    .addGrid(rf.numTrees, [1, 10, 20]) \
    .build()

evaluator = MulticlassClassificationEvaluator(labelCol="labels", 
                                              predictionCol="prediction", 
                                              metricName="accuracy")
evaluator_auc  = BinaryClassificationEvaluator(labelCol='labels',metricName="areaUnderPR")


crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

start_time = datetime.now()

cvModel = crossval.fit(train)
prediction = cvModel.transform(test)
accuracy = evaluator.evaluate(prediction)
auc = evaluator_auc.evaluate(prediction)

end_time = datetime.now()
cost_time=(end_time-start_time).seconds
print("Total time of Grid search: ",cost_time/60," minutes.")

print("Accuracy for best rf model on small data set = %g " % accuracy)
print("Auc for best rf model on small data set = %g " % auc)
bestParamsRF = cvModel.bestModel.stages[-1].extractParamMap()
paramDict1 = {param[0].name: param[1] for param in bestParamsRF.items()}
print("The best parameters:")
print(json.dumps(paramDict1, indent = 4))

####### On bigger data ############
vecTrainingData = vecAssembler.transform(train_bigger)
#vecTrainingData.select("features", "labels").show(5)

# choose the best parameters
rf = RandomForestClassifier(labelCol="labels", featuresCol="features", seed=myseed,
                            featureSubsetStrategy = paramDict1['featureSubsetStrategy'],
                            maxDepth=paramDict1['maxDepth'],numTrees=paramDict1['numTrees'])
stages = [vecAssembler, rf]
pipeline = Pipeline(stages=stages)


start_time = datetime.now()
pipelineModel = pipeline.fit(train_bigger)
end_time = datetime.now()
cost_time=(end_time-start_time).seconds
print("Fitting time of RF model on whole data set: ",cost_time/60," minutes.")

# the performance
predictions = pipelineModel.transform(test_bigger)
accuracy = evaluator.evaluate(predictions)
auc = evaluator_auc.evaluate(prediction)
print("Accuracy for best randomFroest on bigger data: %g " % accuracy)
print("Auc for best randomFroest on bigger data: %g " % auc)

# the most importent features
featureImp = pd.DataFrame(
  list(zip(vecAssembler.getInputCols(), pipelineModel.stages[-1].featureImportances)),
  columns=["feature", "importance"])
print("The top 3 important features:")
top3_important = featureImp.sort_values(by="importance", ascending=False).head(3)
print(top3_important)
print()


###############Gradient Boosting Grid search ###############
gbt = GBTClassifier(labelCol="labels",featuresCol="features",seed=myseed)
stages = [vecAssembler, gbt]
pipeline = Pipeline(stages=stages)

paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [1,10,15]) \
    .addGrid(gbt.maxIter, [10, 20, 30]) \
    .addGrid(gbt.featureSubsetStrategy,['all','sqrt','log2']) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

start_time = datetime.now()
cvModel = crossval.fit(train)
end_time = datetime.now()
cost_time=(end_time-start_time).seconds
print("Total time of Grid search: ",cost_time/60," minutes.")

prediction = cvModel.transform(test)
accuracy = evaluator.evaluate(prediction)
auc = evaluator_auc.evaluate(prediction)

print("Accuracy for best GBT model on small data set = %g " % accuracy)
print("Auc for best GBT model on small data set = %g " % auc)

bestParamsGB = cvModel.bestModel.stages[-1].extractParamMap()
paramDict2 = {param[0].name: param[1] for param in bestParamsGB.items()}
print("The best parameters:")
print(json.dumps(paramDict2, indent = 4))

######### on biggger data ############
gbt = GBTClassifier(labelCol="labels",featuresCol="features",seed=myseed,
                   featureSubsetStrategy = paramDict2['featureSubsetStrategy'],
                    maxDepth=paramDict2['maxDepth'],maxIter=paramDict2['maxIter'])
stages = [vecAssembler, gbt]
pipeline = Pipeline(stages=stages)

start_time = datetime.now()

pipelineModel = pipeline.fit(train_bigger)

end_time = datetime.now()
cost_time=(end_time-start_time).seconds
print("Fitting time of GBT model on whole data set: ",cost_time/60," minutes.")

predictions = pipelineModel.transform(test_bigger)
accuracy = evaluator.evaluate(predictions)
auc = evaluator_auc.evaluate(prediction)
print("Accuracy for best GBTClassifier on bigger data: %g " % accuracy)
print("Auc for best GBTClassifier on bigger data: %g " % auc)

# the most importent features
featureImp = pd.DataFrame(
  list(zip(vecAssembler.getInputCols(), pipelineModel.stages[-1].featureImportances)),
  columns=["feature", "importance"])
print("The top 3 important features:")
top3_important = featureImp.sort_values(by="importance", ascending=False).head(3)
print(top3_important)
print()

##################### MultilayerPerceptronClassifier ########################
layers = [len(train.columns)-1, 20, 5, 2] 
mpc = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features", layers=layers, seed=myseed)
stages = [vecAssembler, mpc]
pipeline = Pipeline(stages=stages)

paramGrid = ParamGridBuilder() \
    .addGrid(mpc.maxIter,[30,50,100]) \
    .addGrid(mpc.blockSize,[64,128,256]) \
    .addGrid(mpc.stepSize,[0.01,0.03,0.05]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

start_time = datetime.now()
cvModel = crossval.fit(train)
end_time = datetime.now()
cost_time=(end_time-start_time).seconds
print("Total time of Grid search: ",cost_time/60," minutes.")

prediction = cvModel.transform(test)
accuracy = evaluator.evaluate(prediction)
auc = evaluator_auc.evaluate(prediction)

print("Accuracy for best MPC model on small data = %g " % accuracy)
print("Auc for best MPC model on small data = %g " % auc)
bestParamsMPC = cvModel.bestModel.stages[-1].extractParamMap()
paramDict3 = {param[0].name: param[1] for param in bestParamsMPC.items()}
print("The best parameters:")
print(json.dumps(paramDict3, indent = 4))

#################### On Bigger Data ########################
mpc = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features", layers=layers, seed=myseed,
                                    maxIter = paramDict3['maxIter'] ,blockSize = paramDict3['blockSize'] ,stepSize = paramDict3['stepSize'])

stages = [vecAssembler, mpc]
pipeline = Pipeline(stages=stages)

start_time = datetime.now()
pipelineModel = pipeline.fit(train_bigger)
end_time = datetime.now()
cost_time=(end_time-start_time).seconds
print("Fitting time of MPC model on whole data set: ",cost_time/60," minutes.")

predictions = pipelineModel.transform(test_bigger)
accuracy = evaluator.evaluate(predictions)
auc = evaluator_auc.evaluate(prediction)
print("Accuracy for best MPCClassifier on bigger data: %g " % accuracy)
print("Auc for best MPCClassifier on bigger data: %g " % auc)