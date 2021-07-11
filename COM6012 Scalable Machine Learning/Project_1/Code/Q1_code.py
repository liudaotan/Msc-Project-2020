from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder \
        .master("local[4]") \
        .appName("assignment1") \
        .config("spark.local.dir","/fastdata/acp20dl") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 

logFile = spark.read.text("../Data/NASA_access_log_Jul95.gz").cache()  # add it to cache, so it can be used in the following steps efficiently      
# logFile.show(20, False)

#  1) all hosts from Japanese universities ending with “.ac.jp”
japan_university = logFile.filter(logFile.value.contains(".ac.jp"))
# japan_university.show(5,False)
num_jp_university = japan_university.count()

#2) all hosts from UK universities ending with “.ac.uk”
uk_university = logFile.filter(logFile.value.contains(".ac.uk"))
# uk_university.show(5,False)
num_uk_university = uk_university.count()

#3) all hosts from US universities ending with “.edu”. 
us_university = logFile.filter(logFile.value.contains(".edu"))
# us_university.show(5,False)
num_us_university = us_university.count()

# Report these three numbers and visualise them using one bar graph. [3 marks]
print("\n\nThere are %i hosts from universities in Japan .\n\n" % (num_jp_university))
print("\n\nThere are %i hosts from universities in UK.\n\n" % (num_uk_university))
print("\n\nThere are %i hosts from universities in US.\n\n" % (num_us_university))

# plot results
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20,12))

universities_loc = ["JP","UK","US"]
request_num_of_jp_uk_us = [num_jp_university,num_uk_university,num_us_university]
rects = ax.bar(universities_loc, request_num_of_jp_uk_us, label = "request number")

ax.set_ylabel('request numbers')
ax.set_title("Hosts from universities in Japan,UK,and US")
ax.yaxis.set_data_interval(min(request_num_of_jp_uk_us), max(request_num_of_jp_uk_us),True)
for rect in rects:
    height = rect.get_height()
    ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.savefig("../Output/request_num_of_jp_uk_us.png")

# B 1) For each of the three countries in Question A (Japan, UK, and US), 
# 		find the top 9 most frequent universities according to the host domain.

# split into 5 columns using regex and split
# another way to regex: '\.(.*\.ac\.jp)\.*'
japan_data = japan_university.withColumn('host', F.regexp_extract('value', '([^\.]{1,50}\.ac\.jp)\.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()

uk_data = uk_university.withColumn('host', F.regexp_extract('value', '([^\.]{1,50}\.ac\.uk)\.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()

us_data = us_university.withColumn('host', F.regexp_extract('value', '([^\.]{1,50}\.edu)\.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()

# top 9 most frequent universities

jp_host_count = japan_data.select('host').groupBy('host').count().sort('count', ascending=False)
uk_host_count = uk_data.select("host").groupBy('host').count().sort('count', ascending=False)
us_host_count = us_data.select("host").groupBy('host').count().sort('count', ascending=False)


# show top 9 of each country
print("========================================")
print("Top 9 of Japan")
jp_host_count.show(9,False) 
print("========================================")
print("Top 9 of UK")
uk_host_count.show(9,False)
print("========================================")
print("Top 9 of US") 
us_host_count.show(9,False) 

#2) For each country, produce a pie chart visualising the percentage (with respect to the total) 
#	of requests by each of the top 9 most frequent universities and the rest

# save the university and request number into list "jp_host_top9"

# japan
list_jp_top9 = jp_host_count.head(9) 

jp_labels = []
jp_requsets_list = []

for item in list_jp_top9:
	jp_labels.append(item[0])
	jp_requsets_list.append(item[1])

jp_labels.append('rest')
jp_requsets_list.append(num_jp_university - sum(jp_requsets_list))

fig, ax = plt.subplots(figsize=(20,12))
jp_pie = ax.pie( jp_requsets_list,labels = jp_labels,autopct = '%1.1f%%',
	explode=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.02])
plt.legend(loc = 'lower left')
plt.title("The top 9 requests of univeristies in Japan")
plt.savefig("../Output/jp_pie_chart.png")

# UK
list_uk_top9 = uk_host_count.head(9) 

uk_labels = []
uk_requsets_list = []

for item in list_uk_top9:
	uk_labels.append(item[0])
	uk_requsets_list.append(item[1])
uk_labels.append('rest')
uk_requsets_list.append(num_uk_university - sum(uk_requsets_list))

fig, ax = plt.subplots(figsize=(20,12))
uk_pie = ax.pie(uk_requsets_list,labels = uk_labels,autopct = '%1.1f%%',
	explode=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.02])
plt.legend(loc = 'lower left')
plt.title("The top 9 requests of univeristies in UK")
plt.savefig("../Output/uk_pie_chart.png")

# US
list_us_top9 = us_host_count.head(9) 

us_labels = []
us_requsets_list = []

for item in list_us_top9:
	us_labels.append(item[0])
	us_requsets_list.append(item[1])
us_labels.append('rest')
us_requsets_list.append(num_us_university - sum(us_requsets_list))

fig, ax = plt.subplots(figsize=(20,12))
us_pie = ax.pie(us_requsets_list,labels = us_labels,autopct = '%1.1f%%',
	explode=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.02])
plt.legend(loc = 'lower left')
plt.title("The top 9 requests of univeristies in US")
plt.savefig("../Output/us_pie_chart.png")

plt.close("all")

# C. For the most frequent university from each of the three countries, 
#   	produce a plot with day as the x-axis, and the hour of visit as the y-axis (0 to 23). 
#	    Three x-y plots need to be produced with the day and hour clearly labelled. [3 marks]

#Top1
#top1 
top1_jp = japan_data.where("host='tohoku.ac.jp'")
top1_uk = uk_data.where("host='hensa.ac.uk'")
top1_us = us_data.where("host='tamu.edu'")

top1_jp_timestamp = top1_jp.select("host","timestamp") 
top1_uk_timestamp = top1_uk.select("host","timestamp")
top1_us_timestamp = top1_us.select("host","timestamp")

top1_jp_date_hour = top1_jp_timestamp.withColumn('Date', F.regexp_extract('timestamp', '(([0-2][1-9])|10|20|30|31)(?=\/Jul\/1995)', 1)).withColumn('Hour', F.regexp_extract('timestamp', '(?<=\/Jul\/1995:)(([0-1][0-9])|20|21|22|23)(?=:)', 1)).cache()
top1_uk_date_hour = top1_uk_timestamp.withColumn('Date', F.regexp_extract('timestamp', '(([0-2][1-9])|10|20|30|31)(?=\/Jul\/1995)', 1)).withColumn('Hour', F.regexp_extract('timestamp', '(?<=\/Jul\/1995:)(([0-1][0-9])|20|21|22|23)(?=:)', 1)).cache()
top1_us_date_hour = top1_us_timestamp.withColumn('Date', F.regexp_extract('timestamp', '(([0-2][1-9])|10|20|30|31)(?=\/Jul\/1995)', 1)).withColumn('Hour', F.regexp_extract('timestamp', '(?<=\/Jul\/1995:)(([0-1][0-9])|20|21|22|23)(?=:)', 1)).cache()

jp_date_list = []
jp_hour_list = []
for date_hour in top1_jp_date_hour.select('Date','Hour').collect():
	jp_date_list.append(int(date_hour[0]))
	jp_hour_list.append(int(date_hour[1]))

import numpy as np

jp_grid = np.zeros((24,31))
for i in range(len(jp_date_list)):
    jp_grid[(jp_hour_list[i]-1),(jp_date_list[i]-1)] += 1

# heatmap
fig, ax = plt.subplots(figsize=(20,12))
plt.imshow(jp_grid, cmap='hot', interpolation='none')
plt.gca().invert_yaxis()
plt.xticks(range(0,31),range(1,32))  # set the tick of axis
plt.yticks(range(24))
plt.xlabel('Date in July')
plt.ylabel('Hour in a day')
plt.title("The heat map of requests time in japan(1995/July)", fontsize='large',fontweight='bold')
plt.savefig("../Output/jp_heatmap.png")


#UK
uk_date_list = []
uk_hour_list = []
for date_hour in top1_uk_date_hour.select('Date','Hour').collect():
	uk_date_list.append(int(date_hour[0]))
	uk_hour_list.append(int(date_hour[1]))

uk_grid = np.zeros((24,31))
for i in range(len(uk_date_list)):
    uk_grid[(uk_hour_list[i]-1),(uk_date_list[i]-1)] += 1

# heatmap
fig, ax = plt.subplots(figsize=(20,12))
plt.imshow(uk_grid, cmap='hot', interpolation='none')
plt.gca().invert_yaxis()
plt.xticks(range(0,31),range(1,32))  # 设置x刻度 !! heatmap的坐标轴是0-30，可能报错
plt.yticks(range(24))
plt.xlabel('Date in July')
plt.ylabel('Hour in a day')
plt.title("The heat map of requests time in UK(1995/July)", fontsize='large', fontweight='bold')
plt.savefig("../Output/uk_heatmap.png")

# US
us_date_list = []
us_hour_list = []
for date_hour in top1_us_date_hour.select('Date','Hour').collect():
	us_date_list.append(int(date_hour[0]))
	us_hour_list.append(int(date_hour[1]))

# make grid
us_grid = np.zeros((24,31))
for i in range(len(us_date_list)):
    us_grid[(us_hour_list[i]-1),(us_date_list[i]-1)] += 1

# heatmap
fig, ax = plt.subplots(figsize=(20,12))
plt.imshow(us_grid, cmap='hot', interpolation='none')
plt.gca().invert_yaxis()
plt.xticks(range(0,31),range(1,32))  
plt.yticks(range(24))
plt.xlabel('Date in July')
plt.ylabel('Hour in a day')
plt.title("The heat map of requests time in US(1995/July)", fontsize='large', fontweight='bold')
plt.savefig("../Output/us_heatmap.png")

