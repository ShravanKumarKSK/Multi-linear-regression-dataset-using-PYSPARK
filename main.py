# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/tips.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "True"
delimiter = ","

df = spark.read.csv(file_location,header = True, inferSchema = True)
df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.columns

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

# COMMAND ----------

indexer = StringIndexer(inputCol='sex',outputCol= "sex_indexed")
df_r = indexer.fit(df).transform(df)
df_r.show()

# COMMAND ----------

indexer = StringIndexer(inputCols=['sex','smoker','day','time'],outputCols= ["smoker_indexed","day_indexed","time_indexed", "sex_indexed"])
df_r = indexer.fit(df).transform(df)
df_r.show()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
feat = VectorAssembler(inputCols =['tip','size','sex_indexed','smoker_indexed','day_indexed','time_indexed'], outputCol = 'independent feature')
output = feat.transform(df_r)

# COMMAND ----------

output.show()

# COMMAND ----------

output.select("independent feature").show()

# COMMAND ----------

final_d = output.select("independent feature","total_bill")

# COMMAND ----------

final_d.show()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

train_d, test_d = final_d.randomSplit([0.75,0.25])
reg = LinearRegression(featuresCol = "independent feature", labelCol = 'total_bill')
reg = reg.fit(train_d)

# COMMAND ----------

reg.coefficients


# COMMAND ----------

reg.intercept

# COMMAND ----------

pred_results = reg.evaluate(test_d)

# COMMAND ----------

pred_results.predictions.show()

# COMMAND ----------


