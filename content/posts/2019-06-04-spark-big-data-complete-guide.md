---
title: "Spark大数据处理完全手册：从基础到进阶"
date: 2019-06-04 08:59:32 +0800
lastmod: 2020-11-20 22:08:55 +0800
draft: false
description: "Apache Spark完全使用指南，涵盖Spark SQL、文件操作、MLlib机器学习、PySpark实战技巧等核心内容"
tags: ["系统管理"]
categories:
  - Big Data
  - Data&ML&AI

# 封面图片
images:
  - https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&h=630&fit=crop

# 作者
author: "Sun"
---

Apache Spark是当前最流行的大数据处理框架之一，以其高效、易用和强大的功能著称。本文将从实践角度出发，全面介绍Spark的核心功能和使用技巧，帮助读者快速掌握大数据处理的必备技能。

## 目录

- [环境配置与问题排查](#环境配置与问题排查)
- [Spark SQL核心函数与操作](#spark-sql核心函数与操作)
- [文件读写与数据处理](#文件读写与数据处理)
- [排序与分区控制](#排序与分区控制)
- [PySpark实战技巧](#pyspark实战技巧)
- [MLlib机器学习应用](#mllib机器学习应用)

## 环境配置与问题排查

### Mac单机PySpark环境配置

在Mac上搭建本地PySpark环境时，可能会遇到主机名解析问题：

```text
Caused by: java.net.UnknownHostException: master: nodename nor servname provided, or not known
```

**解决方案：**

1. 修改`$SPARK_HOME/conf/spark-env.sh`配置文件：

```bash
export SPARK_MASTER_IP=StevenMac
export SPARK_LOCAL_IP=StevenMac
```

2. 在`/etc/hosts`文件中添加：

```bash
127.0.0.1  StevenMac
```

### PySpark函数导入问题

使用PySpark时，如果遇到无法找到`col`函数的问题：

```python
from pyspark.sql.functions import col  # 报错：找不到col函数
```

**解决方案：**

安装PySpark类型存根（stubs）：

```bash
pip install pyspark-stubs
```

这样不仅能解决导入问题，还能提供更好的IDE自动补全支持。

### 初始化SparkSession

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName('MySparkApplication') \
    .config("spark.executor.memory", "3g") \
    .config("spark.executor.cores", "8") \
    .getOrCreate()
```

## Spark SQL核心函数与操作

Spark SQL提供了丰富的API用于结构化数据处理，以下是核心操作函数的完整指南。

### 常用函数速查

**数据选择与转换：**
- `select` - 选择列
- `lit` - 创建字面量值
- `withColumn` - 添加或替换列
- `withColumnRenamed` - 重命名列
- `cast` - 类型转换

**过滤与聚合：**
- `filter` / `where` - 数据过滤
- `groupBy` - 分组
- `agg` - 聚合
- `sum`, `countDistinct` - 聚合函数

**连接与合并：**
- `join` - 数据连接
- `union` - 数据合并

**排序与窗口函数：**
- `orderBy`, `sort` - 排序
- `asc`, `desc` - 升序/降序
- `row_number`, `over`, `partitionBy` - 窗口函数

**条件表达式：**
- `when`, `otherwise` - 条件逻辑
- `isin` - 成员判断

### 基础操作示例

```scala
// 选择与字面量
df.select($"id", lit(1).as("cnt"))

// 分组聚合
df.groupBy("id")
  .agg(sum("cnt").as("total"))
  .where("total >= 10")
  .select("uid", "total")

// 条件表达式
df.withColumn("category",
  when($"score" >= 90, "A")
  .when($"score" >= 80, "B")
  .otherwise("C"))

// 类型转换
df.withColumn("new_id", $"id".cast("long"))

// 重命名列
df.withColumnRenamed("oldName", "newName")

// 成员判断
df.filter($"status".isin("active", "pending"))
```

### 高级操作示例

```scala
// 动态SQL表达式
df.selectExpr("id", "score * 0.8 as adjusted_score")

// 透视表
df.groupBy("department").pivot("year").sum("salary")

// 窗口函数
import org.apache.spark.sql.expressions.Window
val windowSpec = Window.partitionBy("department").orderBy("salary")

df.withColumn("rank", row_number().over(windowSpec))
  .withColumn("dense_rank", dense_rank().over(windowSpec))

// 复杂条件组合
df.filter(
  ($"date" >= lit("2020-01-01")) &&
  ($"date" <= lit("2020-12-31")) &&
  ($"status" === "active")
)
```

### 自定义UDF函数

```scala
// 注册UDF
spark.udf.register("myUDF", (input: String) => {
  input.toUpperCase
})

// 使用UDF
df.createOrReplaceTempView("table")
spark.sql("SELECT id, myUDF(name) as upper_name FROM table")
```

### 参考资源

- [Spark SQL官方文档](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [Column API](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/Column.html)
- [Dataset API](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/Dataset.html)
- [functions参考](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/functions$.html)

## 文件读写与数据处理

### 文件读取操作

```scala
// 读取文本文件
val raw = spark.read.textFile("path/to/file.txt")

// 读取CSV文件
val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("path/to/file.csv")

// 读取Parquet文件
val df = spark.read.parquet("path/to/file.parquet")

// 读取JSON文件
val df = spark.read.json("path/to/file.json")
```

### 数据过滤与转换

```scala
// 过滤并映射为DataFrame
val raw_df = raw.filter(x => x.split("\t").length == 3)
  .map(x => (
    x.split("\t")(0),
    x.split("\t")(1).toLong,
    x.split("\t")(2)
  ))
  .toDF("field1", "field2", "field3")

// 复杂过滤条件
val filtered = df.filter(
  ($"timestamp" >= lit("2020-01-01")) &&
  ($"timestamp" <= lit("2020-12-31")) &&
  ($"type" === lit(15)) &&
  ($"name" =!= "") &&
  ($"value" > 0)
)
```

### 数据连接操作

```scala
// 内连接
val joined = df1.join(df2,
  df1("key1") === df2("key1") &&
  df1("key2") === df2("key2"),
  "inner"
)

// 选择所需列
val result = joined.select($"col1", $"col2", $"col3", $"col4")
```

### 数据写入操作

```scala
// 写入文本文件
result.rdd
  .repartition(1)
  .map(row => s"${row(0)},${row(1)}")
  .saveAsTextFile("path/to/output")

// 写入CSV文件
result_df.repartition(1)
  .write
  .mode("overwrite")
  .option("delimiter", "\t")
  .csv("path/to/output")

// 写入Parquet文件
df.write
  .mode("overwrite")
  .parquet("path/to/output")
```

### PySpark文件操作

```python
from pyspark.sql import SparkSession
import pyspark.sql.functions as func

# 读取Parquet文件
df = spark.read.parquet("path/to/file.parquet")

# 数据过滤
df_filtered = df.select('f1', 'f2', 'f3', 'f4').filter(
    (df['f1'] >= 1531670400) &
    (df['f1'] <= 1532188800) &
    (df['f2'] == 15) &
    (df['f3'] != '') &
    (df['f4'] > 0)
)

# 分组聚合
result = df_filtered.groupby(df['f3']).agg(
    func.countDistinct('f1')
)

# 保存结果
result.rdd.repartition(1).map(
    lambda row: str(row[0]) + "," + str(row[1])
).saveAsTextFile("path/to/output")
```

## 排序与分区控制

### 全局排序与分区

```python
# 读取TSV文件
df = spark.read.text("file.tsv").rdd \
    .map(lambda r: r[0]) \
    .map(lambda line: line.split("\t")) \
    .toDF()

# 全局排序并合并为单个文件
df.orderBy("_1", "_2") \
    .coalesce(1) \
    .write.csv("output", sep='\t')
```

### 分区内排序

```scala
// 重新分区并分区内排序
df.repartition(100)
  .sortWithinPartitions("id", "date")
  .write.parquet("output")

// 全局排序后输出
df.orderBy($"id".asc, $"date".desc)
  .repartition(1)
  .write.mode("overwrite")
  .csv("output")
```

### 排序与性能优化

**注意：** 全局排序（`repartition(1)`）会将所有数据集中到单个分区，这在数据量大时会导致性能问题甚至OOM。

**推荐做法：**

1. **小数据集**：使用`coalesce(1)`或`repartition(1)`
2. **大数据集**：保持多个分区，使用分区内排序
3. **特定需求**：使用`sortWithinPartitions`进行分区内排序

## PySpark实战技巧

### 动态导入与初始化

```python
import findspark
findspark.init()

import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName('PySpark-Analysis') \
    .config("spark.executor.memory", "3g") \
    .config("spark.executor.cores", "8") \
    .getOrCreate()
```

### 常用数据处理模式

```python
# 1. 数据清洗
from pyspark.sql.functions import col, trim, lower

df_clean = df.select(
    trim(col("name")).alias("name"),
    lower(col("email")).alias("email")
)

# 2. 数据聚合
from pyspark.sql.functions import sum, count, avg

result = df.groupBy("category") \
    .agg(
        sum("amount").alias("total"),
        count("*").alias("count"),
        avg("score").alias("avg_score")
    )

# 3. 窗口函数
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

window_spec = Window.partitionBy("department").orderBy(col("salary").desc())

df_with_rank = df.withColumn("rank", row_number().over(window_spec))
```

### 数据类型处理

```python
# 字符串转日期
df = df.withColumn("date", col("date_str").cast("date"))

# 字符串转数字
df = df.withColumn("amount", col("amount_str").cast("double"))

# 类型检查
df.printSchema()
df.dtypes
```

## MLlib机器学习应用

Spark MLlib提供了可扩展的机器学习库，支持常见的算法和工具。

### 数据预处理

```scala
// 定义case class
case class CheckIn(user: String, time: String, location: String)

// 读取数据
case class UserFeature(user: String, name: String)

val data = spark.read.textFile(input)
  .map(_.split("\t"))
  .mapPartitions { iter =>
    iter.map { items =>
      UserFeature(items(0), items(1))
    }
  }
```

### 特征工程

```scala
// reduceByKey进行特征聚合
case class CheckIn(user: String, time: String, location: String)

val features = gowalla.map {
  check: CheckIn =>
    (check.user, (1L, Set(check.time), Set(check.location)))
}.rdd.reduceByKey {
  case (left, right) =>
    (left._1 + right._1,
     left._2.union(right._2),
     left._3.union(right._3))
}.map {
  case (_, (checkins: Long, days: Set[String], locations: Set[String])) =>
    // 创建特征向量：签到次数、不同天数、不同地点数
    Vectors.dense(
      checkins.toDouble,
      days.size.toDouble,
      locations.size.toDouble
    )
}
```

### 机器学习流程

```scala
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator

// 1. 特征组装
val assembler = new VectorAssembler()
  .setInputCols(Array("feature1", "feature2", "feature3"))
  .setOutputCol("features")

// 2. 特征标准化
val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false)

// 3. 模型训练
val lr = new LinearRegression()
  .setLabelCol("label")
  .setFeaturesCol("scaledFeatures")
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

// 4. Pipeline
import org.apache.spark.ml.Pipeline

val pipeline = new Pipeline()
  .setStages(Array(assembler, scaler, lr))

val model = pipeline.fit(trainingData)

// 5. 模型评估
val predictions = model.transform(testData)

val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) = $rmse")
```

## 最佳实践与性能优化

### 1. 缓存策略

```scala
// 缓存频繁使用的DataFrame
df.cache()
df.persist(StorageLevel.MEMORY_AND_DISK)

// 取消缓存
df.unpersist()
```

### 2. 分区优化

```scala
// 根据数据量和集群资源调整分区数
df.repartition(200)  // 增加分区
df.coalesce(50)      // 减少分区

// 避免数据倾斜
df.repartition($"key"))
```

### 3. 广播变量

```scala
// 小表广播
val broadcastDF = broadcast(smallDF)
largeDF.join(broadcastDF, largeDF("key") === broadcastDF("key"))
```

### 4. 内存管理

```scala
// 调整executor配置
spark.conf.set("spark.executor.memory", "4g")
spark.conf.set("spark.executor.memoryOverhead", "1g")
spark.conf.set("spark.memory.fraction", "0.8")
spark.conf.set("spark.memory.storageFraction", "0.3")
```

## 总结

Apache Spark提供了强大而灵活的大数据处理能力，本文涵盖了从基础配置到高级应用的完整技术栈：

- **环境配置**：Mac单机环境搭建与问题排查
- **Spark SQL**：核心函数、DSL语法、UDF开发
- **文件操作**：读写、转换、连接等ETL操作
- **排序优化**：全局排序、分区内排序与性能权衡
- **PySpark实战**：Python生态集成与开发技巧
- **MLlib应用**：特征工程与机器学习流程

掌握这些技能后，你可以高效处理TB级数据，构建可扩展的数据处理管道，并实现复杂的机器学习任务。Spark的分布式计算能力将大大提升你的数据处理效率。

## 参考资源

- [Apache Spark官方文档](https://spark.apache.org/docs/latest/)
- [Spark SQL编程指南](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [MLlib机器学习库](https://spark.apache.org/docs/latest/ml-guide.html)
- [PySpark官方文档](https://spark.apache.org/docs/latest/api/python/)
