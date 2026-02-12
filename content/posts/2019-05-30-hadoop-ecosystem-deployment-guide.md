---
title: "Hadoop生态系统部署实践：从数据准备到集群配置"
date: 2019-05-30T09:23:39+08:00
lastmod: 2019-05-30T09:23:39+08:00
draft: false
keywords: ["Hadoop", "Presto", "HUE", "Zeppelin", "EMR", "大数据", "集群部署"]
description: "详细介绍Hadoop生态系统的部署实践，包括数据准备、Presto配置、HUE用户管理和Zeppelin集成"
categories: ["大数据"]
tags: ["系统管理"]

# 封面图片配置
cover:
    image: "images/covers/1518186285589-2f7649de83e0.jpg"
    alt: "大数据处理"
    caption: "Hadoop生态系统：分布式计算的基石"
    relative: false
    hidden: false
---

本文将详细介绍Hadoop生态系统各组件的实际部署与配置经验，涵盖数据准备、Presto调试、HUE用户管理和Zeppelin集成等关键环节。

## 一、气象数据准备与处理

《Hadoop权威指南》是一本经典的Hadoop学习资料，书中使用了NCDC（国家气候数据中心）的气象数据作为示例。这些真实的气象数据不仅有助于理解Hadoop的工作原理，更能培养处理复杂数据的实战能力。

### 1.1 数据下载

NCDC提供了丰富的历史气象数据，我们可以通过脚本批量下载：

```bash
#!/bin/bash

# 进入目标下载目录
cdir="$(cd `dirname $0`; pwd)"

# 下载1930-1960年的气象数据
# 注意：tar文件从1930年开始才有实际数据
for i in $(seq 1930 1960)
do
    wget --execute robots=off \
         --accept=tar \
         -r -np -nH \
         --cut-dirs=4 \
         -R index.html* \
         ftp://ftp.ncdc.noaa.gov/pub/data/gsod/$i/
done
```

### 1.2 数据预处理

下载完成后，需要重新组织文件结构：

```bash
# 将 1930/gsod_1930.tar 重命名为 1930/1930.tar
# 并将所有文件集中到gsod目录
# 最终结构：gsod/1930/1930.tar, gsod/1931/1931.tar ...
```

### 1.3 HDFS数据上传

在HDFS上创建目录并上传数据：

```bash
# 创建HDFS目录
hdfs dfs -mkdir /GSOD /GSOD_ALL

# 上传数据文件
hdfs dfs -put gsod/* /GSOD/
```

**常见问题处理：**

如果遇到NameNode无法找到或安全模式问题，可以执行：

```bash
# 停止所有服务
stop-all.sh

# 格式化NameNode（注意：这会清空现有数据）
hdfs namenode -format

# 启动所有服务
start-all.sh

# 离开安全模式
hadoop dfsadmin -safemode leave

# 重新上传数据
hdfs dfs -put gsod/* /GSOD/
```

### 1.4 MapReduce数据处理

#### 生成输入文件列表

```bash
#!/bin/bash

a=$1
rm -rf ncdc_files.txt
hdfs dfs -rm /ncdc_files.txt

while [ $a -le $2 ]
do
    filename="/GSOD/${a}/${a}.tar"
    echo "$filename" >> ncdc_files.txt
    a=`expr $a + 1`
done

hdfs dfs -put ncdc_files.txt /
```

使用方法：
```bash
sh generate_input_list.sh 1901 1956
```

#### 创建Map处理脚本

```bash
#!/bin/bash

read offset hdfs_file
echo -e "$offset\t$hdfs_file"

# 从HDFS获取文件到本地
echo "reporter:status:Retrieving $hdfs_file" >&2
hdfs dfs -get $hdfs_file .

# 创建本地目录
target=`basename $hdfs_file .tar`
mkdir $target

# 解压tar文件
echo "reporter:status:Un-tarring $hdfs_file to $target" >&2
tar xf `basename $hdfs_file` -C $target

# 解压每个站点文件并合并
echo "reporter:status:Un-gzipping $target" >&2
for file in $target/*
do
    gunzip -c $file >> $target.all
    echo "reporter:status:Processed $file" >&2
done

# 压缩并上传到HDFS
echo "reporter:status:Gzipping $target and putting in HDFS" >&2
gzip -c $target.all | hdfs dfs -put - /GSOD_ALL/$target.gz

# 清理临时文件
rm `basename $hdfs_file`
rm -r $target
rm $target.all
```

#### 提交MapReduce任务

```bash
#!/bin/bash

hadoop jar ${HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming-3.1.2.jar \
    -D mapreduce.job.reduces=0 \
    -D mapreduce.map.speculative=false \
    -D mapreduce.task.timeout=12000000 \
    -inputformat org.apache.hadoop.mapred.lib.NLineInputFormat \
    -input /ncdc_files.txt \
    -output /output/gsod \
    -mapper load_ncdc_map.sh \
    -file load_ncdc_map.sh
```

#### 验证结果

```bash
# 检查输出文件
hdfs dfs -ls /GSOD_ALL

# 查看处理结果
hdfs dfs -cat /output/gsod/part-00053
```

## 二、Presto配置与调试

Presto是Facebook开源的分布式SQL查询引擎，能够快速查询海量数据。

### 2.1 Presto CLI调试

使用调试模式连接Presto：

```bash
presto-cli --catalog hive \
           --schema $schema \
           --output-format CSV_HEADER \
           --server $ip:$port \
           --debug
```

这个命令会输出详细的诊断信息，帮助定位问题。

### 2.2 服务管理

```bash
# 查看Presto服务状态
initctl list | grep -i presto

# 停止Presto服务
sudo stop presto-server

# 启动Presto服务
sudo start presto-server
```

### 2.3 日志查看

Presto的日志文件位于 `/var/log/presto` 目录，这些日志对于问题诊断非常重要：

```bash
# 查看最新日志
tail -f /var/log/presto/server.log

# 查看错误日志
grep ERROR /var/log/presto/server.log
```

**常见问题排查：**

1. **连接失败**：检查服务状态和端口配置
2. **查询超时**：调整查询超时参数
3. **内存不足**：优化JVM堆内存设置

## 三、HUE超级用户创建

HUE（Hadoop User Experience）是一个开源的Web界面，用于与Hadoop生态系统交互。

### 3.1 创建超级用户

在EMR环境中，使用以下命令创建HUE超级用户：

```bash
cd /usr/lib/hue/
sudo build/env/bin/hue createsuperuser
```

执行后，系统会提示输入用户名、邮箱和密码等信息。

### 3.2 用户权限管理

创建超级用户后，可以：
- 管理其他HUE用户
- 配置用户权限
- 管理HUE的工作流和查询

## 四、Zeppelin集成Presto

Apache Zeppelin是一个基于Web的笔记本，支持交互式数据分析和可视化。

### 4.1 安装JDBC解释器

在EMR的master节点上安装JDBC解释器以支持Presto：

```bash
# 安装JDBC解释器
sudo /usr/lib/zeppelin/bin/install-interpreter.sh --name jdbc

# 重启Zeppelin服务
sudo stop zeppelin
sudo start zeppelin
```

### 4.2 配置Presto连接

在Zeppelin中配置Presto JDBC连接：

1. 打开Zeppelin Web界面
2. 进入Interpreter设置
3. 找到JDBC解释器
4. 添加Presto配置：

```properties
presto.url=jdbc:presto://<presto-coordinator-host>:8080/hive/default
presto.user=your_username
presto.password=your_password
presto.driver=com.facebook.presto.jdbc.PrestoDriver
```

### 4.3 使用示例

在Zeppelin笔记本中执行Presto查询：

```sql
%jdbc(presto)
SELECT * FROM your_table LIMIT 10;
```

## 五、最佳实践与建议

### 5.1 数据管理

- 定期清理HDFS上的临时文件
- 合理设置数据副本系数
- 使用压缩格式存储数据

### 5.2 性能优化

- 根据数据量调整MapReduce参数
- 合理配置Presto的内存和并发设置
- 使用分区表提升查询效率

### 5.3 监控与维护

- 建立完善的日志监控机制
- 定期检查集群健康状态
- 及时升级组件版本

## 六、总结

本文详细介绍了Hadoop生态系统各组件的实际部署经验：

1. **数据准备**：从NCDC下载气象数据，使用MapReduce进行处理
2. **Presto调试**：掌握CLI调试技巧和服务管理方法
3. **HUE管理**：创建超级用户进行权限管理
4. **Zeppelin集成**：配置JDBC解释器连接Presto

这些实践经验对于构建稳定、高效的Hadoop集群具有重要参考价值。在实际应用中，还需要根据具体业务需求进行调整和优化。

## 参考资料

- [Hadoop: The Definitive Guide](http://hadoopbook.com)
- [NCDC气象数据](ftp://ftp.ncdc.noaa.gov/pub/data/gsod/)
- [Presto官方文档](https://prestodb.io/)
- [Apache Zeppelin文档](https://zeppelin.apache.org/)
- [HUE官方文档](https://gethue.com/)
