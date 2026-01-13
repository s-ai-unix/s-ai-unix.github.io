---
title: "Perl One-liners实用指南"
date: 2014-07-24T15:05:16+08:00
draft: false
tags:
  - Perl
  - One-liners
  - 命令行
  - 文本处理
description: "Perl one-liners是强大的命令行工具，可以在不创建脚本文件的情况下快速完成文本处理任务。本文收集了最实用的Perl one-liners示例。"
cover:
  image: "/images/covers/1526374965328-7f61d4dc18c5.jpg"
  alt: "终端命令行"
  caption: "Perl One-liners：命令行下的瑞士军刀"
---

Perl one-liners是命令行下的瑞士军刀，能够在不创建脚本文件的情况下快速完成复杂的文本处理任务。它们简洁、强大且高效。

## 命令行参数基础

### 常用参数

- `-e`：执行后面的代码
- `-n`：逐行读取输入，类似于`while (<>) {...}`
- `-p`：逐行读取并自动打印
- `-l`：自动处理行结束符
- `-a`：自动分割行到`@F`数组
- `-F`：指定分割模式
- `-i`：原地编辑文件
- `-M`：加载模块

### 基本模式

```bash
# -n模式（不自动打印）
perl -ne 'print if /pattern/' file.txt

# -p模式（自动打印）
perl -pe 's/old/new/g' file.txt

# -i模式（原地编辑）
perl -pi -e 's/old/new/g' file.txt

# -a模式（自动分割）
perl -lane 'print $F[0]' file.txt
```

## 文本处理

### 删除空行

```bash
# 删除所有空行
perl -ne 'print unless /^$/' file.txt
cat file.txt | perl -ne 'print unless /^$/'

# 删除连续空行，只保留一行
perl -00 -pe '' file.txt

# 压缩/扩展空行为N行
perl -00 -pe '$_.="\n"x4' file.txt

# 替代方案
perl -pi -e 's!^\s+?$!!' file.txt
```

### 行操作

```bash
# 在每行前添加空行
perl -pe 's//\n/' file.txt

# 删除每行前导空格
perl -ple 's/^[ \t]+//' file.txt

# 删除每行尾随空格
perl -ple 's/[ \t]+$//' file.txt

# 删除首尾空格
perl -ple 's/^[ \t]+|[ \t]+$//g' file.txt
```

### 大小写转换

```bash
# 转换为大写
cat file | perl -nle 'print uc'

# 驼峰式命名
cat file | perl -ple 's/(\w+)/\u$1/g'
```

## 搜索与替换

### 基本替换

```bash
# 全局替换
perl -pi -e 's/good/bad/g' file.txt

# 只在匹配的行上替换
perl -pi -e 's/good/bad/g if /matched/' file

# 多条件替换
cat file | perl -pe '/baz/ && s/foo/bar/'
```

### 复杂匹配

```bash
# 匹配多个正则（任意顺序）
cat file | perl -ne '/AAA/ && /BBB/ && print'

# 匹配正则序列
cat file | perl -ne '/AAA.*BBB.*CCC/ && print'

# 不匹配某些模式
cat file | perl -ne '!/regex/ && print'

# 不匹配多个模式
cat file | perl -ne '!/AAA/ && !/BBB/ && print'
```

## 行选择与过滤

### 按行号选择

```bash
# 打印第13行
perl -ne '$. == 13 && print && exit' file.txt

# 打印前10行（模拟head -10）
perl -ne 'print if $. <= 10' file.txt

# 打印第一行（模拟head -1）
cat file | perl -ne 'print; exit'

# 打印最后一行
cat file | perl -ne '$last = $_; END { print $last }'
# 或
cat file | perl -ne 'print if eof'

# 打印最后10行（模拟tail -10）
perl -ne 'push @a, $_; @a = @a[@a-10..$#a]; END { print @a }' file.txt

# 打印行13-30
perl -ne 'print if $. >= 17 && $. <= 30' file.txt

# 打印指定行
perl -ne 'print if $. == 13 || $. == 19 || $. == 67' file.txt

# 排除特定行
perl -ne '$. != 13 && print' file.txt
```

### 按模式选择

```bash
# 打印两个正则之间的行
cat file | perl -ne 'print if /regex1/../regex2/'

# 打印前一行
cat file | perl -ne '/regex/ && $last && print $last; $last = $_'

# 打印后一行
cat file | perl -ne 'if ($p) { print; $p = 0 } $p++ if /regex/'

# 只打印包含字母的行
perl -ne 'print if /^[[:alpha:]]+$/' file.txt
```

### 行统计

```bash
# 打印非空行数
cat file.txt | perl -le 'print scalar(grep{/./}<>)'

# 打印空行数
cat file.txt | perl -lne '$a++ if /^$/; END {print $a+0}'
# 或
cat file.txt | perl -le 'print scalar(grep{/^$/}<>)'
# 或
cat file.txt | perl -le 'print ~~grep{/^$/}<>'

# 匹配模式的行数（模拟grep -c）
cat file.txt | perl -lne '$a++ if /good/; END {print $a+0}'
# 或
cat file.txt | grep -c "good"
```

## 数据处理

### 数值计算

```bash
# 对每行的数字求和
cat file.txt | perl -MList::Util=sum -alne 'print sum @F'

# 计算第一列的和
cat file.txt | perl -lane '$sum += $F[0]; END { print $sum }'

# 计算所有数字的和
cat file.txt | perl -alne '$sum += $_ for @F; END { print $sum }'
```

### 数据转换

```bash
# Base64编码字符串
perl -MMIME::Base64 -e 'print encode_base64("string")'

# Base64编码整个文件
perl -MMIME::Base64 -0777 -ne 'print encode_base64($_)' file

# Base64解码
perl -MMIME::Base64 -le 'print decode_base64("c3RyaW5n")'

# URL转义
perl -MURI::Escape -le 'print uri_escape("1+2")'

# URL反转义
perl -MURI::Escape -le 'print uri_unescape("1%2B2")'

# HTML编码
perl -MHTML::Entities -le 'print encode_entities("<br>")'

# HTML解码
perl -MHTML::Entities -le 'print decode_entities("&lt;br&gt;")'
```

### 重复行处理

```bash
# 查找所有重复行
perl -ne 'print if $a{$_}++' file.txt

# 只打印第一次出现的重复行
perl -ne 'print if ++$a{$_} == 2' file.txt

# 打印唯一行
perl -ne 'print unless $a{$_}++' file.txt
```

## 列表生成

### 生成序列

```bash
# 生成并打印字母表
perl -le 'print ("a".."z")'
# 或
perl -le 'print a..z'
# 或
perl -le 'print join "", ("a".."z")'

# 生成1-100的奇数
perl -le '@odd = grep {$_ % 2 == 1} 1..100; print "@odd"'

# 生成随机8字符密码
perl -le 'print map { ("a".."z")[rand 26] } 1..8'
```

### 数据分析

```bash
# 打印字符串长度
perl -le 'print length "hello boy"'

# 计算数组元素数
perl -le '@array = ("a".."z"); print ~~@array'
# 或
perl -le '@array = ("a".."z"); print scalar @array'
# 或
perl -le '@array = ("a".."z"); print $#array + 1'

# 获取字符的数值
perl -le 'print join ", ", map { ord } split //, "hello world"'
```

## 系统管理

### 用户信息

```bash
# 获取系统所有用户名
perl -a -F: -lne 'print $F[4]' /etc/passwd
```

### 日期计算

```bash
# 计算10天前的日期
perl -MPOSIX -le '@now = localtime; $now[3] -= 10; print scalar localtime mktime @now'
```

## 实用技巧

### 行号处理

```bash
# 添加行号
perl -ne 'print "$. $_"' file.txt
# 或
perl -pe '$_ = "$. $_"' file.txt
```

### 长度过滤

```bash
# 打印长度>=80的行
perl -ne 'print if length >= 80' file.txt

# 打印最长的行
perl -ne '$l = $_ if length($_) > length($l); END { print $l }' file.txt

# 打印最短的行
perl -ne '$s = $_ if $. == 1; $s = $_ if length($_) < length($s); END { print $s }' file.txt
```

### 调试技巧

```bash
# 查看自动分割后的数组
cat file.txt | perl -MData::Dumper -alne 'print Dumper @F'
```

## 高级示例

### 复杂管道操作

```bash
# 实际工作中的复杂示例
cat file1.txt | \
  perl -nle 'print $1 if /\b(__[0-9a-z]\w+)\b/i;' | \
  sort | uniq | \
  xargs -I {} grep {} -w fileb.txt | \
  awk '$2==0' | \
  awk '{print $7}' | \
  sort | uniq | \
  xargs -I {} grep {} -w filec.txt | \
  awk '$8==0' | \
  awk '{print $8," ",$13}' > /tmp/result.txt
```

这个命令链：
1. 从file1提取特定模式
2. 排序去重
3. 在fileb中查找
4. 过滤第二列为0的行
5. 提取第7列
6. 再次在filec中查找
7. 输出特定列到结果文件

## 性能提示

1. 对于大文件，`-n`和`-p`比`while (<>)`更高效
2. 使用`-a`自动分割比手动split快
3. 尽量使用内置函数和操作符
4. 复杂操作考虑编写完整的脚本

## 安全注意事项

1. 处理不可信数据时小心eval
2. 使用`-T`开关启用污点检查
3. 验证和清理输入数据
4. 小心使用`-i`（会原地修改文件）

## 学习资源

- [Perl One-Liners Explained](http://www.catonmat.net/blog/perl-one-liners-explained-part-one/)
- [Introduction to Perl One-Liners](http://www.catonmat.net/blog/introduction-to-perl-one-liners/)
- [Perl One-Liners PDF](http://linux.gda.pl/spotkania/sp_13/perl-one-liners.pdf)

## 小结

Perl one-liners是文本处理的利器，掌握这些技巧可以大幅提高命令行工作效率。从简单的文本替换到复杂的数据处理，Perl one-liners都能胜任。记住常用的模式，并根据需要组合使用，你会发现它们是日常工作的得力助手。

实践是最好的学习方式，建议在实际工作中尝试使用这些one-liners，逐渐积累自己的技巧库。
