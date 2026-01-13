---
title: "Shell文本处理三剑客：sed、awk与grep实战指南"
date: 2015-12-18T14:48:18+08:00
draft: false
tags:
  - Shell
  - sed
  - awk
  - grep
  - 文本处理
  - 命令行
description: "深入掌握sed、awk和grep这三个强大的文本处理工具，通过实战示例学习如何在命令行下高效处理文本数据。"
cover:
  image: "/images/covers/1555421689-491a97ff2040.jpg"
  alt: "文本处理命令行"
  caption: "sed、awk、grep：命令行文本三剑客"
---

在Unix/Linux系统中，sed、awk和grep被称为文本处理三剑客。它们各自擅长不同的文本处理任务，配合使用可以解决绝大多数文本处理需求。本文将通过实战示例帮助你掌握这些工具的核心用法。

## grep：文本搜索利器

### 基本搜索

grep主要用于在文件中搜索匹配特定模式的行。

```bash
# 在文件中搜索单词
grep "pattern" file.txt

# 递归搜索目录
grep -r "pattern" /path/to/dir

# 忽略大小写
grep -i "pattern" file.txt

# 显示行号
grep -n "pattern" file.txt

# 反向匹配（不包含pattern的行）
grep -v "pattern" file.txt

# 统计匹配行数
grep -c "pattern" file.txt
```

### 多文件操作

```bash
# 查找在多个文件中都存在的行
grep -F -x -f file1 file2 file3

# 查找在file1中但不在file2中的行
grep -F -x -v -f file2 file1

# 在多个文件中搜索
grep "pattern" file1.txt file2.txt file3.txt
```

**参数说明**：
- `-F`：将模式视为固定字符串而非正则表达式
- `-x`：整行匹配
- `-f`：从文件读取模式
- `-v`：反向选择

### 实用示例

```bash
# 查找包含"error"或"warning"的行
grep -E "(error|warning)" logfile.txt

# 查找以"#"开头的注释行
grep "^#" config.conf

# 查找空行
grep "^$" file.txt

# 查找非空行
grep -v "^$" file.txt

# 查找恰好10个字符的行
grep -E "^.{10}$" file.txt

# 递归查找当前目录下所有.py文件中的"TODO"
grep -r "TODO" --include="*.py" .
```

## sed：流编辑器

sed是一个强大的流编辑器，擅长进行文本替换和删除操作。

### 基本替换

```bash
# 替换每行第一个匹配
sed 's/foo/bar/' file.txt

# 全局替换（每行所有匹配）
sed 's/foo/bar/g' file.txt

# 只在第4个匹配处替换
sed 's/foo/bar/4' file.txt

# 删除匹配的行
sed '/pattern/d' file.txt

# 原地编辑文件（直接修改文件）
sed -i 's/foo/bar/g' file.txt
```

### 行操作

```bash
# 删除前10行
sed '1,10d' file.txt

# 删除最后一行
sed '$d' file.txt

# 打印前10行（模拟head -10）
sed 10q file.txt

# 打印第52行
sed -n '52p' file.txt

# 打印8-12行
sed -n '8,12p' file.txt

# 从第3行开始，每7行打印一次
sed -n '3,${p;n;n;n;n;n;n;}' file.txt
```

### 空行处理

```bash
# 删除所有空行
sed '/^$/d' file.txt

# 删除连续的空行（压缩为单个空行）
sed '/./,/^$/!d' file.txt

# 删除文件开头的所有空行
sed '/./,$!d' file.txt

# 删除文件末尾的所有空行
sed -e :a -e '/^\n*$/{$d;N;ba' -e '}' file.txt

# 在匹配regex的行前后添加空行
# 之前添加
sed '/regex/{x;p;x;}' file.txt
# 之后添加
sed '/regex/G' file.txt
# 前后都添加
sed '/regex/{x;p;x;G;}' file.txt
```

### 高级替换技巧

```bash
# 只在包含"baz"的行上替换"foo"为"bar"
sed '/baz/s/foo/bar/g' file.txt

# 只在不包含"baz"的行上替换
sed '/baz/!s/foo/bar/g' file.txt

# 将scarlet、ruby或puce都改为red
sed 's/scarlet/red/g;s/ruby/red/g;s/puce/red/g' file.txt

# 删除行首空白
sed 's/^[ \t]*//' file.txt

# 删除行尾空白
sed 's/[ \t]*$//' file.txt

# 删除首尾空白
sed 's/^[ \t]*//;s/[ \t]*$//' file.txt

# 在每行开头插入5个空格
sed 's/^/     /' file.txt
```

### 换行符处理

```bash
# DOS转Unix（删除^M）
sed 's/.$//' file.txt
# 或
sed 's/^M$//' file.txt
# 或
sed 's/\x0D$//' file.txt

# Unix转DOS
sed 's/$/\r/' file.txt
```

### 打印控制

```bash
# 只打印匹配正则表达式的行（模拟grep）
sed -n '/regexp/p' file.txt

# 只打印不匹配的行（模拟grep -v）
sed -n '/regexp/!p' file.txt

# 打印匹配regex的前一行（但不打印匹配行本身）
sed -n '/regexp/{g;1!p;};h' file.txt

# 打印匹配regex的后一行
sed -n '/regexp/{n;p;}' file.txt

# 打印regex前后的行及行号（模拟grep -A1 -B1）
sed -n -e '/regexp/{=;x;1!p;g;$!N;p;D;}' -e h file.txt

# 打印长度>=65字符的行
sed -n '/^.\{65\}/p' file.txt

# 打印长度<65字符的行
sed '/^.\{65\}/d' file.txt
```

### 模式匹配

```bash
# 匹配AAA和BBB和CCC（任意顺序）
sed '/AAA/!d; /BBB/!d; /CCC/!d' file.txt

# 匹配AAA.*BBB.*CCC（固定顺序）
sed '/AAA.*BBB.*CCC/!d' file.txt

# 匹配AAA或BBB或CCC
sed -e '/AAA/b' -e '/BBB/b' -e '/CCC/b' -e d file.txt

# 打印两个正则表达式之间的行
sed -n '/Iowa/,/Montana/p' file.txt

# 删除两个正则表达式之间的行
sed '/Iowa/,/Montana/d' file.txt

# 从正则表达式到文件末尾
sed -n '/regexp/,$p' file.txt
```

### 去重操作

```bash
# 删除重复的连续行（模拟uniq）
sed '$!N; /^\(.*\)\n\1$/!P; D' file.txt

# 只删除重复的连续行（模拟uniq -d）
sed '$!N; s/^\(.*\)\n\1$/\1/; t; D' file.txt

# 每5行后添加一个空行
sed 'n;n;n;n;G;' file.txt
```

## awk：文本处理语言

awk是一种完整的编程语言，特别适合处理结构化文本和报表生成。

### 基本用法

```bash
# 打印包含正则的行（模拟grep）
awk '/regex/' file.txt

# 打印不包含正则的行（模拟grep -v）
awk '!/regex/' file.txt

# 打印字段数>4的行
awk 'NF > 4' file.txt

# 打印最后一个字段>4的行
awk '$NF > 4' file.txt

# 打印第5个字段等于"abc123"的行
awk '$5 == "abc123"' file.txt

# 打印第5个字段不等于"abc123"的行
awk '$5 != "abc123"' file.txt

# 打印第7个字段匹配正则的行
awk '$7 ~ /^[a-f]/' file.txt
# 或不匹配
awk '$7 !~ /^[a-f]/' file.txt
```

### 字段处理

```bash
# 打印第一个字段
awk '{print $1}' file.txt

# 交换前两个字段
awk '{ temp = $1; $1 = $2; $2 = temp; print }' file.txt

# 删除第二个字段
awk '{ $2 = ""; print }' file.txt

# 反向打印每行的字段
awk '{ for (i=NF; i>0; i--) printf("%s ", $i); printf ("\n") }' file.txt

# 中心对齐文本（79字符宽度）
awk '{ l=length(); s=int((79-l)/2); printf "%"(s+l)"s\n", $0 }' file.txt
```

### 文本替换与清理

```bash
# 删除行首空白
awk '{ sub(/^[ \t]+/, ""); print }' file.txt

# 删除行尾空白
awk '{ sub(/[ \t]+$/, ""); print }' file.txt

# 删除首尾空白
awk '{ gsub(/^[ \t]+|[ \t]+$/, ""); print }' file.txt

# 在行首插入5个空格
awk '{ sub(/^/, "     "); print }' file.txt

# 替换foo为bar
awk '{ sub(/foo/,"bar"); print }' file.txt

# 只在包含baz的行上替换
awk '/baz/ { gsub(/foo/, "bar") }; { print }' file.txt

# 只在不包含baz的行上替换
awk '!/baz/ { gsub(/foo/, "bar") }; { print }' file.txt

# 将scarlet|ruby|puce改为red
awk '{ gsub(/scarlet|ruby|puce/, "red"); print}' file.txt
```

### 换行符处理

```bash
# DOS转Unix
awk '{ sub(/\r$/,""); print }' file.txt

# Unix转DOS
awk '{ sub(/$/,"\r"); print }' file.txt
```

### 行选择

```bash
# 打印8-12行
awk 'NR==8,NR==12' file.txt

# 打印第52行
awk 'NR==52' file.txt

# 从正则表达式到文件末尾
awk '/regex/,0' file.txt

# 两个正则表达式之间
awk '/Iowa/,/Montana/' file.txt

# 删除所有空行
awk NF file.txt

# 打印regex的前一行
awk '/regex/ { print x }; { x=$0 }' file.txt

# 打印regex的后一行
awk '/regex/ { getline; print }' file.txt
```

### 去重与排序

```bash
# 删除连续重复行（模拟uniq）
awk 'a !~ $0; { a = $0 }' file.txt

# 删除所有重复行（包括非连续）
awk '!a[$0]++' file.txt

# 反转行顺序（模拟tac）
awk '{ a[i++] = $0 } END { for (j=i-1; j>=0;) print a[j--] }' file.txt

# 每5行用逗号连接
awk 'ORS=NR%5?",":"\n"' file.txt
```

### 统计计算

```bash
# 统计包含"Regex"的行数
awk '/Regex/ { n++ }; END { print n+0 }' file.txt

# 打印长度>64的行
awk 'length > 64' file.txt
```

## 组合使用技巧

### 按行长度排序

```bash
# 按行长度从长到短排序
cat $file | awk '{ print length($0) " " $0; }' | sort -r -n | cut -d ' ' -f 2-
```

### 查看最常用命令

```bash
# 统计使用最频繁的10个命令
history | awk '{a[$2]++} END {for(i in a) {print a[i]" "i}}'| sort -rn | head
```

### 批量下载文件

```bash
# 下载连续编号的文件
wget http://example.com/lecture{1..26}.pdf
```

## 工具选择指南

### 何时使用grep

- 只需要查找匹配的行
- 简单的模式匹配
- 不需要修改文件内容
- 快速搜索大量文件

### 何时使用sed

- 需要进行文本替换
- 删除特定行
- 简单的行编辑操作
- 基于位置的行操作

### 何时使用awk

- 需要处理结构化数据（如CSV）
- 需要进行字段级别的操作
- 需要进行计算或统计
- 复杂的条件判断

## 性能优化建议

1. **grep最快**：对于简单的搜索任务，优先使用grep
2. **sed适合替换**：文本替换首选sed，比awk更快
3. **awk最灵活**：复杂的数据处理用awk
4. **减少管道**：尽量在一个工具内完成，减少数据传输
5. **使用基本正则**：除非必要，否则使用基本正则而非扩展正则

## 实战案例

### 日志分析

```bash
# 统计访问量前十的IP
awk '{print $1}' access.log | sort | uniq -c | sort -rn | head

# 查找500错误
grep " 500 " error.log | awk '{print $4,$5,$6}' | sort | uniq -c

# 分析响应时间
awk '{print $NF}' log.txt | awk -F'"' '{print $2}' | sort -n
```

### 配置文件处理

```bash
# 启用被注释的配置
sed -e 's/^#\(.*\)/\1/' -e '/^$/d' config.conf

# 只保留有效的配置行
grep -v '^#' config.conf | grep -v '^$'

# 替换配置值
sed 's/DB_NAME=.*/DB_NAME=production/' .env
```

## 学习资源

- [Sed One-Liners Explained](http://www.catonmat.net/blog/sed-one-liners-explained-part-one/)
- [Awk One-Liners Explained](http://www.catonmat.net/blog/awk-one-liners-explained-part-one/)
- [Grep Tutorial](https://www.grymoire.com/Unix/Grep.html)

## 小结

sed、awk和grep各有所长：
- **grep**：搜索专家，快速找到你需要的内容
- **sed**：替换能手，高效进行文本转换
- **awk**：处理语言，解决复杂的数据操作

掌握这三个工具，你就拥有了处理任何文本问题的能力。记住它们各自的优势，在实际工作中灵活运用，你会发现命令行下的文本处理既高效又优雅。
