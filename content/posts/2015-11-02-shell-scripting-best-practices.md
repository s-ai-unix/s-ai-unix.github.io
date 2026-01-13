---
title: "Shell脚本编程最佳实践"
date: 2015-11-02T16:21:57+08:00
draft: false
tags:
  - Shell
  - Bash
  - 脚本编程
  - 自动化
  - 系统管理
  - 文本处理
description: "深入学习Shell脚本编程，从基础语法到高级技巧，掌握命令行自动化的核心技能。"
cover:
  image: "/images/covers/1550751827-4bd374c3f58b.jpg"
  alt: "终端命令行"
  caption: "Shell脚本编程：自动化的艺术"
---

Shell脚本是系统管理和自动化任务的利器。本文将带你从基础到高级，全面掌握Shell脚本编程的最佳实践。

## Shell基础

### 第一个Shell脚本

```bash
#!/bin/bash
# 这是一个注释
echo "Hello, World!"

# 变量赋值和使用
name="World"
echo "Hello, $name!"

# 命令替换
current_date=$(date)
echo "Today is: $current_date"

# 反引号方式（不推荐）
current_date=`date`
echo "Today is: $current_date"
```

**Shebang说明**：
- `#!/bin/bash`：使用bash解释器
- `#!/bin/sh`：使用sh解释器（更通用）
- `#!/usr/bin/env bash`：自动查找bash（更便携）

### 变量和数据类型

```bash
# 字符串变量
greeting="Hello"
name="Alice"

# 只读变量
readonly PI=3.14159

# 删除变量
unset name

# 环境变量
export PATH=$PATH:/new/path

# 字符串拼接
fullname="John $greeting"
echo $fullname

# 获取字符串长度
string="Hello, World"
echo ${#string}  # 13

# 字符串切片
echo ${string:0:5}  # Hello
echo ${string:7}    # World

# 默认值
echo ${name:-"Guest"}  # 如果name未设置或为空，使用"Guest"

# 数组
arr=(apple banana cherry)
echo ${arr[0]}        # apple
echo ${arr[@]}        # 所有元素
echo ${#arr[@]}       # 数组长度
arr[3]="date"         # 添加元素
unset arr[1]          # 删除元素
```

## 控制结构

### 条件判断

```bash
# if语句
if [ "$name" == "Alice" ]; then
    echo "Welcome, Alice!"
elif [ "$name" == "Bob" ]; then
    echo "Welcome, Bob!"
else
    echo "Welcome, Guest!"
fi

# 数字比较
count=10
if [ $count -eq 10 ]; then
    echo "Count is 10"
fi

if [ $count -gt 5 ]; then
    echo "Count is greater than 5"
fi

if [ $count -lt 20 ]; then
    echo "Count is less than 20"
fi

# 字符串比较
if [ "$string1" == "$string2" ]; then
    echo "Strings are equal"
fi

if [ -n "$string" ]; then
    echo "String is not empty"
fi

# 文件测试
if [ -f "file.txt" ]; then
    echo "File exists and is a regular file"
fi

if [ -d "/tmp" ]; then
    echo "Directory exists"
fi

if [ -r "file.txt" ]; then
    echo "File is readable"
fi

if [ -w "file.txt" ]; then
    echo "File is writable"
fi

if [ -x "script.sh" ]; then
    echo "File is executable"
fi

# 逻辑运算
if [ $count -gt 5 ] && [ $count -lt 20 ]; then
    echo "Count is between 5 and 20"
fi

if [ $count -lt 5 ] || [ $count -gt 20 ]; then
    echo "Count is outside range 5-20"
fi

# 使用test命令
if test -f "file.txt"; then
    echo "File exists"
fi

# 双括号（更强大的算术比较）
if (( count > 5 && count < 20 )); then
    echo "Count is between 5 and 20"
fi
```

### 循环结构

```bash
# for循环
for i in 1 2 3 4 5; do
    echo $i
done

# 遍历文件
for file in *.txt; do
    echo "Processing: $file"
done

# C风格for循环
for ((i=0; i<10; i++)); do
    echo $i
done

# while循环
count=0
while [ $count -lt 5 ]; do
    echo $count
    count=$((count + 1))
done

# 读取文件行
while IFS= read -r line; do
    echo "$line"
done < file.txt

# until循环
count=0
until [ $count -ge 5 ]; do
    echo $count
    count=$((count + 1))
done

# break和continue
for i in {1..10}; do
    if [ $i -eq 5 ]; then
        continue  # 跳过5
    fi
    if [ $i -eq 8 ]; then
        break     # 在8处停止
    fi
    echo $i
done
```

### case语句

```bash
# 简单的case语句
read -p "Enter a color: " color
case $color in
    red)
        echo "You chose red"
        ;;
    blue|green)
        echo "You chose blue or green"
        ;;
    *)
        echo "You chose something else"
        ;;
esac

# 复杂的case语句
case $1 in
    start)
        echo "Starting service..."
        ;;
    stop)
        echo "Stopping service..."
        ;;
    restart)
        echo "Restarting service..."
        ;;
    status)
        echo "Checking service status..."
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
```

## 函数编程

### 定义和使用函数

```bash
# 定义函数
greet() {
    echo "Hello, $1!"
}

# 调用函数
greet "Alice"

# 返回值
add() {
    local result=$(($1 + $2))
    echo $result
}

sum=$(add 5 3)
echo "Sum: $sum"

# 返回状态码
check_file() {
    if [ -f "$1" ]; then
        return 0  # 成功
    else
        return 1  # 失败
    fi
}

if check_file "file.txt"; then
    echo "File exists"
else
    echo "File does not exist"
fi

# 局部变量
global_var="I am global"

my_function() {
    local local_var="I am local"
    echo "Inside function: $local_var"
    echo "Inside function: $global_var"
}

my_function
echo "Outside function: $global_var"
# echo "Outside function: $local_var"  # 错误：local_var未定义
```

### 函数参数

```bash
# 处理多个参数
process_args() {
    echo "First argument: $1"
    echo "Second argument: $2"
    echo "All arguments: $@"
    echo "Number of arguments: $#"
    echo "Script name: $0"
}

process_args arg1 arg2 arg3

# 遍历所有参数
iterate_args() {
    for arg in "$@"; do
        echo "Processing: $arg"
    done
}

iterate_args file1.txt file2.txt file3.txt

# shift命令
shift_test() {
    echo "Total arguments: $#"
    echo "First: $1"
    shift
    echo "After shift, first: $1"
    echo "Remaining arguments: $#"
}

shift_test a b c d
```

### 递归函数

```bash
# 阶乘（尾递归）
factorial() {
    local n=$1
    local acc=${2:-1}

    if [ $n -le 1 ]; then
        echo $acc
    else
        factorial $((n - 1)) $((acc * n))
    fi
}

echo "Factorial of 5: $(factorial 5)"

# Fibonacci
fibonacci() {
    local n=$1
    if [ $n -le 1 ]; then
        echo $n
    else
        echo $(( $(fibonacci $((n - 1))) + $(fibonacci $((n - 2))) ))
    fi
}

echo "Fibonacci of 10: $(fibonacci 10)"
```

## 输入输出

### 读取用户输入

```bash
# 简单输入
read -p "Enter your name: " name
echo "Hello, $name!"

# 密码输入（不显示）
read -s -p "Enter password: " password
echo

# 带超时的输入
read -t 5 -p "Enter your choice (5 seconds): " choice
echo "You chose: $choice"

# 读取多个值
read -p "Enter name age: " name age
echo "Name: $name, Age: $age"

# 从文件读取
while IFS= read -r line; do
    echo "Line: $line"
done < input.txt

# 读取确认
read -p "Continue? (y/n): " confirm
if [[ $confirm == [yY] ]]; then
    echo "Continuing..."
else
    echo "Aborting..."
    exit 1
fi
```

### 输出格式化

```bash
# echo选项
echo -n "No newline"  # 不换行
echo -e "Line1\nLine2"  # 解释转义字符
echo "Hello\tWorld"  # 需要配合-e

# printf格式化输出
printf "Name: %s, Age: %d\n" "Alice" 25
printf "Pi: %.2f\n" 3.14159
printf "%-10s %10s\n" "Left" "Right"

# 重定向输出
echo "Error message" >&2  # 输出到stderr
echo "Log message" >> logfile  # 追加到文件

# 管道
echo "Hello World" | tr '[:upper:]' '[:lower:]'

# Here文档
cat << EOF
This is a multi-line
string using Here document.
EOF

# Here字符串
grep "pattern" <<< "This is a string to search"
```

## 命令行参数

### 处理位置参数

```bash
#!/bin/bash
# script.sh

echo "Script name: $0"
echo "First argument: $1"
echo "Second argument: $2"
echo "All arguments: $@"
echo "Number of arguments: $#"

# 检查参数数量
if [ $# -lt 2 ]; then
    echo "Usage: $0 <arg1> <arg2>"
    exit 1
fi
```

### 使用getopts

```bash
#!/bin/bash
# 使用getopts处理选项

usage() {
    echo "Usage: $0 [-a] [-b VALUE] [-c] filename"
    exit 1
}

while getopts ":ab:c" opt; do
    case $opt in
        a)
            echo "Option -a triggered"
            ;;
        b)
            echo "Option -b triggered with value: $OPTARG"
            value=$OPTARG
            ;;
        c)
            echo "Option -c triggered"
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument"
            usage
            ;;
    esac
done

shift $((OPTIND-1))

echo "Remaining arguments: $@"
```

### 使用getopt（更强大）

```bash
#!/bin/bash
# 使用getopt处理长选项

TEMP=$(getopt -o ab:c:: --long alpha,bravo:,charlie:: -n 'example.sh' -- "$@")

if [ $? != 0 ]; then
    echo "Terminating..." >&2
    exit 1
fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        -a|--alpha)
            echo "Option a"
            shift
            ;;
        -b|--bravo)
            echo "Option b, argument '$2'"
            shift 2
            ;;
        -c|--charlie)
            case "$2" in
                "")
                    echo "Option c, no argument"
                    shift 2
                    ;;
                *)
                    echo "Option c, argument '$2'"
                    shift 2
                    ;;
            esac
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done

echo "Remaining arguments:"
for arg in "$@"; do
    echo "  --> '$arg'"
done
```

## 信号处理

### 捕获中断

```bash
#!/bin/bash
# 捕获Ctrl+C

cleanup() {
    echo "Cleaning up..."
    # 删除临时文件等
    rm -f /tmp/my_script_temp*
    exit 1
}

trap cleanup SIGINT SIGTERM

echo "Press Ctrl+C to interrupt..."

for i in {1..100}; do
    echo "Working... $i"
    sleep 1
done
```

### 捕获EXIT信号

```bash
#!/bin/bash
# 确保清理代码总是执行

cleanup() {
    echo "Script is exiting..."
    rm -f /tmp/tempfile
}

trap cleanup EXIT

# 创建临时文件
touch /tmp/tempfile

echo "Doing some work..."
# 即使脚本出错，cleanup也会执行
```

## 文本处理

### 文件操作

```bash
# 读取文件
while IFS= read -r line; do
    echo "$line"
done < file.txt

# 写入文件
echo "Hello" > output.txt
echo "World" >> output.txt

# 检查文件是否存在
if [ -f "file.txt" ]; then
    echo "File exists"
fi

# 检查文件是否可读
if [ -r "file.txt" ]; then
    echo "File is readable"
fi

# 获取文件大小
size=$(wc -c < file.txt)
echo "File size: $size bytes"

# 获取行数
lines=$(wc -l < file.txt)
echo "File lines: $lines"
```

### 文本转换

```bash
# 转换为大写
echo "hello" | tr '[:lower:]' '[:upper:]'

# 删除重复行
sort file.txt | uniq

# 只显示重复行
sort file.txt | uniq -d

# 统计重复次数
sort file.txt | uniq -c

# 替换文本
sed 's/old/new/g' file.txt

# 删除空行
sed '/^$/d' file.txt

# 提取特定列
awk '{print $1, $3}' file.txt

# 按模式分割文件
awk '/pattern/{filename="part_"++count".txt"; print > filename}'
```

## 进程管理

### 后台执行

```bash
# 后台运行
command &

# 后台运行并重定向输出
command > /dev/null 2>&1 &

# 使用nohup（退出终端后继续运行）
nohup command &

# 查看后台任务
jobs

# 带回后台任务
fg %1

# 继续后台任务
bg %1

# 杀死后台任务
kill %1
```

### 进程监控

```bash
# 查看进程
ps aux

# 查找特定进程
ps aux | grep nginx

# 实时监控
top

# 杀死进程
kill PID
kill -9 PID  # 强制杀死

# 等待进程完成
wait PID
```

## 调试技巧

### 调试模式

```bash
#!/bin/bash
# 启用调试模式

set -x  # 在执行前打印命令
set -v  # 打印输入行

# 或者
bash -x script.sh

# 只调试部分代码
set -x  # 开始调试
# 需要调试的代码
set +x  # 结束调试
```

### 错误处理

```bash
#!/bin/bash
# 遇到错误立即退出
set -e

# 使用未定义变量时报错
set -u

# 管道命令失败时退出
set -o pipefail

# 组合使用
set -euo pipefail

# 捕获错误
trap 'echo "Error on line $LINENO"; exit 1' ERR
```

### 日志记录

```bash
#!/bin/bash
# 日志函数

log() {
    local level=$1
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $@" | tee -a script.log
}

log INFO "Script started"
log ERROR "An error occurred"
log WARNING "This is a warning"
```

## 实用示例

### 系统监控脚本

```bash
#!/bin/bash
# 系统监控脚本

while true; do
    clear
    echo "=== System Monitor ==="
    echo "Time: $(date)"
    echo

    # CPU使用率
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}'

    # 内存使用
    echo -e "\nMemory Usage:"
    free -h

    # 磁盘使用
    echo -e "\nDisk Usage:"
    df -h

    sleep 5
done
```

### 日志分析脚本

```bash
#!/bin/bash
# Apache日志分析

log_file="/var/log/apache2/access.log"

echo "=== Top 10 IPs ==="
awk '{print $1}' "$log_file" | sort | uniq -c | sort -rn | head

echo -e "\n=== Top 10 URLs ==="
awk '{print $7}' "$log_file" | sort | uniq -c | sort -rn | head

echo -e "\n=== HTTP Status Codes ==="
awk '{print $9}' "$log_file" | sort | uniq -c | sort -rn
```

### 自动备份脚本

```bash
#!/bin/bash
# 自动备份脚本

SOURCE_DIR="/path/to/source"
BACKUP_DIR="/path/to/backup"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="backup_$DATE.tar.gz"

# 创建备份
echo "Creating backup..."
tar -czf "$BACKUP_DIR/$BACKUP_NAME" "$SOURCE_DIR"

# 删除30天前的备份
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_NAME"
```

## 最佳实践

### 代码风格

1. **使用Shebang**：始终在脚本开头指定解释器
2. **添加注释**：解释复杂逻辑和重要步骤
3. **使用有意义的变量名**：避免单字母变量（除循环变量外）
4. **缩进代码**：使用一致的缩进（通常是4个空格）
5. **引用变量**：始终使用引号包裹变量（"$var"而非$var）

### 安全建议

1. **验证输入**：始终验证用户输入和参数
2. **使用绝对路径**：避免路径混淆
3. **最小权限原则**：只授予必要的权限
4. **清理临时文件**：脚本结束时清理
5. **避免eval**：除非绝对必要，否则不使用eval

### 性能优化

1. **避免外部命令**：尽量使用内置功能
2. **减少子shell**：避免不必要的进程创建
3. **使用管道**：而不是临时文件
4. **批量处理**：一次处理多个项目
5. **缓存结果**：避免重复计算

## 小结

Shell脚本是系统管理和自动化的强大工具。通过本文，你学习了：

1. **基础语法**：变量、控制结构、函数
2. **输入输出**：参数处理、用户交互
3. **文件操作**：读写、文本处理
4. **进程管理**：后台任务、信号处理
5. **调试技巧**：错误处理、日志记录
6. **最佳实践**：代码风格、安全建议

掌握Shell脚本编程，你将能够：
- 自动化重复任务
- 管理系统和服务
- 处理文本和数据
- 创建自定义工具

记住，Shell脚本的关键在于简单和实用。开始时保持简单，随着经验的积累，逐步掌握更高级的技巧。
