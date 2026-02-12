---
title: "Linux系统管理速查手册：常用命令与问题排查"
date: 2019-06-06T14:30:00+08:00
draft: false
tags:
  - 系统管理
  - 命令行
description: "Linux系统管理员必备速查手册，涵盖Node.js安装、时区配置、内存监控等常用操作技巧与问题排查方法。"
cover:
  image: "images/covers/1551434678-e076c223a692.jpg"
  alt: "Linux系统管理"
  caption: "Linux系统管理：命令与实践手册"
---

在日常的系统管理工作中，我们经常需要处理各种常见的配置和监控任务。本文整理了Linux系统管理中最常用的操作命令，包括软件安装、时区配置和资源监控，帮助你快速定位和解决问题。

## Node.js安装与配置

Node.js是现代Web开发中不可或缺的运行时环境。在CentOS系统上，我们可以通过NodeSource官方源快速安装最新版本。

### 使用NodeSource安装Node.js

NodeSource提供了Node.js的官方RPM包，确保我们能够获得最新的稳定版本。

#### 安装步骤

```bash
# 1. 添加NodeSource仓库（以Node.js 8.x为例）
curl --silent --location https://rpm.nodesource.com/setup_8.x | sudo bash -

# 2. 使用yum安装Node.js
sudo yum -y install nodejs

# 3. 验证安装
node -v
npm -v
```

#### 版本选择

根据项目需求选择合适的Node.js版本：

- **LTS版本**：生产环境推荐使用，长期支持
- **Current版本**：最新特性，适合开发测试

```bash
# Node.js 16.x LTS
curl --silent --location https://rpm.nodesource.com/setup_16.x | sudo bash -

# Node.js 18.x LTS
curl --silent --location https://rpm.nodesource.com/setup_18.x | sudo bash -
```

### 安装后配置

```bash
# 配置npm国内镜像源（加速包下载）
npm config set registry https://registry.npmmirror.com

# 全局安装常用工具
npm install -g pm2           # 进程管理器
npm install -g yarn          # 包管理工具
npm install -g npx           # 包执行器
```

## 系统时区配置

正确的时区配置对于日志记录、定时任务和系统监控至关重要。Linux系统使用`timedatectl`命令来管理系统时区和时间设置。

### 查看当前时区状态

```bash
# 查看详细的时间和日期状态
timedatectl

# 输出示例：
#      Local time: 三 2019-06-06 14:33:59 CST
#  Universal time: 三 2019-06-06 06:33:59 UTC
#        RTC time: 三 2019-06-06 06:33:59
#       Time zone: Asia/Shanghai (CST, +0800)
#     NTP enabled: yes
#NTP synchronized: yes
# RTC in local TZ: no
```

### 列出所有可用时区

```bash
# 列出所有时区
timedatectl list-timezones

# 筛选特定地区时区
timedatectl list-timezones | grep Asia

# 常用时区：
# Asia/Shanghai      - 中国标准时间 (UTC+8)
# Asia/Tokyo         - 日本标准时间 (UTC+9)
# Asia/Hong_Kong     - 香港时间 (UTC+8)
# America/New_York   - 美国东部时间
# Europe/London      - 格林威治时间
```

### 修改系统时区

```bash
# 设置时区为上海时间
sudo timedatectl set-timezone Asia/Shanghai

# 设置时区为UTC
sudo timedatectl set-timezone UTC

# 验证时区修改
timedatectl
date
```

### 时间同步配置

```bash
# 启用自动时间同步（NTP）
sudo timedatectl set-ntp true

# 禁用自动时间同步
sudo timedatectl set-ntp false

# 手动同步时间（如果使用chrony）
sudo chronyc sources -v
```

### 传统时区设置方法

如果你的系统使用传统的时区配置方式：

```bash
# 查看当前时区
date +%Z

# 修改时区（传统方法）
sudo ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# 编辑时区配置文件
sudo vim /etc/sysconfig/clock
# 添加内容：
# ZONE="Asia/Shanghai"
```

## 内存监控与问题排查

内存管理是系统性能优化的关键环节。掌握内存使用情况的监控和排查技巧，能够帮助我们及时发现和解决性能瓶颈。

### 查看整体内存使用情况

```bash
# 显示内存使用概览
free -h

# 输出示例：
#               total        used        free      shared  buff/cache   available
# Mem:           7.6G        2.1G        3.2G        234M        2.3G        5.1G
# Swap:          2.0G          0B        2.0G

# 持续监控内存使用（每2秒更新）
free -hs 2

# 显示更详细的内存信息
cat /proc/meminfo
```

### 进程内存占用排序

查看哪些进程占用了最多内存，并按占用率降序排列：

```bash
# 显示所有占用内存的进程，并按内存使用率排序
ps -o pid,user,%mem,command ax | awk '($3 > 0){print}' | sort -b -k3 -r

# 输出示例：
#  PID USER     %MEM COMMAND
# 1234 root      5.2 /usr/bin/python3 /usr/bin/ansible
# 5678 nginx     3.8 nginx: worker process
# 9012 mysql     12.5 /usr/sbin/mysqld
```

### 创建便捷的内存检查命令

将常用命令设置为别名，方便日常使用：

```bash
# 添加到 ~/.bashrc 或 ~/.bash_profile
alias memcheck="ps -o pid,user,%mem,command ax | awk '(\$3 > 0){print}' | sort -b -k3 -r"

# 重新加载配置
source ~/.bashrc

# 使用别名快速查看
memcheck
```

### 高级内存监控工具

#### top命令

```bash
# 启动交互式监控工具
top

# 在top中按以下键进行排序：
# M - 按内存使用率排序
# P - 按CPU使用率排序
# q - 退出

# 直接显示内存排序的结果
top -b -n 1 -o %MEM | head -n 20
```

#### htop命令（需要安装）

```bash
# 安装htop
sudo yum install htop

# 启动htop（更友好的界面）
htop

# 功能特点：
# - 彩色显示，更直观
# - 支持鼠标操作
# - 可纵向/横向滚动
# - 支持杀死进程
```

### 内存占用详细分析

```bash
# 查看特定进程的内存详情
ps -p 1234 -o pid,ppid,cmd,%mem,%cpu

# 查看所有进程的完整内存信息
ps aux --sort=-%mem | head -20

# 使用smem工具查看更准确的内存占用（需要安装）
sudo yum install smem
sudo smem -k -s memory

# 查看共享内存使用情况
ipcs -m
```

### 内存泄漏排查

当怀疑某个进程存在内存泄漏时：

```bash
# 持续监控特定进程的内存使用
watch -n 5 "ps -p 1234 -o pid,ppid,cmd,%mem,%cpu,vsz,rss"

# 记录内存使用日志（每分钟记录一次）
while true; do
    echo "$(date): $(ps -p 1234 -o %mem,rss)" >> mem_log.txt
    sleep 60
done

# 分析内存变化趋势
cat mem_log.txt | awk '{print $2, $3}'
```

### 系统内存优化建议

```bash
# 清理页面缓存（需要root权限）
sudo sync && sudo sysctl -w vm.drop_caches=3

# 查看内存相关内核参数
sysctl -a | grep memory

# 调整swappiness值（0-100，值越小越倾向于使用RAM）
sudo sysctl vm.swappiness=10
# 持久化配置
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
```

## 实用技巧总结

### 日常检查清单

```bash
# 1. 系统基础信息检查
echo "=== 系统时间 ===" && timedatectl
echo "=== 内存使用 ===" && free -h
echo "=== CPU负载 ===" && uptime

# 2. 资源占用Top进程
echo "=== 内存占用Top5 ===" && ps -o pid,user,%mem,command ax | awk '($3 > 0){print}' | sort -b -k3 -r | head -5
echo "=== CPU占用Top5 ===" && ps -o pid,user,%cpu,command ax | awk '($3 > 0){print}' | sort -b -k3 -r | head -5

# 3. 磁盘使用情况
echo "=== 磁盘使用 ===" && df -h
```

### 快速排查命令集合

```bash
# 环境检查
node --version           # Node.js版本
npm --version            # npm版本
timedatectl              # 时间时区状态

# 资源监控
free -h                  # 内存概览
df -h                    # 磁盘使用
du -sh *                 # 当前目录大小
top -o %MEM              # 内存排序

# 进程管理
ps aux | grep name       # 查找进程
kill -9 PID              # 强制结束进程
systemctl status service # 服务状态
```

### 配置文件位置速查

```bash
# Node.js相关
~/.npmrc                 # npm配置文件
~/.bashrc                # 环境变量配置

# 时区配置
/etc/localtime           # 当前时区链接
/usr/share/zoneinfo/     # 时区数据文件

# 系统配置
/etc/sysctl.conf         # 内核参数配置
/etc/systemd/system/     # systemd服务配置
```

## 故障排查流程

当遇到系统问题时，按照以下流程进行排查：

1. **检查时间和时区**
   ```bash
   timedatectl
   date
   ```

2. **检查资源使用情况**
   ```bash
   free -h
   df -h
   top
   ```

3. **查看系统日志**
   ```bash
   journalctl -xe
   tail -f /var/log/messages
   ```

4. **检查服务状态**
   ```bash
   systemctl status服务名
   systemctl list-units --failed
   ```

## 结语

Linux系统管理是一个需要不断积累经验的领域。本文整理的命令和技巧涵盖了日常工作中最常见的场景，建议将常用的命令保存为别名或脚本，提高工作效率。

记住这些关键点：

- 使用官方源安装软件，确保安全性和稳定性
- 正确的时区配置对日志和定时任务至关重要
- 定期监控内存使用，及时发现性能问题
- 建立自己的命令速查清单，不断积累经验

通过掌握这些基础操作，你将能够更自信地应对各种系统管理任务。
