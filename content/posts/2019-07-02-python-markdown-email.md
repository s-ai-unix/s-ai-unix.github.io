---
title: "Python���送Markdown格式邮件实战"
date: 2019-07-02T15:19:06+08:00
draft: false
description: "详细介绍如何使用Python发送Markdown格式的邮件,提升邮件可读性"
categories: ["Python", "实用技巧"]
tags: ["Python", "邮件", "Markdown"]
cover:
    image: "/images/python-markdown-email.jpg"
    alt: "Python发送Markdown邮件"
    caption: "用Markdown格式发送邮件,让邮件更专业、更美观"
---

## 前言

在日常工作中,我们经常需要发送邮件来分享数据报告、技术文档或项目进度。纯文本邮件格式单调,富文本邮件又需要复杂的HTML代码。Markdown作为一种轻量级标记语言,既能保持良好的可读性,又能转换为美观的HTML格式。

本文将介绍如何使用Python发送Markdown格式的邮件。

## 为什么使用Markdown发送邮件

### Markdown的优势

1. **简洁易读**:源文件和渲染结果都清晰易读
2. **格式丰富**:支持标题、列表、代码块、表格等
3. **易于维护**:专注于内容,不关心格式
4. **版本控制友好**:纯文本格式,便于Git管理
5. **跨平台**:支持几乎所有平台和编辑器

### 应用场景

- **数据报告**:发送数据分析报告
- **技术文档**:分享技术方案和教程
- **项目更新**:汇报项目进展
- **邮件订阅**:发送定期通讯
- **自动化报告**:定时发送统计报告

## 解决方案

### 方案1:使用mdmail库

[mdmail](https://github.com/yejianye/mdmail)是一个专门用于发送Markdown格式邮件的Python库,简单易用。

#### 安装mdmail

```bash
pip install mdmail
```

#### 基本用法

```python
import mdmail

email = """
# 项目周报

## 本周进展

- 完成了用户认证模块
- 优化了数据库查询性能
- 修复了3个关键bug

## 下周计划

1. 开始支付模块开发
2. 进行性能测试
3. 编写API文档

## 数据统计

| 指标 | 本周 | 上周 | 变化 |
|------|------|------|------|
| 用户数 | 1200 | 1000 | +20% |
| 订单数 | 350 | 300 | +16.7% |

## 代码示例

```python
def greet(name):
    return f"Hello, {name}!"
```

如有问题,请随时联系。
"""

mdmail.send(
    email,
    subject='项目周报 - 2025年第1周',
    from_email='sender@example.com',
    to_email='recipient@example.com'
)
```

#### 高级用法

**使用邮件模板**:

```python
import mdmail

# 读取模板文件
with open('template.md', 'r', encoding='utf-8') as f:
    email_content = f.read()

# 替换变量
email_content = email_content.replace('{{name}}', '张三')
email_content = email_content.replace('{{date}}', '2025-01-11')

# 发送邮件
mdmail.send(
    email_content,
    subject='周报 - 张三',
    from_email='sender@example.com',
    to_email='recipient@example.com',
    cc=['manager@example.com']
)
```

**发送多个收件人**:

```python
recipients = [
    'user1@example.com',
    'user2@example.com',
    'user3@example.com'
]

mdmail.send(
    email_content,
    subject='团队通知',
    from_email='sender@example.com',
    to_email=recipients
)
```

**使用SMTP服务器**:

```python
import mdmail

mdmail.send(
    email_content,
    subject='重要通知',
    from_email='sender@example.com',
    to_email='recipient@example.com',
    smtp_host='smtp.gmail.com',
    smtp_port=587,
    username='your_email@gmail.com',
    password='your_password',
    use_tls=True
)
```

### 方案2:使用markdown2 + email

如果不使用mdmail,我们可以手动将Markdown转换为HTML,然后发送邮件。

#### 安装依赖

```bash
pip install markdown2
```

#### 实现代码

```python
import smtplib
import markdown2
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header

def send_markdown_email(
    markdown_content,
    subject,
    from_email,
    to_email,
    smtp_host='localhost',
    smtp_port=25,
    username=None,
    password=None,
    use_tls=False
):
    """
    发送Markdown格式邮件

    参数:
        markdown_content: Markdown格式的文本
        subject: 邮件主题
        from_email: 发件人邮箱
        to_email: 收件人邮箱
        smtp_host: SMTP服务器地址
        smtp_port: SMTP服务器端口
        username: 用户名
        password: 密码
        use_tls: 是否使用TLS
    """
    # 将Markdown转换为HTML
    html_content = markdown2.markdown(
        markdown_content,
        extras=['tables', 'fenced-code-blocks', 'header-ids']
    )

    # 创建HTML邮件
    msg = MIMEMultipart('alternative')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = from_email
    msg['To'] = to_email

    # 添加纯文本版本(备用)
    part1 = MIMEText(markdown_content, 'plain', 'utf-8')
    msg.attach(part1)

    # 添加HTML版本
    part2 = MIMEText(html_content, 'html', 'utf-8')
    msg.attach(part2)

    # 发送邮件
    smtp = smtplib.SMTP(smtp_host, smtp_port)

    if use_tls:
        smtp.starttls()

    if username and password:
        smtp.login(username, password)

    smtp.sendmail(from_email, [to_email], msg.as_string())
    smtp.quit()

# 使用示例
markdown_text = """
# 本周数据报告

## 关键指标

- 新增用户: **1,234**
- 活跃用户: *5,678*
- 转化率: 12.3%

## 详细数据

| 产品 | 销量 | 增长 |
|------|------|------|
| 产品A | 100 | +10% |
| 产品B | 200 | +15% |

```python
import pandas as pd
data = pd.read_csv('sales.csv')
print(data.describe())
```
"""

send_markdown_email(
    markdown_text,
    subject='数据周报',
    from_email='sender@example.com',
    to_email='recipient@example.com',
    smtp_host='smtp.gmail.com',
    smtp_port=587,
    username='your_email@gmail.com',
    password='your_password',
    use_tls=True
)
```

### 方案3:使用Jinja2模板

对于复杂的邮件模板,可以使用Jinja2进行模板渲染。

#### 安装依赖

```bash
pip install jinja2 markdown2
```

#### 实现代码

```python
import smtplib
from jinja2 import Template
import markdown2
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header

def send_template_email(
    template_path,
    template_vars,
    subject,
    from_email,
    to_email,
    smtp_config
):
    """
    使用Jinja2模板发送邮件

    参数:
        template_path: 模板文件路径
        template_vars: 模板变量字典
        subject: 邮件主题
        from_email: 发件人邮箱
        to_email: 收件人邮箱
        smtp_config: SMTP配置字典
    """
    # 读取模板
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()

    # 渲染模板
    template = Template(template_content)
    markdown_content = template.render(**template_vars)

    # 转换为HTML
    html_content = markdown2.markdown(markdown_content)

    # 创建邮件
    msg = MIMEMultipart('alternative')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = from_email
    msg['To'] = to_email

    # 添加内容
    msg.attach(MIMEText(markdown_content, 'plain', 'utf-8'))
    msg.attach(MIMEText(html_content, 'html', 'utf-8'))

    # 发送邮件
    smtp = smtplib.SMTP(
        smtp_config['host'],
        smtp_config['port']
    )

    if smtp_config.get('use_tls'):
        smtp.starttls()

    if smtp_config.get('username'):
        smtp.login(
            smtp_config['username'],
            smtp_config['password']
        )

    smtp.sendmail(from_email, [to_email], msg.as_string())
    smtp.quit()

# 模板文件 (template.md)
"""
# {{report_title}}

## 报告周期
- 开始日期: {{start_date}}
- 结束日期: {{end_date}}

## 数据概览

{% for metric in metrics %}
- **{{metric.name}}**: {{metric.value}}
{% endfor %}

## 详细数据

| 类别 | 数量 | 占比 |
|------|------|------|
{% for item in details %}
| {{item.category}} | {{item.count}} | {{item.percentage}}% |
{% endfor %}

## 备注

{{notes}}
"""

# 使用示例
template_vars = {
    'report_title': '月度销售报告',
    'start_date': '2025-01-01',
    'end_date': '2025-01-31',
    'metrics': [
        {'name': '总销售额', 'value': '¥1,000,000'},
        {'name': '订单数', 'value': '5,000'},
        {'name': '客单价', 'value': '¥200'}
    ],
    'details': [
        {'category': '线上', 'count': 3000, 'percentage': 60},
        {'category': '线下', 'count': 2000, 'percentage': 40}
    ],
    'notes': '本月销售额较上月增长15%。'
}

smtp_config = {
    'host': 'smtp.gmail.com',
    'port': 587,
    'username': 'your_email@gmail.com',
    'password': 'your_password',
    'use_tls': True
}

send_template_email(
    'template.md',
    template_vars,
    subject='月度销售报告',
    from_email='sender@example.com',
    to_email='recipient@example.com',
    smtp_config=smtp_config
)
```

## 实战案例

### 案例1:自动化数据报告

```python
import pandas as pd
import mdmail
from datetime import datetime, timedelta

def generate_report_email():
    """生成并发送数据报告"""
    # 读取数据
    df = pd.read_csv('daily_data.csv')

    # 计算统计数据
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')

    total_users = df['users'].sum()
    active_users = df[df['status'] == 'active']['users'].sum()
    new_users = df['new_users'].sum()

    # 生成Markdown内容
    email_content = f"""
# 数据日报 - {yesterday_str}

## 核心指标

- 总用户数: **{total_users:,}**
- 活跃用户: *{active_users:,}*
- 新增用户: +{new_users:,}

## 用户增长趋势

```python
import matplotlib.pyplot as plt
df.plot(x='date', y='users')
plt.show()
```

## 详细数据

{df.to_markdown(index=False)}

> 报告自动生成于: {today.strftime('%Y-%m-%d %H:%M:%S')}
"""

    # 发送邮件
    mdmail.send(
        email_content,
        subject=f'数据日报 - {yesterday_str}',
        from_email='report@example.com',
        to_email='manager@example.com'
    )

# 定时任务
if __name__ == '__main__':
    generate_report_email()
```

### 案例2:错误监控邮件

```python
import mdmail
import traceback
from datetime import datetime

def send_error_alert(error_info):
    """发送错误警报"""
    email_content = f"""
# ⚠️ 系统错误警报

## 错误信息

- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **服务**: {error_info['service']}
- **错误类型**: {error_info['error_type']}

## 错误详情

```
{error_info['error_message']}
```

## 堆栈跟踪

```python
{error_info['traceback']}
```

## 影响范围

- 影响用户: {error_info.get('affected_users', '未知')}
- 错误频率: {error_info.get('error_count', 0)} 次/小时

## 处理建议

1. 检查日志: `tail -f {error_info.get('log_file', '/var/log/app.log')}`
2. 重启服务: `systemctl restart {error_info['service']}`
3. 联系负责人: {error_info.get('contact', '技术团队')}

---

此邮件由监控系统自动发送,请及时处理。
"""

    mdmail.send(
        email_content,
        subject=f'⚠️ 错误警报 - {error_info["service"]}',
        from_email='monitor@example.com',
        to_email='ops@example.com'
    )

# 使用示例
try:
    # 你的代码
    pass
except Exception as e:
    error_info = {
        'service': 'api-server',
        'error_type': type(e).__name__,
        'error_message': str(e),
        'traceback': traceback.format_exc(),
        'affected_users': 150,
        'error_count': 5,
        'log_file': '/var/log/api/server.log',
        'contact': '张三 (zhangsan@example.com)'
    }
    send_error_alert(error_info)
```

### 案例3:周报自动化

```python
import mdmail
from datetime import datetime, timedelta

def generate_weekly_report(tasks_data):
    """生成周报"""
    # 获取本周日期范围
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)

    week_str = f"{week_start.strftime('%Y-%m-%d')} 至 {week_end.strftime('%Y-%m-%d')}"

    # 统计任务
    completed = [t for t in tasks_data if t['status'] == 'completed']
    in_progress = [t for t in tasks_data if t['status'] == 'in_progress']
    pending = [t for t in tasks_data if t['status'] == 'pending']

    # 生成Markdown内容
    content = f"""
# 项目周报
**报告周期**: {week_str}

## 📊 本周概览

- **完成任务**: {len(completed)} 个
- **进行中**: {len(in_progress)} 个
- **待开始**: {len(pending)} 个
- **完成率**: {len(completed)/len(tasks_data)*100:.1f}%

## ✅ 已完成任务

{% for task in completed %}
{{loop.index}}. **{{task['title']}}**
   - 负责人: {{task['owner']}}
   - 完成日期: {{task['completed_date']}}
   - 备注: {{task.get('note', '无')}}
{% endfor %}

## 🚧 进行中任务

{% for task in in_progress %}
{{loop.index}}. **{{task['title']}}**
   - 负责人: {{task['owner']}}
   - 进度: {{task.get('progress', 0)}}%
   - 预计完成: {{task.get('due_date', '待定')}}
{% endfor %}

## 📋 下周计划

{% for task in pending %}
{{loop.index}}. **{{task['title']}}**
   - 优先级: {{task.get('priority', '中')}}
   - 预计工期: {{task.get('estimate', '未知')}}
{% endfor %}

## 💬 备注

- 本周团队协作良好,项目进展顺利
- 需要关注:{{tasks_data[0]['title']}} 的进度
- 下周重点:推进{{pending[0]['title']}}项目

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    mdmail.send(
        content,
        subject=f'项目周报 - {week_str}',
        from_email='pm@example.com',
        to_email='team@example.com',
        cc=['manager@example.com']
    )

# 使用示例
tasks = [
    {
        'title': '用户认证模块',
        'status': 'completed',
        'owner': '张三',
        'completed_date': '2025-01-08',
        'note': '按计划完成'
    },
    {
        'title': '支付接口对接',
        'status': 'in_progress',
        'owner': '李四',
        'progress': 60,
        'due_date': '2025-01-15'
    },
    {
        'title': '数据报表功能',
        'status': 'pending',
        'priority': '高',
        'estimate': '5天'
    }
]

generate_weekly_report(tasks)
```

## 最佳实践

### 1. 邮件模板管理

```
project/
├── templates/
│   ├── weekly_report.md
│   ├── daily_alert.md
│   └── monthly_summary.md
├── scripts/
│   ├── send_report.py
│   └── send_alert.py
└── config/
    └── email_config.py
```

### 2. 配置管理

```python
# config/email_config.py
EMAIL_CONFIG = {
    'smtp_host': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': 'your_email@gmail.com',
    'password': 'your_password',
    'use_tls': True,
    'default_sender': 'noreply@example.com'
}

# 使用
from config.email_config import EMAIL_CONFIG
```

### 3. 错误处理

```python
def send_email_safely(content, subject, to_email):
    """安全发送邮件"""
    try:
        mdmail.send(
            content,
            subject=subject,
            from_email=EMAIL_CONFIG['default_sender'],
            to_email=to_email
        )
        return True
    except Exception as e:
        print(f"邮件发送失败: {e}")
        # 记录日志
        logger.error(f"Email send failed: {e}")
        return False
```

### 4. 测试

```python
# 测试时发送到测试邮箱
TEST_MODE = True
TEST_EMAIL = 'test@example.com'

def send_email(content, subject, to_email):
    if TEST_MODE:
        to_email = TEST_EMAIL
        subject = f'[TEST] {subject}'

    mdmail.send(content, subject, from_email='sender@example.com', to_email=to_email)
```

## 常见问题

### 1. 邮件被标记为垃圾邮件

**解决方法**:
- 使用专业的SMTP服务(SendGrid, Mailgun等)
- 设置正确的SPF、DKIM记录
- 避免频繁发送相同内容

### 2. 中文乱码

**解决方法**:
```python
# 确保使用UTF-8编码
msg = MIMEText(content, 'html', 'utf-8')
msg['Subject'] = Header(subject, 'utf-8')
```

### 3. 附件支持

```python
from email.mime.application import MIMEApplication

# 添加附件
with open('report.pdf', 'rb') as f:
    part = MIMEApplication(f.read(), Name='report.pdf')
part['Content-Disposition'] = 'attachment; filename="report.pdf"'
msg.attach(part)
```

## 总结

使用Python发送Markdown格式邮件可以大大提升邮件的可读性和专业度。主要方法包括:

1. **使用mdmail库**:最简单的方式,适合快速实现
2. **手动转换**:使用markdown2等库,更灵活
3. **使用模板**:适合复杂场景和批量发送

> 实践建议:对于简单的邮件发送,使用mdmail;对于复杂的业务场景,使用Jinja2模板。记住做好错误处理和日志记录。
