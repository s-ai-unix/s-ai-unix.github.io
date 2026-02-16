# AI 生图配置经验总结

## 问题背景

对比 Jimeng 和 Gemini Flash 生成纽约客风格插图时，遇到了多个配置问题，耗费了大量调试时间。

## 关键发现

### 1. API 网关选择

| 网关 | 状态 | 说明 |
|------|------|------|
| `jimeng.jianying.com` | ❌ 不可用 | 官方 API，返回非 JSON 响应 |
| `newapi.aisonnet.org` | ✅ 可用 | 第三方网关，支持多模型 |

### 2. 正确配置（aisonnet 网关）

```python
# API 配置
API_KEY = "sk-S51NEPFTWvJmyQE5oiZp21BruJxV7APdH28zRsiRimSOKjcW"
API_URL = "https://newapi.aisonnet.org/v1/chat/completions"

# 可用模型
JIMENG_MODEL = "jimeng-4.5"      # Jimeng 官方模型
GEMINI_MODEL = "gemini-2.5-flash-image"  # Google Gemini
```

### 3. 请求格式

aisonnet 网关使用 **chat completions** 格式：

```python
data = {
    "extra_body": {"imageConfig": {"aspectRatio": "16:9"}},
    "model": "jimeng-4.5",  # 或 "gemini-2.5-flash-image"
    "messages": [
        {
            "role": "system",
            "content": '{"imageConfig": {"aspectRatio": "16:9"}}',
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ],
    "max_tokens": 150,
    "temperature": 0.7,
}
```

### 4. 响应格式

响应包含 markdown 格式的图片 URL：

```
![g2pimage](https://...)
```

需要用正则表达式提取：

```python
url_match = re.search(r"https?://[^\s\)]+", content)
```

## 超时时间建议

| 模型 | 超时时间 | 说明 |
|------|----------|------|
| jimeng-4.5 | 180s | 生成较慢 |
| gemini-2.5-flash-image | 120s | 生成较快 |

## 纽约客风格提示词模板

```python
visual_strategy = "A minimalist editorial illustration showing [描述场景]..."

style_prompt = (
    "Black and white pen and ink sketch, loose confident strokes, "
    "intentional white space, New Yorker magazine illustration style, "
    "minimalist, elegant"
)

full_prompt = f"Generate an image: {visual_strategy}. {style_prompt}. Chinese caption at bottom: {caption}"
```

## 快速生图脚本模板

```python
#!/usr/bin/env python3
import requests
import re
from pathlib import Path

API_KEY = "sk-S51NEPFTWvJmyQE5oiZp21BruJxV7APdH28zRsiRimSOKjcW"
API_URL = "https://newapi.aisonnet.org/v1/chat/completions"

def generate_image(model, prompt, aspect_ratio="16:9", timeout=180):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "extra_body": {"imageConfig": {"aspectRatio": aspect_ratio}},
        "model": model,
        "messages": [
            {"role": "system", "content": f'{{"imageConfig": {{"aspectRatio": "{aspect_ratio}"}}}}'},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        "max_tokens": 150,
        "temperature": 0.7,
    }
    response = requests.post(API_URL, headers=headers, json=data, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    url_match = re.search(r"https?://[^\\s\\)]+", content)
    if url_match:
        return url_match.group(0)
    raise ValueError(f"未找到图片 URL: {content}")

# 使用示例
visual_strategy = "A digital meter with flowing data tokens"
style_prompt = "Black and white pen and ink sketch, New Yorker magazine style"
caption = "文章标题"
full_prompt = f"Generate an image: {visual_strategy}. {style_prompt}. Chinese caption at bottom: {caption}"

# 生成图片
image_url = generate_image("jimeng-4.5", full_prompt)

# 下载保存
output_path = Path("static/images/plots/my-image.png")
img_data = requests.get(image_url, timeout=60).content
output_path.write_bytes(img_data)
print(f"✅ 已保存: {output_path}")
```

## 常见错误

### 错误 1: JSONDecodeError
```
simplejson.errors.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```
**原因**: 使用了 jimeng 官方 API，返回非 JSON 格式
**解决**: 改用 aisonnet 网关

### 错误 2: 503 Service Unavailable
```
503 Server Error: Service Unavailable
```
**原因**: 网关限流或暂时不可用
**解决**: 等待一段时间后重试

### 错误 3: 524 Timeout
```
ReadTimeout: HTTPConnectionPool: Read timed out.
```
**原因**: jimeng-4.5 生成时间较长
**解决**: 增加超时时间到 180s

## 模型风格对比

| 模型 | 风格特点 | 适用场景 |
|------|----------|----------|
| jimeng-4.5 | 极简、留白多、构图简洁 | 技术文章、学术配图 |
| gemini-2.5-flash-image | 细节丰富、画面饱满 | 封面图、宣传图 |

## 文件位置

- 脚本: `scripts/compare_jimeng_gemini_mingpt.py`
- 图片输出: `static/images/plots/`
- 文档: `docs/JIMENG-IMAGE-GEN.md`

## 更新日志

- 2026-02-16: 首次整理，确认 aisonnet 网关 + jimeng-4.5 配置可用
