#!/bin/bash

# 生成包含 shards 字段的索引文件
cd "$(dirname "$0")/static/quotes" || exit 1

# 获取所有分片文件名（按数字排序）
shards=$(ls quotes_[0-9]*.json 2>/dev/null | sort -V | tr '\n' ' ' | sed 's/ /", "/g' | sed 's/", "$//' | sed 's/^"/["/' | sed 's/$/"]/')

if [ "$shards" = "[]" ]; then
    echo "错误：没有找到分片文件"
    exit 1
fi

# 计算总数量（从所有分片文件中统计）
total=$(cat quotes_[0-9]*.json 2>/dev/null | jq 'length' | awk '{s+=$1} END {print s}')

# 生成索引文件
cat > quotes_index.json << IDX_EOF
{
  "total_quotes": $total,
  "total_shards": $(ls quotes_[0-9]*.json 2>/dev/null | wc -l | tr -d ' '),
  "quotes_per_shard": 1000,
  "last_updated": "$(date +%Y-%m-%d)",
  "shards": $(ls quotes_[0-9]*.json 2>/dev/null | sort -V | jq -R -s -c 'split("\n")[:-1]'),
  "history": [
    {
      "date": "2026-01-20",
      "action": "修复索引文件，添加 shards 字段"
    }
  ]
}
IDX_EOF

echo "✓ 索引文件已更新"
echo "  总名句数: $total"
echo "  分片数: $(ls quotes_[0-9]*.json 2>/dev/null | wc -l | tr -d ' ')"
cat quotes_index.json | jq '.shards | length'
