#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并现有名言和新提取的古籍名言，并生成分片文件
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import random


def load_existing_quotes(quotes_dir: Path) -> List[Dict]:
    """加载现有的名言数据"""
    all_quotes = []

    # 加载所有分片文件
    for i in range(1, 11):  # quotes_1.json 到 quotes_10.json
        file_path = quotes_dir / f'quotes_{i}.json'
        if not file_path.exists():
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            quotes = json.load(f)
            all_quotes.extend(quotes)

    print(f"加载现有名言: {len(all_quotes)} 条")
    return all_quotes


def load_new_quotes(file_path: Path) -> List[Dict]:
    """加载新提取的古籍名言"""
    with open(file_path, 'r', encoding='utf-8') as f:
        quotes = json.load(f)

    print(f"加载新提取名言: {len(quotes)} 条")
    return quotes


def normalize_new_quote(quote: Dict) -> Dict:
    """标准化新名言的格式"""
    # 从 source 中提取书名和章节
    source = quote.get('source', '')
    author = quote.get('author', quote.get('dynasty', '先秦'))

    # 尝试从 source 中提取更合适的作者名
    # 《老子·道经·第一章》 -> 作者: 老子
    if '《' in source and '·' in source:
        book_name = source.split('《')[1].split('·')[0]
        # 使用书名作为作者，除非是具体的作者名
        if book_name in ['老子', '庄子', '论语', '孟子']:
            author = book_name
        elif book_name in ['史记']:
            author = '司马迁'
        elif book_name in ['资治通鉴', '续资治通鉴']:
            author = '司马光'
        elif book_name in ['孙子兵法']:
            author = '孙武'
        elif book_name in ['孙膑兵法']:
            author = '孙膑'
        elif book_name in ['礼记']:
            author = '戴圣'
        elif book_name in ['淮南子']:
            author = '刘安'
        elif book_name in ['韩非子']:
            author = '韩非'
        elif book_name in ['荀子']:
            author = '荀子'
        elif book_name in ['墨子']:
            author = '墨子'
        elif book_name in ['列子']:
            author = '列子'
        elif book_name in ['吕氏春秋']:
            author = '吕不韦'
        elif book_name in ['左传']:
            author = '左丘明'
        elif book_name in ['周易', '易经', '易传']:
            author = '周文王'
        else:
            author = book_name

    return {
        'quote': quote.get('quote', ''),
        'author': author,
        'source': source,
        'dynasty': quote.get('dynasty', ''),
        'tags': []
    }


def remove_duplicates(all_quotes: List[Dict]) -> List[Dict]:
    """去重"""
    seen = {}
    unique_quotes = []

    for quote in all_quotes:
        quote_text = quote['quote'].strip()

        if quote_text in seen:
            continue

        seen[quote_text] = quote
        unique_quotes.append(quote)

    removed = len(all_quotes) - len(unique_quotes)
    if removed > 0:
        print(f"去重: 移除了 {removed} 条重复名言")

    return unique_quotes


def shuffle_quotes(quotes: List[Dict]) -> List[Dict]:
    """随机打乱名言顺序"""
    random.shuffle(quotes)
    return quotes


def shard_quotes(quotes: List[Dict], shard_size: int = 1000) -> List[List[Dict]]:
    """将名言分片"""
    shards = []
    for i in range(0, len(quotes), shard_size):
        shard = quotes[i:i + shard_size]
        shards.append(shard)

    return shards


def save_shards(shards: List[List[Dict]], output_dir: Path):
    """保存分片文件"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存分片文件
    for i, shard in enumerate(shards, 1):
        file_path = output_dir / f'quotes_{i}.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(shard, f, ensure_ascii=False, indent=2)

        print(f"保存: {file_path.name} ({len(shard)} 条)")

    # 生成索引文件
    index = {
        'total_shards': len(shards),
        'total_quotes': sum(len(shard) for shard in shards),
        'shards': [f'quotes_{i}.json' for i in range(1, len(shards) + 1)],
        'generated_at': '2026-01-20'
    }

    index_file = output_dir / 'quotes_index.json'
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n索引文件: {index_file.name}")
    print(f"总分片数: {index['total_shards']}")
    print(f"总名言数: {index['total_quotes']}")


def main():
    # 目录路径
    quotes_dir = Path('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/quotes')
    new_quotes_file = quotes_dir / 'classical_quotes_new.json'

    # 加载现有名言
    existing_quotes = load_existing_quotes(quotes_dir)

    # 加载新提取的名言
    new_quotes_raw = load_new_quotes(new_quotes_file)

    # 标准化新名言格式
    new_quotes = [normalize_new_quote(q) for q in new_quotes_raw]
    print(f"标准化后新名言: {len(new_quotes)} 条")

    # 合并
    all_quotes = existing_quotes + new_quotes
    print(f"合并前总计: {len(all_quotes)} 条")

    # 去重
    all_quotes = remove_duplicates(all_quotes)
    print(f"去重后总计: {len(all_quotes)} 条")

    # 随机打乱
    all_quotes = shuffle_quotes(all_quotes)

    # 分片
    shards = shard_quotes(all_quotes, shard_size=1000)
    print(f"\n生成 {len(shards)} 个分片")

    # 保存分片文件
    save_shards(shards, quotes_dir)

    print("\n完成！")


if __name__ == '__main__':
    main()
