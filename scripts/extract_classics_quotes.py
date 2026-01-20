#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取经典文本中的精华名言警句
从 chinese-ancient-text 仓库中提取《菜根谭》、《世说新语》、《了凡四训》等经典
"""

import json
import re
from pathlib import Path
from typing import List, Dict


def is_valid_quote(text: str) -> bool:
    """验证文本是否是有效的名言警句"""
    if not text or len(text.strip()) < 8:
        return False
    if len(text.strip()) > 150:
        return False
    # 过滤掉明显不是名言的内容
    if re.match(r'^[第0-9一二三四五六七八九十百千]+[章节卷篇]', text):
        return False
    return True


def extract_quotes_from_content(content: str, book_name: str, chapter_title: str = "") -> List[Dict]:
    """从内容中提取名言警句"""
    quotes = []

    # 按句子分割（保留标点）
    sentences = re.split(r'([。！？；])', content)

    # 重组句子（将标点符号加回句子）
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
        else:
            sentence = sentences[i]

        sentence = sentence.strip()
        if not sentence:
            continue

        # 如果句子太短，尝试合并下一句
        if len(sentence) < 10 and i + 2 < len(sentences):
            next_sentence = sentences[i + 2].strip()
            if next_sentence:
                combined = sentence + next_sentence
                if is_valid_quote(combined):
                    quotes.append({
                        "text": combined,
                        "source": book_name,
                        "chapter": chapter_title,
                        "dynasty": "",
                        "author": ""
                    })
                continue

        if is_valid_quote(sentence):
            quotes.append({
                "text": sentence,
                "source": book_name,
                "chapter": chapter_title,
                "dynasty": "",
                "author": ""
            })

    return quotes


def process_book_json(json_path: Path, book_name: str, dynasty: str = "", author: str = "") -> List[Dict]:
    """处理单本书籍的JSON文件"""
    print(f"\n处理书籍: {book_name}")
    quotes = []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查是否是 chinese-ancient-text 格式
    if 'articles' in data:
        articles = data['articles']
        print(f"  章节数: {len(articles)}")

        for item in articles:
            if 'content' not in item:
                continue

            content_array = item['content']
            chapter_title = item.get('title', '')

            # content 是一个数组，每个元素是一句话
            for sentence in content_array:
                sentence = sentence.strip()
                if is_valid_quote(sentence):
                    quotes.append({
                        "text": sentence,
                        "source": book_name,
                        "chapter": chapter_title,
                        "dynasty": dynasty,
                        "author": author
                    })
    else:
        # 其他格式
        print(f"  章节数: {len(data)}")

        for item in data:
            if 'paragraphs' in item:
                content = ''.join(item['paragraphs'])
            elif 'content' in item:
                content = item['content']
            else:
                continue

            chapter_title = item.get('chapterTitle', item.get('title', ''))

            # 提取名言
            book_quotes = extract_quotes_from_content(content, book_name, chapter_title)
            quotes.extend(book_quotes)

    print(f"  提取名言数: {len(quotes)}")
    return quotes


def main():
    # 定义要处理的书籍
    books = [
        {"file": "菜根谭.json", "name": "菜根谭", "dynasty": "明代", "author": "洪应明"},
        {"file": "世说新语.json", "name": "世说新语", "dynasty": "南北朝", "author": "刘义庆"},
        {"file": "了凡四训.json", "name": "了凡四训", "dynasty": "明代", "author": "袁了凡"},
        {"file": "围炉夜话.json", "name": "围炉夜话", "dynasty": "清代", "author": "王永彬"},
        {"file": "颜氏家训.json", "name": "颜氏家训", "dynasty": "南北朝", "author": "颜之推"},
        {"file": "孙子兵法.json", "name": "孙子兵法", "dynasty": "春秋", "author": "孙武"},
        {"file": "三十六计.json", "name": "三十六计", "dynasty": "", "author": ""},
        {"file": "冰鉴.json", "name": "冰鉴", "dynasty": "清代", "author": "曾国藩"},
    ]

    source_dir = Path("/tmp/chinese-ancient-text")
    all_quotes = []

    for book in books:
        json_path = source_dir / book["file"]
        if not json_path.exists():
            print(f"⚠️  文件不存在: {book['file']}")
            continue

        quotes = process_book_json(
            json_path,
            book["name"],
            book["dynasty"],
            book["author"]
        )
        all_quotes.extend(quotes)

    # 去重
    seen = set()
    unique_quotes = []
    for quote in all_quotes:
        text = quote["text"]
        if text not in seen:
            seen.add(text)
            unique_quotes.append(quote)

    print(f"\n{'='*50}")
    print(f"总共提取名言: {len(all_quotes)}")
    print(f"去重后名言: {len(unique_quotes)}")
    print(f"{'='*50}")

    # 保存到 JSON 文件
    output_file = Path("/tmp/classics_quotes.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_quotes, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 名言已保存到: {output_file}")

    # 显示一些示例
    print("\n📝 示例名言:")
    for quote in unique_quotes[:10]:
        print(f"  - {quote['text']}")
        if quote['source']:
            print(f"    出处: {quote['source']}", end="")
            if quote['author']:
                print(f" ({quote['author']})", end="")
            print()

    return unique_quotes


if __name__ == "__main__":
    main()
