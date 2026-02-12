#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名句数据获取、验证和分片存储脚本
从 GitHub 数据源获取名句，去重、验证，并分片存储

数据源：
- 中文古诗词：https://github.com/chinese-poetry/chinese-poetry (336,000+首)
- 国际名句 API：https://api.quotable.io (RESTful API)
- 国际名句 CSV：https://github.com/ShivaliGoel/Quotes-500K (500,000条)
"""

import json
import requests
import os
import hashlib
import time
from typing import List, Dict, Set
from pathlib import Path
import csv
import random


class QuotesManager:
    """名句管理器：负责获取、验证、去重和分片存储"""

    def __init__(self, output_dir: str = "./static/quotes", shard_size: int = 500):
        """
        初始化名句管理器

        Args:
            output_dir: 输出目录
            shard_size: 每个分片的名句数量
        """
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.all_quotes: List[Dict] = []
        self.seen_hashes: Set[str] = set()
        self.seen_content_authors: Set[str] = set()

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def normalize_text(self, text: str) -> str:
        """规范化文本：去除首尾空格、多余空格"""
        return text.strip()

    def generate_quote_hash(self, quote: Dict) -> str:
        """生成名句的唯一哈希值"""
        content = self.normalize_text(quote.get("quote", quote.get("content", "")))
        author = self.normalize_text(quote.get("author", ""))
        hash_input = f"{content}|{author}"
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()

    def is_duplicate(self, quote: Dict) -> bool:
        """检查是否重复"""
        content = self.normalize_text(quote.get("quote", quote.get("content", "")))
        author = self.normalize_text(quote.get("author", ""))

        # 方法1: 内容+作者组合检查
        content_author = f"{content}|{author}"
        if content_author in self.seen_content_authors:
            return True

        # 方法2: 哈希值检查
        quote_hash = self.generate_quote_hash(quote)
        if quote_hash in self.seen_hashes:
            return True

        return False

    def normalize_quote_format(self, quote: Dict) -> Dict:
        """统一名句格式"""
        # 标准化字段名
        normalized = {
            "quote": self.normalize_text(
                quote.get("quote") or quote.get("content") or ""
            ),
            "author": self.normalize_text(
                quote.get("author") or quote.get("authorSlug") or ""
            ),
            "source": self.normalize_text(
                quote.get("source") or quote.get("origin") or quote.get("title") or ""
            ),
            "dynasty": self.normalize_text(
                quote.get("dynasty") or quote.get("category") or ""
            ),
            "tags": quote.get("tags", [])
        }

        # 如果有其他字段也保留
        for key, value in quote.items():
            if key not in normalized and not isinstance(value, (list, dict)):
                normalized[key] = value

        return normalized

    def validate_quote(self, quote: Dict) -> bool:
        """验证名句是否有效"""
        if not quote:
            return False

        content = self.normalize_text(quote.get("quote", quote.get("content", "")))

        # 基本验证
        if len(content) < 5:  # 太短
            return False

        if len(content) > 500:  # 太长
            return False

        # 检查乱码（简单启发式）
        try:
            content.encode('utf-8').decode('utf-8')
        except UnicodeError:
            return False

        # 检查是否有过多特殊字符
        special_chars = sum(1 for c in content if not c.isalnum() and not c.isspace() and not ord(c) > 127)
        if special_chars / len(content) > 0.5:  # 超过50%是特殊字符
            return False

        return True

    def add_quote(self, quote: Dict) -> bool:
        """
        添加名句到集合

        Returns:
            bool: 是否成功添加（false表示重复或无效）
        """
        # 规范化格式
        normalized = self.normalize_quote_format(quote)

        # 验证
        if not self.validate_quote(normalized):
            return False

        # 去重
        if self.is_duplicate(normalized):
            return False

        # 添加到集合
        content = normalized["quote"]
        author = normalized["author"]

        self.seen_hashes.add(self.generate_quote_hash(normalized))
        self.seen_content_authors.add(f"{content}|{author}")
        self.all_quotes.append(normalized)

        return True

    def fetch_from_chinese_poetry_github(self, limit: int = 2000) -> int:
        """
        从 Chinese Poetry GitHub 获取数据

        使用 Raw GitHub URL 直接获取 JSON 数据
        """
        print("📚 从 Chinese Poetry GitHub 获取数据...")
        added_count = 0

        # 使用 jackeyGao 的 fork，数据格式更规范
        base_url = "https://raw.githubusercontent.com/jackeyGao/chinese-poetry/master/json/"

        # 唐诗和宋词数据文件
        collections = [
            "poet.tang.0.json",
            "poet.tang.1.json",
            "poet.tang.2.json",
            "poet.song.0.json",
            "ci.song.980.json",
            "ci.south.json",
        ]

        for collection in collections:
            if added_count >= limit:
                break

            try:
                url = f"{base_url}{collection}"
                print(f"  正在获取: {collection}")
                response = requests.get(url, timeout=15)

                if response.status_code == 200:
                    data = response.json()

                    if isinstance(data, list):
                        for poem in data:
                            if added_count >= limit:
                                break

                            # 从诗词中提取名句（取前两句）
                            paragraphs = poem.get("paragraphs", [])

                            if paragraphs and len(paragraphs) >= 1:
                                # 提取前1-2句作为名句
                                quote_text = paragraphs[0]
                                if len(paragraphs) > 1 and len(paragraphs[0]) < 10:
                                    quote_text = "\n".join(paragraphs[:2])

                                # 如果太长，只取第一句
                                if len(quote_text) > 50:
                                    quote_text = paragraphs[0]

                                quote = {
                                    "quote": quote_text,
                                    "author": poem.get("author", ""),
                                    "source": f"《{poem.get('title', '')}》",
                                    "dynasty": ""
                                }

                                if self.add_quote(quote):
                                    added_count += 1
                                    if added_count % 100 == 0:
                                        print(f"    已添加 {added_count} 条")

                    time.sleep(0.5)

            except Exception as e:
                print(f"    ⚠️  获取失败: {e}")
                continue

        print(f"✅ 从 Chinese Poetry 添加了 {added_count} 条名句")
        return added_count

    def fetch_from_quotable_api(self, limit: int = 1000) -> int:
        """从 Quotable API 获取国际名句"""
        print("🌍 从 Quotable API 获取国际名句...")
        added_count = 0

        base_url = "https://api.quotable.io/quotes"
        params = {"limit": 20, "page": 1}

        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        while added_count < limit:
            try:
                response = requests.get(base_url, params=params, timeout=10, verify=False)
                response.raise_for_status()
                data = response.json()

                quotes = data.get("results", [])

                if not quotes:
                    break

                for item in quotes:
                    if added_count >= limit:
                        break

                    quote = {
                        "quote": item.get("content", ""),
                        "author": item.get("author", ""),
                        "source": item.get("tags", [""])[0] if item.get("tags") else "",
                        "dynasty": "",
                        "tags": item.get("tags", [])
                    }

                    if self.add_quote(quote):
                        added_count += 1
                        if added_count % 100 == 0:
                            print(f"    已添加 {added_count} 条")

                params["page"] += 1
                time.sleep(0.2)

            except Exception as e:
                print(f"    ⚠️  获取失败: {e}")
                break

        print(f"✅ 从 Quotable API 添加了 {added_count} 条名句")
        return added_count

    def fetch_from_gushici_api(self, limit: int = 500) -> int:
        """从 Gushici API 获取古诗词名句"""
        print("🎋 从 Gushici API 获取古诗词名句...")
        added_count = 0

        api_url = "https://v1.jinrishici.com/all.json"

        for i in range(limit):
            try:
                response = requests.get(api_url, timeout=10)
                response.raise_for_status()
                data = response.json()

                quote = {
                    "quote": data.get("content", ""),
                    "author": data.get("author", ""),
                    "source": f"《{data.get('origin', '')}》",
                    "dynasty": "",
                    "category": data.get("category", "")
                }

                if self.add_quote(quote):
                    added_count += 1
                    if added_count % 100 == 0:
                        print(f"    已添加 {added_count} 条")

                time.sleep(0.2)  # 礼貌性延迟

            except Exception as e:
                print(f"    ⚠️  获取失败: {e}")
                break

        print(f"✅ 从 Gushici API 添加了 {added_count} 条名句")
        return added_count

    def load_existing_quotes(self) -> int:
        """加载现有的名句数据"""
        existing_file = self.output_dir / "quotes.json"

        if not existing_file.exists():
            return 0

        print(f"📖 加载现有名句: {existing_file}")

        try:
            with open(existing_file, 'r', encoding='utf-8') as f:
                existing_quotes = json.load(f)

            for quote in existing_quotes:
                normalized = self.normalize_quote_format(quote)
                if self.validate_quote(normalized):
                    self.seen_hashes.add(self.generate_quote_hash(normalized))
                    content = normalized["quote"]
                    author = normalized["author"]
                    self.seen_content_authors.add(f"{content}|{author}")

            print(f"✅ 已加载 {len(self.seen_hashes)} 条现有名句")
            return len(self.seen_hashes)

        except Exception as e:
            print(f"⚠️  加载失败: {e}")
            return 0

    def shuffle_quotes(self):
        """随机打乱名句顺序"""
        print("🔀 随机打乱名句顺序...")
        random.shuffle(self.all_quotes)

    def save_shards(self) -> List[str]:
        """
        分片存储名句

        Returns:
            List[str]: 生成的分片文件名列表
        """
        print(f"💾 开始分片存储（每片 {self.shard_size} 条）...")

        shard_files = []
        total_quotes = len(self.all_quotes)
        num_shards = (total_quotes + self.shard_size - 1) // self.shard_size

        for i in range(num_shards):
            start_idx = i * self.shard_size
            end_idx = min((i + 1) * self.shard_size, total_quotes)
            shard_quotes = self.all_quotes[start_idx:end_idx]

            shard_filename = f"quotes_{i + 1}.json"
            shard_path = self.output_dir / shard_filename

            with open(shard_path, 'w', encoding='utf-8') as f:
                json.dump(shard_quotes, f, ensure_ascii=False, indent=2)

            shard_files.append(shard_filename)
            print(f"  ✅ {shard_filename}: {len(shard_quotes)} 条")

        return shard_files

    def save_index(self, shard_files: List[str], total_quotes: int):
        """保存分片索引文件"""
        index = {
            "version": "1.0",
            "total_quotes": total_quotes,
            "shard_size": self.shard_size,
            "total_shards": len(shard_files),
            "shards": shard_files,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        index_path = self.output_dir / "quotes_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        print(f"✅ 索引文件已保存: {index_path}")

    def generate_all(self, target_count: int = 5000) -> Dict:
        """
        生成完整的名句数据集

        Args:
            target_count: 目标名句数量

        Returns:
            Dict: 生成统计信息
        """
        print("=" * 60)
        print("🚀 开始生成名句数据集")
        print("=" * 60)

        stats = {
            "chinese_poetry": 0,
            "quotable_api": 0,
            "gushici_api": 0,
            "existing": 0,
            "total": 0
        }

        # 1. 加载现有名句
        stats["existing"] = self.load_existing_quotes()

        # 2. 从各个数据源获取名句
        remaining = target_count - stats["existing"]

        if remaining > 0:
            # 按比例分配
            chinese_quota = int(remaining * 0.6)  # 60% 中文
            international_quota = remaining - chinese_quota  # 40% 国际

            stats["chinese_poetry"] = self.fetch_from_chinese_poetry_github(min(chinese_quota, 2000))
            stats["gushici_api"] = self.fetch_from_gushici_api(min(chinese_quota - stats["chinese_poetry"], 500))

            stats["quotable_api"] = self.fetch_from_quotable_api(international_quota)

        # 3. 随机打乱
        self.shuffle_quotes()

        # 4. 分片存储
        shard_files = self.save_shards()

        # 5. 保存索引
        stats["total"] = len(self.all_quotes)
        self.save_index(shard_files, stats["total"])

        # 6. 统计信息
        print("\n" + "=" * 60)
        print("📊 生成完成！统计信息：")
        print("=" * 60)
        print(f"  现有名句: {stats['existing']} 条")
        print(f"  Chinese Poetry: {stats['chinese_poetry']} 条")
        print(f"  Gushici API: {stats['gushici_api']} 条")
        print(f"  Quotable API: {stats['quotable_api']} 条")
        print(f"  " + "-" * 50)
        print(f"  总计: {stats['total']} 条")
        print(f"  分片数: {len(shard_files)} 个文件")
        print(f"  每片大小: {self.shard_size} 条")
        print("=" * 60)

        return stats


def main():
    """主函数"""
    # 配置
    OUTPUT_DIR = "./static/quotes"
    SHARD_SIZE = 500
    TARGET_COUNT = 5000

    # 创建管理器
    manager = QuotesManager(
        output_dir=OUTPUT_DIR,
        shard_size=SHARD_SIZE
    )

    # 生成数据集
    stats = manager.generate_all(target_count=TARGET_COUNT)

    print("\n✨ 完成！可以开始使用了。")


if __name__ == "__main__":
    main()
