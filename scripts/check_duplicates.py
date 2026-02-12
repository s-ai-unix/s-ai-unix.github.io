#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去重检查工具 - 检测名句数据中的重复项
"""

import json
import yaml
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict


class DuplicateChecker:
    """重复检查器"""

    def __init__(self):
        self.duplicates = []
        self.stats = defaultdict(int)

    def check_duplicates(self, quotes: List[Dict]) -> bool:
        """
        检查重复名句
        返回 True 表示没有重复
        """
        # 基于名句文本的重复检查
        quote_map = defaultdict(list)
        quote_hash_map = defaultdict(list)

        for idx, quote in enumerate(quotes, 1):
            quote_text = quote.get('quote', '').strip()
            if not quote_text:
                continue

            # 原始文本
            quote_map[quote_text].append((idx, quote))

            # MD5 哈希（用于检测相似但不完全相同的文本）
            text_hash = hashlib.md5(quote_text.encode('utf-8')).hexdigest()
            quote_hash_map[text_hash].append((idx, quote))

        # 检查完全重复
        for text, occurrences in quote_map.items():
            if len(occurrences) > 1:
                self.duplicates.append({
                    'type': 'exact',
                    'text': text,
                    'count': len(occurrences),
                    'indices': [idx for idx, _ in occurrences],
                    'entries': occurrences
                })
                self.stats['exact_duplicates'] += 1

        # 检查可能相似的内容（基于规范化后的文本）
        normalized_map = defaultdict(list)
        for idx, quote in enumerate(quotes, 1):
            quote_text = quote.get('quote', '').strip()
            # 规范化：去除空格、标点
            normalized = self._normalize_text(quote_text)
            normalized_map[normalized].append((idx, quote))

        for normalized, occurrences in normalized_map.items():
            if len(occurrences) > 1:
                # 只添加尚未在 exact 中的
                texts = [occ[1].get('quote', '') for occ in occurrences]
                if len(set(texts)) > 1:  # 不是完全相同的
                    self.duplicates.append({
                        'type': 'similar',
                        'normalized': normalized,
                        'count': len(occurrences),
                        'indices': [idx for idx, _ in occurrences],
                        'entries': occurrences
                    })
                    self.stats['similar_duplicates'] += 1

        return len(self.duplicates) == 0

    def _normalize_text(self, text: str) -> str:
        """规范化文本用于相似度检测"""
        # 去除空格和常见标点
        import re
        normalized = re.sub(r'[\s，。、；：？！""''（）\[\]【】《》]', '', text)
        return normalized.lower()

    def print_report(self):
        """打印检查报告"""
        print("\n" + "=" * 60)
        print("重复检查报告")
        print("=" * 60)

        if self.stats:
            print("\n统计:")
            for key, value in sorted(self.stats.items()):
                print(f"  {key}: {value}")

        if not self.duplicates:
            print("\n✓ 未发现重复！")
            return True
        else:
            print(f"\n发现 {len(self.duplicates)} 组重复:")
            print()

            for dup in self.duplicates[:30]:  # 只显示前30组
                if dup['type'] == 'exact':
                    print(f"【完全重复】出现 {dup['count']} 次:")
                    print(f"  文本: {dup['text'][:80]}...")
                    print(f"  位置: {dup['indices'][:10]}")
                else:
                    print(f"【相似内容】出现 {dup['count']} 次:")
                    print(f"  规范化: {dup['normalized'][:80]}...")
                    print(f"  位置: {dup['indices'][:10]}")
                    # 显示不同的文本
                    print(f"  文本变体:")
                    for idx, quote in dup['entries'][:5]:
                        print(f"    #{idx}: {quote.get('quote', '')[:60]}...")
                print()

            if len(self.duplicates) > 30:
                print(f"... 还有 {len(self.duplicates) - 30} 组重复未显示")

            print("\n❌ 发现重复！")
            return False


def check_yaml_file(yaml_path: Path) -> bool:
    """检查 YAML 文件"""
    print(f"\n正在检查: {yaml_path}")

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            quotes = yaml.safe_load(f) or []
    except Exception as e:
        print(f"✗ 无法读取文件: {e}")
        return False

    print(f"加载了 {len(quotes)} 条名句")

    checker = DuplicateChecker()
    return checker.check_duplicates(quotes) and checker.print_report()


def check_json_file(json_path: Path) -> bool:
    """检查 JSON 文件"""
    print(f"\n正在检查: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            quotes = json.load(f)
    except Exception as e:
        print(f"✗ 无法读取文件: {e}")
        return False

    print(f"加载了 {len(quotes)} 条名句")

    checker = DuplicateChecker()
    return checker.check_duplicates(quotes) and checker.print_report()


def remove_duplicates(quotes: List[Dict]) -> Tuple[List[Dict], int]:
    """
    去除重复名句
    返回 (去重后的列表, 移除的数量)
    """
    seen: Set[str] = set()
    unique_quotes = []
    removed = 0

    for quote in quotes:
        quote_text = quote.get('quote', '').strip()
        if quote_text and quote_text not in seen:
            seen.add(quote_text)
            unique_quotes.append(quote)
        else:
            removed += 1

    return unique_quotes, removed


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='检查名句重复')
    parser.add_argument(
        'file',
        nargs='?',
        help='要检查的文件路径（YAML 或 JSON）'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='检查所有数据文件'
    )
    parser.add_argument(
        '--remove',
        action='store_true',
        help='去除重复并保存（危险操作！）'
    )
    args = parser.parse_args()

    if args.all:
        # 检查所有数据文件
        yaml_path = Path(__file__).parent.parent / 'data' / 'quotes.yaml'
        json_path = Path(__file__).parent.parent / 'static' / 'quotes' / 'quotes.json'

        print("=" * 60)
        print("批量检查模式")
        print("=" * 60)

        success = True
        if yaml_path.exists():
            if not check_yaml_file(yaml_path):
                success = False
        if json_path.exists():
            if not check_json_file(json_path):
                success = False

        if success:
            print("\n" + "=" * 60)
            print("✓ 所有文件检查通过！")
            print("=" * 60)
        exit(0 if success else 1)

    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"✗ 文件不存在: {file_path}")
            exit(1)

        # 如果指定了 --remove，去除重复并保存
        if args.remove:
            print(f"\n正在去除重复: {file_path}")
            print("⚠ 警告：这将修改原文件！")

            # 读取文件
            if file_path.suffix in ['.yaml', '.yml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    quotes = yaml.safe_load(f) or []
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    quotes = json.load(f)
            else:
                print(f"✗ 不支持的文件类型: {file_path.suffix}")
                exit(1)

            # 去重
            original_count = len(quotes)
            unique_quotes, removed = remove_duplicates(quotes)

            print(f"\n原始数量: {original_count}")
            print(f"去重后: {len(unique_quotes)}")
            print(f"移除: {removed}")

            # 备份原文件
            backup_path = file_path.with_suffix(f'{file_path.suffix}.bak')
            import shutil
            shutil.copy2(file_path, backup_path)
            print(f"\n已备份至: {backup_path}")

            # 保存去重后的文件
            if file_path.suffix in ['.yaml', '.yml']:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(unique_quotes, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(unique_quotes, f, ensure_ascii=False, indent=2)

            print(f"✓ 已保存去重后的文件")
            exit(0)

        # 普通检查模式
        if file_path.suffix in ['.yaml', '.yml']:
            success = check_yaml_file(file_path)
        elif file_path.suffix == '.json':
            success = check_json_file(file_path)
        else:
            print(f"✗ 不支持的文件类型: {file_path.suffix}")
            exit(1)

        exit(0 if success else 1)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
