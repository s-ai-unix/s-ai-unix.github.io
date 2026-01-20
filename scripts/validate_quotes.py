#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名句数据���证工具
检查数据完整性、编码、格式等问题
"""

import json
import yaml
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict


# 朝代枚举（用于验证）
VALID_DYNASTIES = {
    # 中国古代
    '先秦', '春秋', '战国', '秦', '汉', '西汉', '东汉', '三国', '西晋', '东晋',
    '南北朝', '隋', '唐', '五代', '宋', '北宋', '南宋', '元', '明', '清',
    # 现代
    '现代', '当代', '民国',
    # 外国
    'Ancient Greece', 'Roman Empire', 'English Renaissance',
    '17th Century', '18th Century', '19th Century', '20th Century',
    'Modern', 'Contemporary',
}


class QuoteValidator:
    """名句验证器"""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)

    def validate_all(self, quotes: List[Dict]) -> bool:
        """
        验证所有名句
        返回 True 表示全部通过
        """
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)

        for idx, quote in enumerate(quotes, 1):
            self._validate_quote(idx, quote)

        return len(self.errors) == 0

    def _validate_quote(self, idx: int, quote: Dict):
        """验证单条名句"""
        # 检查必填字段
        required_fields = ['quote', 'author', 'source', 'dynasty']
        for field in required_fields:
            if field not in quote or not quote[field]:
                self.errors.append(f"#{idx}: 缺少必填字段 '{field}'")
                self.stats['missing_fields'] += 1

        # 检查字段类型
        if 'quote' in quote:
            self._validate_quote_text(idx, quote['quote'])
        if 'author' in quote:
            self._validate_author(idx, quote['author'])
        if 'dynasty' in quote:
            self._validate_dynasty(idx, quote['dynasty'])

    def _validate_quote_text(self, idx: int, text: str):
        """验证名句文本"""
        # 长度检查
        if len(text) < 5:
            self.warnings.append(f"#{idx}: 名句过短（< 5 字）: '{text[:20]}...'")
            self.stats['too_short'] += 1
        elif len(text) > 200:
            self.warnings.append(f"#{idx}: 名句过长（> 200 字）")
            self.stats['too_long'] += 1

        # 检查控制字符（乱码检测）
        control_chars = re.findall(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', text)
        if control_chars:
            self.errors.append(f"#{idx}: 包含控制字符（可能是乱码）")
            self.stats['control_chars'] += 1

        # 检查替换字符（编码问题标志）
        if '' in text:
            self.warnings.append(f"#{idx}: 包含替换字符 ''（可能有编码问题）")
            self.stats['replacement_chars'] += 1

        # 检查空白字符
        if text.strip() != text:
            self.warnings.append(f"#{idx}: 首尾有空格")
            self.stats['whitespace'] += 1

    def _validate_author(self, idx: int, author: str):
        """验证作者字段"""
        if len(author.strip()) == 0:
            self.errors.append(f"#{idx}: 作者为空")
            self.stats['empty_author'] += 1

    def _validate_dynasty(self, idx: int, dynasty: str):
        """验证朝代字段"""
        if dynasty not in VALID_DYNASTIES:
            self.warnings.append(
                f"#{idx}: 朝代 '{dynasty}' 不在标准列表中（可能但不一定是错误）"
            )
            self.stats['unknown_dynasty'] += 1

    def print_report(self):
        """打印验证报告"""
        print("\n" + "=" * 60)
        print("验证报告")
        print("=" * 60)

        if self.stats:
            print("\n统计信息:")
            for key, value in sorted(self.stats.items()):
                print(f"  {key}: {value}")

        if self.warnings:
            print(f"\n警告 ({len(self.warnings)} 条):")
            for warning in self.warnings[:20]:  # 只显示前20条
                print(f"  ⚠ {warning}")
            if len(self.warnings) > 20:
                print(f"  ... 还有 {len(self.warnings) - 20} 条警告")

        if self.errors:
            print(f"\n错误 ({len(self.errors)} 条):")
            for error in self.errors[:20]:  # 只显示前20条
                print(f"  ✗ {error}")
            if len(self.errors) > 20:
                print(f"  ... 还有 {len(self.errors) - 20} 条错误")
            print("\n❌ 验证失败！")
            return False
        else:
            print("\n✓ 验证通过！")
            return True


def validate_yaml_file(yaml_path: Path) -> bool:
    """验证 YAML 文件"""
    print(f"\n正在验证 YAML 文件: {yaml_path}")

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            quotes = yaml.safe_load(f) or []
    except Exception as e:
        print(f"✗ 无法读取文件: {e}")
        return False

    validator = QuoteValidator()
    validator.validate_all(quotes)
    return validator.print_report()


def validate_json_file(json_path: Path) -> bool:
    """验证 JSON 文件"""
    print(f"\n正在验证 JSON 文件: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            quotes = json.load(f)
    except Exception as e:
        print(f"✗ 无法读取文件: {e}")
        return False

    validator = QuoteValidator()
    validator.validate_all(quotes)
    return validator.print_report()


def check_encoding(file_path: Path) -> Tuple[bool, str]:
    """
    检查文件编码
    返回 (是否UTF8, 编码名称)
    """
    try:
        # 尝试用 UTF-8 读取
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        return True, 'UTF-8'
    except UnicodeDecodeError:
        pass

    # 尝试其他编码
    encodings = ['gbk', 'gb2312', 'gb18030', 'big5', 'shift_jis']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                f.read()
            return False, enc
        except UnicodeDecodeError:
            continue

    return False, 'Unknown'


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='验证名句数据')
    parser.add_argument(
        'file',
        nargs='?',
        help='要验证的文件路径（YAML 或 JSON）'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='验证所有数据文件'
    )
    args = parser.parse_args()

    if args.all:
        # 验证所有数据文件
        yaml_path = Path(__file__).parent.parent / 'data' / 'quotes.yaml'
        json_path = Path(__file__).parent.parent / 'static' / 'quotes' / 'quotes.json'

        print("=" * 60)
        print("批量验证模式")
        print("=" * 60)

        # 检查编码
        print("\n编码检查:")
        if yaml_path.exists():
            is_utf8, encoding = check_encoding(yaml_path)
            status = "✓" if is_utf8 else "✗"
            print(f"  {status} {yaml_path}: {encoding}")
        if json_path.exists():
            is_utf8, encoding = check_encoding(json_path)
            status = "✓" if is_utf8 else "✗"
            print(f"  {status} {json_path}: {encoding}")

        # 验证内容
        success = True
        if yaml_path.exists():
            if not validate_yaml_file(yaml_path):
                success = False
        if json_path.exists():
            if not validate_json_file(json_path):
                success = False

        if success:
            print("\n" + "=" * 60)
            print("✓ 所有文件验证通过！")
            print("=" * 60)
        exit(0 if success else 1)

    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"✗ 文件不存在: {file_path}")
            exit(1)

        # 检查编码
        is_utf8, encoding = check_encoding(file_path)
        status = "✓" if is_utf8 else "✗"
        print(f"{status} 编码: {encoding}")

        # 根据扩展名选择验证方法
        if file_path.suffix in ['.yaml', '.yml']:
            success = validate_yaml_file(file_path)
        elif file_path.suffix == '.json':
            success = validate_json_file(file_path)
        else:
            print(f"✗ 不支持的文件类型: {file_path.suffix}")
            exit(1)

        exit(0 if success else 1)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
