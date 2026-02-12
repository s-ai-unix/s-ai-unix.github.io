#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名句数据生成脚本 - 从 GitHub 下载中文名句数据
来源: xuchunyang/mingju 和 chinese-poetry/chinese-poetry
"""

import json
import os
import sys
import subprocess
import hashlib
from typing import List, Dict, Set
from pathlib import Path

# 配置
TARGET_QUOTE_COUNT = 5000  # 目标名句数量
SHARD_SIZE = 500           # 每个分片的名句数量
QUOTES_DIR = Path(__file__).parent.parent / "static" / "quotes"
TEMP_DIR = Path("/tmp/quotes_temp_github")


def run_command(cmd: List[str], cwd=None) -> bool:
    """执行 shell 命令"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        return True
    except Exception as e:
        print(f"命令执行失败: {' '.join(cmd)} - {e}")
        return False


def clone_mingju_repo() -> Path:
    """克隆 mingju 名句仓库"""
    print("正在克隆 xuchunyang/mingju 仓库...")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    repo_path = TEMP_DIR / "mingju"
    if repo_path.exists():
        print(f"仓库已存在，跳过克隆: {repo_path}")
        return repo_path

    if not run_command([
        "git", "clone",
        "--depth", "1",
        "https://github.com/xuchunyang/mingju.git",
        str(repo_path)
    ]):
        raise RuntimeError("克隆 mingju 仓库失败")

    print(f"仓库克隆成功: {repo_path}")
    return repo_path


def clone_chinese_poetry_repo() -> Path:
    """克隆 chinese-poetry 诗歌仓库（用于补充数据）"""
    print("正在克隆 chinese-poetry/chinese-poetry 仓库...")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    repo_path = TEMP_DIR / "chinese-poetry"
    if repo_path.exists():
        print(f"仓库已存在，跳过克隆: {repo_path}")
        return repo_path

    if not run_command([
        "git", "clone",
        "--depth", "1",
        "https://github.com/chinese-poetry/chinese-poetry.git",
        str(repo_path)
    ]):
        raise RuntimeError("克隆 chinese-poetry 仓库失败")

    print(f"仓库克隆成功: {repo_path}")
    return repo_path


def load_mingju_quotes(repo_path: Path) -> List[Dict]:
    """从 mingju 仓库加载名句数据"""
    print("正在加载 mingju 名句数据...")
    quotes = []

    # mingju 仓库可能有多种数据文件格式
    # 查找所有 json 文件
    json_files = list(repo_path.glob("*.json"))
    if not json_files:
        # 查找子目录
        json_files = list(repo_path.glob("**/*.json"))

    print(f"找到 {len(json_files)} 个 JSON 文件")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 处理不同的数据格式
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                # 可能是按分类组织的数据
                if "data" in data:
                    items = data["data"]
                elif "quotes" in data:
                    items = data["quotes"]
                else:
                    items = list(data.values())

            for item in items:
                if not isinstance(item, dict):
                    continue

                # 尝试多种字段名
                quote_text = (
                    item.get("contents") or item.get("诗句") or item.get("content") or
                    item.get("text") or item.get("quote") or
                    item.get("句子") or ""
                )

                if not quote_text or len(quote_text) < 4:
                    continue

                # 处理 source 字段（可能包含作者和作品名）
                source = item.get("source") or item.get("出处") or item.get("title") or ""
                author = item.get("author") or item.get("作者") or ""

                # 从 source 中提取作者（如果 author 为空）
                # 格式通常是: "作者《作品名》"
                if not author and source and "《" in source:
                    parts = source.split("《", 1)
                    if len(parts) == 2:
                        author = parts[0].strip()
                        source = f"《{parts[1]}"

                quote = {
                    "quote": quote_text.strip(),
                    "author": author.strip(),
                    "source": source.strip(),
                    "dynasty": (item.get("朝代") or item.get("dynasty") or "").strip(),
                    "tags": item.get("tags") or []
                }

                if 4 <= len(quote["quote"]) <= 100:
                    quotes.append(quote)

        except Exception as e:
            print(f"读取文件 {json_file} 失败: {e}")
            continue

    print(f"从 mingju 加载了 {len(quotes)} 条名句")
    return quotes


def load_chinese_poetry_quotes(repo_path: Path, limit: int = 3000) -> List[Dict]:
    """从 chinese-poetry 仓库加载诗词数据（补充用）"""
    print(f"正在加载 chinese-poetry 诗词数据（最多 {limit} 条）...")
    quotes = []
    count = 0

    try:
        # chinese-poetry 仓库使用中文目录名
        # 主要目录: 全唐诗、宋词、诗经、论语等
        tang_dirs = list((repo_path / "全唐诗").glob("*.json"))
        song_dirs = list((repo_path / "宋词").glob("**/*.json"))

        all_poetry_files = tang_dirs + song_dirs
        print(f"找到 {len(all_poetry_files)} 个诗词文件")

        for json_file in all_poetry_files[:50]:  # 限制处理文件数量
            if count >= limit:
                break

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 处理不同的数据结构
                poems = []
                if isinstance(data, list):
                    poems = data
                elif isinstance(data, dict):
                    # 可能按作者分组
                    for author_data in data.values():
                        if isinstance(author_data, list):
                            poems.extend(author_data)

                for poem in poems:
                    if count >= limit:
                        break

                    if not isinstance(poem, dict):
                        continue

                    content = poem.get("content", [])
                    author = poem.get("author", "") or ""
                    title = poem.get("title", "") or ""
                    dynasty = "唐" if "全唐诗" in str(json_file) else "宋"

                    if isinstance(content, list) and len(content) >= 2:
                        # 提取对联作为名句
                        for i in range(0, min(len(content) - 1, 6), 2):
                            if count >= limit:
                                break
                            quote_text = content[i] + "，" + content[i + 1]
                            quotes.append({
                                "quote": quote_text,
                                "author": author,
                                "source": f"《{title}》" if title else "",
                                "dynasty": dynasty,
                                "tags": []
                            })
                            count += 1
            except Exception:
                continue

        print(f"从 chinese-poetry 加载了 {len(quotes)} 条诗词")
    except Exception as e:
        print(f"加载 chinese-poetry 数据失败: {e}")

    return quotes


def normalize_text(text: str) -> str:
    """标准化文本"""
    if not text:
        return ""
    return text.strip()


def calculate_hash(quote: Dict) -> str:
    """计算名句的哈希值用于去重"""
    content = quote.get("quote", "")
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def deduplicate_quotes(quotes: List[Dict]) -> List[Dict]:
    """去重名句"""
    print(f"正在去重，原始数量: {len(quotes)}")
    seen: Set[str] = set()
    unique_quotes = []

    for quote in quotes:
        quote_hash = calculate_hash(quote)
        if quote_hash not in seen:
            seen.add(quote_hash)
            unique_quotes.append(quote)

    print(f"去重后数量: {len(unique_quotes)}")
    return unique_quotes


def clean_quotes(quotes: List[Dict]) -> List[Dict]:
    """清洗名句数据"""
    print("正在清洗名句数据...")
    cleaned = []

    for quote in quotes:
        quote_text = quote.get("quote", "")
        
        # 检查必要字段
        if not quote_text or len(quote_text) < 4:
            continue

        # 检查是否包含明显的乱码（连续特殊字符过多）
        special_chars = sum(1 for c in quote_text if ord(c) < 32)
        if special_chars > len(quote_text) * 0.2:
            continue

        # 标准化
        quote["quote"] = normalize_text(quote_text)
        quote["author"] = normalize_text(quote.get("author", ""))
        quote["source"] = normalize_text(quote.get("source", ""))
        quote["dynasty"] = normalize_text(quote.get("dynasty", ""))

        cleaned.append(quote)

    print(f"清洗后数量: {len(cleaned)}")
    return cleaned


def create_shards(quotes: List[Dict], shard_size: int) -> List[Dict]:
    """创建分片索引"""
    print(f"正在创建分片，每个分片 {shard_size} 条名句...")
    total_shards = (len(quotes) + shard_size - 1) // shard_size

    shards = []
    for i in range(total_shards):
        shard_name = f"quotes_{i + 1}.json"
        start = i * shard_size
        end = min(start + shard_size, len(quotes))
        shards.append({
            "name": shard_name,
            "start": start,
            "end": end,
            "count": end - start
        })

    print(f"共创建 {len(shards)} 个分片")
    return shards


def save_quotes(quotes: List[Dict], shards: List[Dict], output_dir: Path):
    """保存名句到分片文件"""
    print(f"正在保存名句到 {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存每个分片
    for shard in shards:
        shard_path = output_dir / shard["name"]
        shard_data = quotes[shard["start"]:shard["end"]]

        with open(shard_path, 'w', encoding='utf-8') as f:
            json.dump(shard_data, f, ensure_ascii=False, indent=2)
        print(f"已保存: {shard_path} ({shard['count']} 条)")

    # 保存索引文件
    index = {
        "total_quotes": len(quotes),
        "total_shards": len(shards),
        "shard_size": SHARD_SIZE,
        "shards": [s["name"] for s in shards],
        "last_updated": ""  # 可以添加时间戳
    }

    index_path = output_dir / "quotes_index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"已保存索引: {index_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("中文名句数据生成工具 - GitHub 版")
    print(f"目标数量: {TARGET_QUOTE_COUNT} 条")
    print(f"分片大小: {SHARD_SIZE} 条/文件")
    print("=" * 60)

    all_quotes = []

    try:
        # 1. 从 mingju 仓库加载
        mingju_path = clone_mingju_repo()
        mingju_quotes = load_mingju_quotes(mingju_path)
        all_quotes.extend(mingju_quotes)

        # 2. 如果数量不足，从 chinese-poetry 补充
        if len(all_quotes) < TARGET_QUOTE_COUNT:
            supplement_count = TARGET_QUOTE_COUNT - len(all_quotes)
            print(f"\n名句数量不足 ({len(all_quotes)})，还需补充 {supplement_count} 条")

            poetry_path = clone_chinese_poetry_repo()
            poetry_quotes = load_chinese_poetry_quotes(poetry_path, limit=supplement_count)
            all_quotes.extend(poetry_quotes)

        # 3. 清洗数据
        all_quotes = clean_quotes(all_quotes)

        # 4. 去重
        all_quotes = deduplicate_quotes(all_quotes)

        # 5. 如果超过目标数量，截取前 N 条
        if len(all_quotes) > TARGET_QUOTE_COUNT:
            print(f"\n名句数量超过目标，截取前 {TARGET_QUOTE_COUNT} 条")
            all_quotes = all_quotes[:TARGET_QUOTE_COUNT]

        print(f"\n最终名句数量: {len(all_quotes)}")

        # 6. 创建分片
        shards = create_shards(all_quotes, SHARD_SIZE)

        # 7. 保存文件
        save_quotes(all_quotes, shards, QUOTES_DIR)

        print("\n" + "=" * 60)
        print("✓ 名句数据生成完成！")
        print(f"  - 总名句数: {len(all_quotes)}")
        print(f"  - 分片数量: {len(shards)}")
        print(f"  - 输出目录: {QUOTES_DIR}")
        print("=" * 60)

        # 清理临时文件
        print(f"\n临时文件保留在: {TEMP_DIR}")
        print(f"如需清理，请运行: rm -rf {TEMP_DIR}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
