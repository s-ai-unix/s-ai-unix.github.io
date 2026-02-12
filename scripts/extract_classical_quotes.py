#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从中国古籍 JSON 文件中提取名言警句
支持：老子、庄子、论语、孟子、史记、资治通鉴、左传等
"""

import json
import os
import re
from typing import List, Dict
from pathlib import Path


# 古籍对应的朝代信息
BOOK_DYNASTY = {
    "老子": "春秋",
    "庄子": "战国",
    "论语": "春秋",
    "孟子": "战国",
    "史记": "西汉",
    "资治通鉴": "北宋",
    "续资治通鉴": "北宋",
    "左传": "春秋",
    "易经": "商周",
    "周易": "商周",
    "易传": "商周",
    "孙子兵法": "春秋",
    "孙膑兵法": "战国",
    "大学章句集注": "南宋",
    "礼记": "西汉",
    "孝经": "西汉",
    "淮南子": "西汉",
    "素书": "秦汉",
    "山海经": "先秦",
    "墨子": "战国",
    "荀子": "战国",
    "韩非子": "战国",
    "管子": "春秋",
    "晏子春秋": "战国",
    "吕氏春秋": "战国",
    "春秋繁露": "西汉",
    "说苑": "西汉",
    "新序": "西汉",
    "列子": "战国",
    "文子": "战国",
    "鹖冠子": "战国",
    "商君书": "战国",
    "慎子": "战国",
    "尹文子": "战国",
    "公孙龙子": "战国",
    "鬼谷子": "战国",
    "尸子": "战国",
    "尉缭子": "战国",
    "吴子": "战国",
    "六韬": "战国",
    "司马法": "春秋",
    "黄石公三略": "秦汉",
    "潜夫论": "东汉",
    "论衡": "东汉",
    "申鉴": "东汉",
    "中鉴": "东汉",
    "太平经": "东汉",
    "周易参同契": "东汉",
    "风俗通义": "东汉",
    "人物志": "三国",
    "诸葛亮集": "三国",
    "曹操集": "三国",
    "曹丕集": "三国",
    "曹植集": "三国",
    "嵇康集": "三国",
    "阮籍集": "三国",
    "抱朴子": "东晋",
    "神仙传": "东晋",
    "搜神记": "东晋",
    "世说新语": "南北朝",
    "颜氏家训": "南北朝",
    "文心雕龙": "南北朝",
    "诗品": "南北朝",
    "水经注": "北魏",
    "洛阳伽蓝记": "北魏",
    "三国志": "西晋",
    "三国志注": "西晋",
    "华阳国志": "东晋",
    "后汉书": "南朝宋",
    "宋书": "南朝梁",
    "南齐书": "南朝梁",
    "梁书": "南朝梁",
    "陈书": "唐",
    "魏书": "北齐",
    "北齐书": "北齐",
    "周书": "唐",
    "隋书": "唐",
    "晋书": "唐",
    "南史": "唐",
    "北史": "唐",
    "旧唐书": "五代",
    "新唐书": "北宋",
    "旧五代史": "北宋",
    "新五代史": "北宋",
    "宋史": "元",
    "辽史": "元",
    "金史": "元",
    "元史": "明",
    "明史": "清",
    "贞观政要": "唐",
    "大唐西域记": "唐",
    "坛经": "唐",
    "黄庭坚集": "北宋",
    "苏轼集": "北宋",
    "王安石集": "北宋",
    "欧阳修集": "北宋",
    "司马光集": "北宋",
    "曾巩集": "北宋",
    "柳宗元集": "唐",
    "韩愈集": "唐",
    "李白集": "唐",
    "杜甫集": "唐",
    "白居易集": "唐",
    "王维集": "唐",
    "孟浩然集": "唐",
    "刘禹锡集": "唐",
    "李商隐集": "唐",
    "杜牧集": "唐",
    "李清照集": "南宋",
    "陆游集": "南宋",
    "辛弃疾集": "南宋",
    "朱熹集": "南宋",
    "程颢集": "北宋",
    "程颐集": "北宋",
    "周敦颐集": "北宋",
    "张载集": "北宋",
    "邵雍集": "北宋",
    "二程集": "北宋",
    "传习录": "明",
    "王阳明集": "明",
    "李贽集": "明",
    "归有光集": "明",
    "袁宏道集": "明",
    "张岱集": "明",
    "顾炎武集": "明末清初",
    "王夫之集": "明末清初",
    "黄宗羲集": "明末清初",
    "方苞集": "清",
    "姚鼐集": "清",
    "刘大櫆集": "清",
    "曾国藩集": "清",
    "左宗棠集": "清",
    "胡林翼集": "清",
    "李鸿章集": "清",
    "张之洞集": "清",
    "梁启超集": "清",
    "谭嗣同集": "清",
    "严复集": "清",
    "章太炎集": "清",
    "王国维集": "清",
}


def is_valid_quote(text: str) -> bool:
    """判断是否是有效的名言警句"""
    if not text or len(text.strip()) < 8:
        return False
    if len(text.strip()) > 100:
        return False

    # 过滤掉一些不合适的句子
    invalid_patterns = [
        r'^.{1,3}$',  # 太短
        r'^.{80,}$',  # 太长
        r'^[0-9]+$',
        r'^第.*章',  # 章节标题
        r'^[卷册篇第].*',
    ]

    for pattern in invalid_patterns:
        if re.match(pattern, text.strip()):
            return False

    return True


def extract_quotes_from_content(content: str, book_name: str, chapter_title: str) -> List[Dict]:
    """从内容中提取名言警句"""
    quotes = []

    # 按句号、问号、感叹号分割完整句子（不包括分号）
    sentences = re.split(r'([。！？])', content)

    # 重新组合句子和标点
    full_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
            full_sentences.append(sentence.strip())

    # 提取有意义的句子作为名言
    for sentence in full_sentences:
        if not is_valid_quote(sentence):
            continue

        # 进一步筛选条件
        quote_len = len(sentence)

        # 长度限制
        if quote_len < 8:
            continue
        if quote_len > 80:
            continue

        # 必须包含至少一个有意义的关键词或模式
        # 有哲理的句子通常包含这些特征
        has_meaningful_content = False

        # 包含有哲理的词汇
        philosophical_keywords = [
            '道', '德', '仁', '义', '礼', '智', '信', '善', '恶', '美', '丑',
            '天', '地', '人', '心', '性', '命', '理', '气', '神', '形',
            '君子', '圣人', '小人', '王者', '智者', '勇者',
            '治国', '修身', '齐家', '平天下',
            '故', '是以', '是以', '是以', '然', '则', '虽', '若',
            '不', '无', '有', '为', '无为', '有为',
            '知', '行', '言', '听', '视', '思',
            '学', '教', '化', '成', '败', '兴', '亡',
            '柔', '刚', '强', '弱', '静', '动',
            '常', '变', '本', '末', '始', '终',
        ]

        if any(keyword in sentence for keyword in philosophical_keywords):
            has_meaningful_content = True

        # 或者包含排比、对偶等修辞手法
        if '，' in sentence or '；' in sentence:
            parts = sentence.split('，')
            if len(parts) >= 2:
                # 检查是否有对偶
                if len(parts[0]) > 4 and len(parts[1]) > 4:
                    has_meaningful_content = True

        # 或者是经典的短句
        if quote_len < 15:
            classic_patterns = [
                r'^.{2,4}者.{2,4}$',  # "...者..." 格式
                r'^不.{2,8}$',  # "不..." 格式
                r'^.{2,4}而不.{2,4}$',  # "...而不..." 格式
                r'有.{2,6}有.{2,6}',  # "有...有..." 排比
                r'无.{2,6}无.{2,6}',  # "无...无..." 排比
            ]
            for pattern in classic_patterns:
                if re.search(pattern, sentence):
                    has_meaningful_content = True
                    break

        if not has_meaningful_content:
            continue

        # 清理句子末尾可能的空格和标点
        sentence = sentence.strip()

        quotes.append({
            "quote": sentence,
            "author": BOOK_DYNASTY.get(book_name, "先秦"),
            "source": f"《{book_name}·{chapter_title}》" if chapter_title else f"《{book_name}》",
            "dynasty": BOOK_DYNASTY.get(book_name, "先秦"),
            "category": "诸子百家" if book_name in ["老子", "庄子", "论语", "孟子"] else "史学典籍"
        })

    return quotes


def process_json_file(file_path: Path) -> List[Dict]:
    """处理单个 JSON 文件"""
    print(f"处理文件: {file_path.name}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  错误: {e}")
        return []

    book_name = data.get('name', file_path.stem)
    quotes = []

    articles = data.get('articles', [])
    for article in articles[:50]:  # 每本书最多取前50章，避免太多
        chapter_title = article.get('title', '').replace(f'{book_name}·', '').strip()
        contents = article.get('content', [])

        if isinstance(contents, list):
            content_text = ''.join(contents)
        else:
            content_text = str(contents)

        # 从章节中提取名言
        chapter_quotes = extract_quotes_from_content(content_text, book_name, chapter_title)
        quotes.extend(chapter_quotes)

    print(f"  提取了 {len(quotes)} 条名言")
    return quotes


def remove_duplicates(quotes: List[Dict]) -> List[Dict]:
    """去重"""
    seen = set()
    unique_quotes = []

    for quote in quotes:
        quote_text = quote['quote'].strip()
        if quote_text not in seen:
            seen.add(quote_text)
            unique_quotes.append(quote)

    removed = len(quotes) - len(unique_quotes)
    if removed > 0:
        print(f"去重: 移除了 {removed} 条重复名言")

    return unique_quotes


def main():
    # 源数据目录
    source_dir = Path('/tmp/chinese-ancient-text')

    # 输出目录
    output_dir = Path('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/quotes')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 要处理的典籍
    target_books = [
        "老子.json",
        "庄子.json",
        "论语.json",
        "孟子.json",
        "史记.json",
        "资治通鉴.json",
        "左传.json",
        "易经.json",
        "周易.json",
        "易传.json",
        "孙子兵法.json",
        "孙膑兵法.json",
        "礼记.json",
        "孝经.json",
        "淮南子.json",
        "素书.json",
        "山海经.json",
        "墨子.json",
        "荀子.json",
        "韩非子.json",
        "管子.json",
        "列子.json",
        "文子.json",
        "商君书.json",
        "慎子.json",
        "吕氏春秋.json",
        "春秋繁露.json",
        "说苑.json",
        "新序.json",
        "战国策.json",
        "晏子春秋.json",
    ]

    all_quotes = []

    # 处理每个文件
    for book_file in target_books:
        file_path = source_dir / book_file
        if not file_path.exists():
            print(f"文件不存在: {book_file}")
            continue

        quotes = process_json_file(file_path)
        all_quotes.extend(quotes)

    print(f"\n总计提取了 {len(all_quotes)} 条名言")

    # 去重
    all_quotes = remove_duplicates(all_quotes)

    # 保存为 JSON 文件
    output_file = output_dir / 'classical_quotes_new.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_quotes, f, ensure_ascii=False, indent=2)

    print(f"\n已保存到: {output_file}")
    print(f"共 {len(all_quotes)} 条名言")


if __name__ == '__main__':
    main()
