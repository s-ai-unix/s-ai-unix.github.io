#!/usr/bin/env python3
"""
查询飞书知识库 (Wiki Spaces) 列表
使用方法:
    python3 scripts/feishu_list_wikis.py --token <USER_ACCESS_TOKEN>
"""

import argparse
import requests
import json
import sys

def list_wikis(token: str):
    url = "https://open.feishu.cn/open-apis/wiki/v2/spaces"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    params = {
        "page_size": 50
    }
    
    print("🔍 正在查询知识库列表...")
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"❌ 请求失败 (HTTP {response.status_code}): {response.text}")
            return

        result = response.json()
        if result.get("code") != 0:
            print(f"❌ API 错误: {result}")
            return
            
        items = result.get("data", {}).get("items", [])
        
        print(f"\n📚 发现 {len(items)} 个知识库:")
        print("-" * 50)
        
        for item in items:
            name = item.get("name", "无标题")
            desc = item.get("description", "无描述")
            space_id = item.get("space_id", "")
            space_type = item.get("space_type", "unknown") # team/person
            
            print(f"[{space_type}] {name}")
            print(f"   📝 {desc}")
            print(f"   🆔 {space_id}")
            print("-" * 30)
            
        if len(items) == 0:
            print("没有找到任何知识库。")
            
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="查询飞书知识库列表")
    parser.add_argument("--token", help="User Access Token", required=True)
    
    args = parser.parse_args()
    
    list_wikis(args.token)

if __name__ == "__main__":
    main()
