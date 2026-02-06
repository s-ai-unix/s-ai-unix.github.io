#!/usr/bin/env python3
"""
查询飞书最近更新的文档
使用方法:
    python3 scripts/feishu_list_docs.py --token <USER_ACCESS_TOKEN> --days 7

注意: 需要提供 User Access Token (用户访问凭证)
"""

import argparse
import json
import requests
import sys
import time
from datetime import datetime, timedelta

def list_recent_docs(token: str, days: int):
    url = "https://open.feishu.cn/open-apis/drive/v1/files"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    # 计算时间范围
    now = datetime.now()
    start_time = now - timedelta(days=days)
    print(f"🔍 正在查询 {start_time.strftime('%Y-%m-%d')} 之后的文档...")
    
    # 飞书 API 可能不支持直接按时间过滤列表，我们获取列表后在本地过滤
    # 或者使用 order_by=edited_time DESC
    
    params = {
        "page_size": 50,
        "order_by": "EditedTime",
        "direction": "DESC"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"❌ 请求失败 (HTTP {response.status_code}): {response.text}")
            return

        result = response.json()
        if result.get("code") != 0:
            print(f"❌ API 错误: {result}")
            return
            
        files = result.get("data", {}).get("files", [])
        
        found_count = 0
        print(f"\n📄 最近 {days} 天更新的文档:")
        print("-" * 50)
        
        for file in files:
            # 飞书返回的时间戳通常是秒或毫秒，需要确认
            # 假设是秒 (如果数值很大则是毫秒)
            edited_time_ts = int(file.get("modified_time", 0))
            
            # 简单的判断：如果是毫秒级（13位），转为秒
            if edited_time_ts > 10000000000: 
                edited_time_ts = edited_time_ts / 1000
                
            file_time = datetime.fromtimestamp(edited_time_ts)
            
            if file_time < start_time:
                # 因为是按时间倒序，一旦遇到旧文件就可以停止了(如果API严格排序)
                # 但为了保险，我们继续检查（或者可以break）
                continue
                
            found_count += 1
            file_type = file.get("type", "unknown")
            file_name = file.get("name", "无标题")
            file_url = file.get("url", "")
            
            print(f"[{file_time.strftime('%Y-%m-%d %H:%M')}] [{file_type}] {file_name}")
            print(f"   🔗 {file_url}")
            
        if found_count == 0:
            print("没有找到最近更新的文档。")
            
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="查询飞书最近更新的文档")
    parser.add_argument("--token", help="User Access Token", required=True)
    parser.add_argument("--days", help="查询最近几天的文档", type=int, default=7)
    
    args = parser.parse_args()
    
    list_recent_docs(args.token, args.days)

if __name__ == "__main__":
    main()
