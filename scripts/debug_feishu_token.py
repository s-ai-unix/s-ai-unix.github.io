#!/usr/bin/env python3
"""
调试飞书 Token 和 API 响应
"""

import argparse
import requests
import json

def debug_token(token: str):
    # 1. 尝试获取用户信息，验证 Token 是否有效
    user_url = "https://open.feishu.cn/open-apis/authen/v1/user_info"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    print("🔍 正在验证 Token 有效性...")
    try:
        resp = requests.get(user_url, headers=headers)
        print(f"用户 API 状态码: {resp.status_code}")
        print(f"用户 API 响应: {resp.text[:500]}")
    except Exception as e:
        print(f"❌ 用户 API 请求失败: {e}")

    # 2. 尝试列出文件（不带任何过滤参数）
    files_url = "https://open.feishu.cn/open-apis/drive/v1/files"
    print("\n🔍 正在尝试列出云空间文件 (无过滤)...")
    try:
        resp = requests.get(files_url, headers=headers)
        print(f"文件 API 状态码: {resp.status_code}")
        print(f"文件 API 响应: {resp.text[:1000]}") # 打印更多内容以便分析
    except Exception as e:
        print(f"❌ 文件 API 请求失败: {e}")

    # 3. 尝试列出我的空间（Explorer API）
    # 有时候 drive/v1/files 为空是因为没有权限访问根目录，或者需要使用 explorer API
    explorer_url = "https://open.feishu.cn/open-apis/drive/explorer/v2/root_folder/meta"
    print("\n🔍 正在尝试获取根目录元数据...")
    try:
        resp = requests.get(explorer_url, headers=headers)
        print(f"根目录 API 状态码: {resp.status_code}")
        print(f"根目录 API 响应: {resp.text[:500]}")
    except Exception as e:
        print(f"❌ 根目录 API 请求失败: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    debug_token(args.token)

if __name__ == "__main__":
    main()
