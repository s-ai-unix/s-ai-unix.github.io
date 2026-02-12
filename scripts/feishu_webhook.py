#!/usr/bin/env python3
"""
飞书 Webhook 消息发送脚本
使用方法:
    python3 scripts/feishu_webhook.py --url <WEBHOOK_URL> --text "Hello World"
"""

import argparse
import json
import requests
import sys

def send_message(webhook_url: str, text: str):
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "msg_type": "text",
        "content": {
            "text": text
        }
    }
    
    try:
        response = requests.post(webhook_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        
        if result.get("code") == 0:
            print(f"✅ 消息发送成功: {text}")
        else:
            print(f"❌ 发送失败: {result}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 发送出错: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="发送飞书 Webhook 消息")
    parser.add_argument("--url", help="飞书 Webhook URL", required=True)
    parser.add_argument("--text", help="要发送的文本内容", default="Hello World")
    
    args = parser.parse_args()
    
    send_message(args.url, args.text)

if __name__ == "__main__":
    main()
