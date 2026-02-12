#!/usr/bin/env python3
"""
批量运行所有配图生成脚本
用法: python3 scripts/generate_all_plots.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    plots_dir = Path(__file__).parent / "plots"
    
    # 获取所有 generate_*_plots.py 文件
    plot_scripts = sorted(plots_dir.glob("generate_*_plots.py"))
    
    print(f"发现 {len(plot_scripts)} 个配图脚本")
    print("=" * 50)
    
    success = 0
    failed = 0
    
    for script in plot_scripts:
        script_name = script.name
        print(f"\n🔄 运行: {script_name}")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=Path(__file__).parent.parent,  # 回到项目根目录
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                print(f"✅ 成功: {script_name}")
                success += 1
            else:
                print(f"❌ 失败: {script_name}")
                print(result.stderr[:200])
                failed += 1
                
        except subprocess.TimeoutExpired:
            print(f"⏱️ 超时: {script_name}")
            failed += 1
        except Exception as e:
            print(f"💥 错误: {script_name} - {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"完成: {success} 成功, {failed} 失败")
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
