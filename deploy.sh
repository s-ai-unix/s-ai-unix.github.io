#!/bin/bash

# 部署脚本 - 构建 Hugo 站点并推送到 gh-pages 分支

echo "开始构建 Hugo 站点..."

# 构建站点
hugo --minify

if [ $? -ne 0 ]; then
    echo "❌ Hugo 构建失败"
    exit 1
fi

echo "✅ Hugo 构建成功"

# 进入 public 目录
cd public

# 初始化 git 仓库（如果还没初始化）
if [ ! -d ".git" ]; then
    git init
    git branch -m main
fi

# 添加远程仓库
git remote add origin https://github.com/s-ai-unix/blog.git 2>/dev/null || true

# 切换到 gh-pages 分支或创建它
git checkout -b gh-pages 2>/dev/null || git checkout gh-pages

# 添加所有文件
git add .

# 提交
git commit -m "Deploy to GitHub Pages"

# 推送到远程 gh-pages 分支
echo "推送到 GitHub (gh-pages 分支)..."
git push origin gh-pages --force

echo "✅ 部署完成！"
echo "博客地址: https://s-ai-unix.github.io/blog/"
