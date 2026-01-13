#!/bin/bash

# 部署脚本 - 构建 Hugo 站点并推送到 gh-pages 分支

echo "🚀 开始部署流程..."

# 保存当前分支
CURRENT_BRANCH=$(git branch --show-current)

echo "📦 1. 更新子模块并构建 Hugo 站点..."
git submodule update --init --recursive
hugo --minify

if [ $? -ne 0 ]; then
    echo "❌ Hugo 构建失败"
    exit 1
fi

echo "✅ Hugo 构建成功"

echo "📝 2. 保存源代码更改..."
git add .
git commit -m "Update content" || echo "没有新的源代码更改"

echo "🚀 3. 推送源代码到 main 分支..."
git push origin $CURRENT_BRANCH

echo "🌐 4. 部署到 gh-pages 分支..."

# 创建临时分支用于部署
git checkout --orphan gh-pages-temp

# 清空所有文件
git rm -rf . > /dev/null 2>&1

# 复制构建产物
cp -r public/* .
rm -rf public

# 添加所有文件
git add .
git commit -m "Deploy to GitHub Pages - $(date +'%Y-%m-%d %H:%M:%S')"

# 强制推送到 gh-pages
git push origin HEAD:gh-pages --force

# 清理临时分支并回到原分支
git checkout $CURRENT_BRANCH
git branch -D gh-pages-temp

echo "✅ 部署完成！"
echo "📍 博客地址: https://s-ai-unix.github.io/blog/"
echo "⏳ 通常需要 1-3 分钟生效"
