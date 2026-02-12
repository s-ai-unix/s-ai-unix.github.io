#!/bin/bash

# 部署脚本 - 构建 Hugo 站点并推送到 gh-pages 分支
# 增强版：添加多项检查，避免部署后才发现问题

set -e  # 遇到错误立即退出

echo "🚀 开始部署流程..."

# 保存当前分支
CURRENT_BRANCH=$(git branch --show-current)

# ==================== 预检查阶段 ====================

echo "🔍 0. 部署前检查..."

# 显示子模块状态（会在构建阶段自动初始化）
echo "   检查子模块状态..."
git submodule status
echo "   💡 子模块将在构建阶段自动更新"

# 检查是否有未来日期的文章
echo "   检查未来日期的文章..."
FUTURE_POSTS=$(find content/posts -name "*.md" -exec grep -l "^draft: false" {} \; | \
                while read file; do
                    DATE=$(grep "^date:" "$file" | head -1 | sed 's/date: //; s/[" ]//g')
                    if [ -n "$DATE" ]; then
                        FILE_DATE=$(date -j -f "%Y-%m-%dT%H:%M:%S%z" "$DATE" +"%s" 2>/dev/null || echo "0")
                        NOW=$(date +"%s")
                        if [ "$FILE_DATE" -gt "$NOW" ]; then
                            echo "      ⚠️  $file (日期: $DATE)"
                        fi
                    fi
                done)

if [ -n "$FUTURE_POSTS" ]; then
    echo "   ⚠️  警告：发现未来日期的文章（不会被发布）："
    echo "$FUTURE_POSTS"
    echo "   💡 提示：如需发布，请将日期改为当前或过去时间"
    read -p "   是否继续部署？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   ❌ 部署已取消"
        exit 1
    fi
fi

# 检查是否有 draft: true 但未标记为草稿的文章
DRAFT_COUNT=$(find content/posts -name "*.md" -exec grep -l "^draft: true" {} \; | wc -l | tr -d ' ')
if [ "$DRAFT_COUNT" -gt 0 ]; then
    echo "   ℹ️  发现 $DRAFT_COUNT 篇草稿文章（不会被发布）"
fi

# ==================== 构建阶段 ====================

echo ""
echo "📦 1. 更新子模块并构建 Hugo 站点..."
git submodule update --init --recursive

# 记录构建前的页面数量（如果有public目录）
if [ -d "public" ]; then
    OLD_PAGE_COUNT=$(find public -name "*.html" | wc -l | tr -d ' ')
else
    OLD_PAGE_COUNT=0
fi

hugo -F --minify

if [ $? -ne 0 ]; then
    echo "❌ Hugo 构建失败"
    exit 1
fi

# 检查构建结果
echo ""
echo "🔍 2. 验证构建结果..."

if [ ! -d "public" ]; then
    echo "   ❌ 错误：public 目录不存在"
    exit 1
fi

# 统计生成的页面（使用 Hugo list 命令）
PAGE_COUNT=$(hugo list all | wc -l | tr -d ' ')
echo "   📄 生成页面数: $PAGE_COUNT"

# 检查页面数量是否异常
if [ -n "$PAGE_COUNT" ] && [ "$PAGE_COUNT" -lt 50 ]; then
    echo "   ⚠️  警告：页面数量异常少（< 50），可能存在问题"
    echo "   💡 可能原因："
    echo "      - 文章日期在未来"
    echo "      - 大部分文章标记为 draft: true"
    echo "      - 主题未正确加载"
fi

# 检查关键页面
if [ ! -f "public/index.html" ]; then
    echo "   ❌ 错误：首页未生成"
    exit 1
fi

# 检查最新文章是否存在（跨平台兼容）
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS (BSD find)
    LATEST_POST=$(find content/posts -name "*.md" -type f -exec stat -f "%m %N" {} \; | sort -n | tail -1 | cut -d' ' -f2-)
else
    # Linux (GNU find)
    LATEST_POST=$(find content/posts -name "*.md" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
fi

if [ -n "$LATEST_POST" ]; then
    POST_NAME=$(basename "$LATEST_POST" .md)
    # 转换为Hugo的URL格式
    POST_URL=$(echo "$POST_NAME" | sed 's/--/\/:/g')
    if [ ! -f "public/posts/$POST_URL/index.html" ]; then
        echo "   ⚠️  警告：最新文章可能未正确生成: $POST_NAME"
    fi
fi

echo "   ✅ 构建验证通过"

# ==================== 提交阶段 ====================

echo ""
echo "📝 3. 保存源代码更改..."
git add .
git commit -m "Update content" || echo "没有新的源代码更改"

echo "🚀 4. 推送源代码到 main 分支..."
git push origin $CURRENT_BRANCH

# ==================== 部署阶段 ====================

echo ""
echo "🌐 5. 部署到 gh-pages 分支..."

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

# 恢复子模块工作目录（git checkout 不会自动恢复子模块）
echo ""
echo "🔄 恢复子模块状态..."
git submodule update --init --recursive
echo "   ✅ 子模块已恢复"

# ==================== 完成阶段 ====================

echo ""
echo "✅ 部署完成！"
echo "📍 博客地址: https://s-ai-unix.github.io/"
echo "⏳ 通常需要 1-3 分钟生效"
echo ""
echo "📊 部署统计："
echo "   - 总页面数: $PAGE_COUNT"
echo "   - 子模块状态: 已更新"
echo ""
echo "💡 提示："
echo "   - 使用 Ctrl+F5 (Windows) 或 Cmd+Shift+R (Mac) 强制刷新浏览器"
echo "   - 如果发现问题，检查: draft设置、文章日期、子模块状态"
