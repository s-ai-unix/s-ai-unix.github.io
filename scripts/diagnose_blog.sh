#!/bin/bash
# 快速诊断博客文章常见问题

ARTICLE="$1"

if [ -z "$ARTICLE" ]; then
    echo "❌ 用法: $0 <article.md>"
    exit 1
fi

if [ ! -f "$ARTICLE" ]; then
    echo "❌ 文件不存在: $ARTICLE"
    exit 1
fi

echo "🔍 诊断文章: $ARTICLE"
echo ""

# 1. 检查数学公式
echo "1️⃣  检查数学公式..."
python3 scripts/check_math_formulas.py "$ARTICLE" 2>&1 | head -20

# 2. 检查图片路径
echo ""
echo "2️⃣  检查图片路径..."
BAD_PATHS=$(grep -n '!\[.*\](images/[^/]' "$ARTICLE" 2>/dev/null)
if [ -n "$BAD_PATHS" ]; then
    echo "   ⚠️  发现可能的相对路径（应该以 / 开头）:"
    echo "$BAD_PATHS" | head -5
else
    echo "   ✅ 图片路径格式正确"
fi

# 3. 检查特殊字符
echo ""
echo "3️⃣  检查特殊字符..."
GREEK=$(grep -E 'α|β|γ|δ|θ|λ|μ|π|σ|φ|ω' "$ARTICLE" 2>/dev/null | wc -l)
if [ "$GREEK" -gt 0 ]; then
    echo "   ⚠️  发现 $GREEK 处希腊字母（应该使用 LaTeX 命令）"
    grep -n '[αβγδθλμπσφω]' "$ARTICLE" | head -3
else
    echo "   ✅ 没有发现直接使用的希腊字母"
fi

# 4. 检查 frontmatter
echo ""
echo "4️⃣  检查 frontmatter..."
if ! head -20 "$ARTICLE" | grep -q "^title:"; then
    echo "   ❌ 缺少 title"
else
    echo "   ✅ title 存在"
fi

if ! head -20 "$ARTICLE" | grep -q "^date:"; then
    echo "   ❌ 缺少 date"
else
    echo "   ✅ date 存在"
fi

if ! head -20 "$ARTICLE" | grep -q "^math: true"; then
    echo "   ⚠️  未启用 math（如果文章包含数学公式，应设置 math: true）"
fi

# 5. 检查 draft 状态
echo ""
echo "5️⃣  检查 draft 状态..."
if grep -q "^draft: true" "$ARTICLE"; then
    echo "   ⚠️  文章标记为草稿（draft: true），不会被发布"
else
    echo "   ✅ 文章可发布"
fi

echo ""
echo "✅ 诊断完成"
echo ""
echo "💡 下一步："
echo "   如果发现问题，请修复后重新检查"
echo "   本地预览: hugo server -D"
