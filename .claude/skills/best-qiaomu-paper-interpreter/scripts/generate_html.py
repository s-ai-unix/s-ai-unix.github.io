#!/usr/bin/env python3
"""
将 Markdown 解读文章转换为独立的、苹果小清新风格 HTML 单文件
所有 CSS 和 JS 都内联到 HTML 中

默认仅生成 HTML（单文件，图片 base64 内嵌）。
"""

import sys
import os
import re
import base64
import html as _html
import markdown
from pathlib import Path
from datetime import datetime
from typing import Optional


class AppleStyleStandaloneHTMLGenerator:
    """苹果小清新风格独立 HTML 生成器 - Steven 风格"""

    def __init__(self, markdown_file, output_file):
        self.markdown_file = Path(markdown_file)
        self.output_file = Path(output_file)
        # ToC label style:
        # - number: compact numeric dots (default)
        # - phrase: short phrase labels
        self.toc_mode = (os.environ.get("STEVEN_TOC_MODE") or "number").strip().lower()
        if self.toc_mode not in ("number", "phrase"):
            self.toc_mode = "number"

        # 确保输出目录存在
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def find_paper_reading_root(self, start_dir: Path) -> Optional[Path]:
        """Best-effort locate Paper-Reading root.

        We commonly reference assets from the root (e.g. `01.inbox/...`) even when
        generating HTML from a nested paper directory. If we can't resolve an
        image path relative to the markdown file, we try again from this root.
        """
        p = start_dir.resolve()
        for _ in range(10):
            if (p / "01.inbox").exists():
                return p
            if p.parent == p:
                break
            p = p.parent
        return None

    def read_markdown(self):
        """读取 Markdown 文件"""
        with open(self.markdown_file, "r", encoding="utf-8") as f:
            return f.read()

    def extract_title(self, content):
        """提取 H1 标题"""
        match = re.search(r"^# (.+)$", content, re.MULTILINE)
        return match.group(1).strip() if match else "论文解读"

    def extract_paper_metadata(self, content):
        """从 markdown 开头提取论文元数据（标题、作者、发表等）"""
        metadata = {
            "paper_title": "",
            "authors": "",
            "publication": "",
            "institutions": "",
        }

        # 匹配论文标题
        title_match = re.search(r"> \*\*论文标题\*\*：(.+)", content)
        if title_match:
            metadata["paper_title"] = title_match.group(1).strip()

        # 匹配作者
        author_match = re.search(r"> \*\*作者\*\*：(.+)", content)
        if author_match:
            metadata["authors"] = author_match.group(1).strip()

        # 匹配发表信息
        pub_match = re.search(r"> \*\*发表\*\*：(.+)", content)
        if pub_match:
            metadata["publication"] = pub_match.group(1).strip()

        # 匹配机构
        inst_match = re.search(r"> \*\*机构\*\*：(.+)", content)
        if inst_match:
            metadata["institutions"] = inst_match.group(1).strip()

        return metadata

    def extract_metadata(self, content):
        """提取元数据"""
        # 先提取论文特定元数据
        paper_metadata = self.extract_paper_metadata(content)

        # 如果没有论文元数据，使用默认
        if paper_metadata["paper_title"]:
            title = paper_metadata["paper_title"]
        else:
            title = self.extract_title(content)

        metadata = {
            "title": title,
            "paper_title": paper_metadata["paper_title"],
            "authors": paper_metadata["authors"],
            "publication": paper_metadata["publication"],
            "institutions": paper_metadata["institutions"],
            "reading_time": self.estimate_reading_time(content),
        }
        return metadata

    def estimate_reading_time(self, content):
        """估计阅读时间"""
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", content))
        english_words = len(re.findall(r"[a-zA-Z]+", content))
        total_words = chinese_chars + english_words
        minutes = max(1, round(total_words / 300))
        return f"{minutes} 分钟阅读"

    def get_inline_css(self):
        """获取内联 CSS 样式"""
        return """<style>
:root {
    --primary-color: #007AFF;
    --secondary-color: #5856D6;
    --accent-color: #FF9500;
    --success-color: #34C759;
    --text-primary: #1D1D1F;
    --text-secondary: #86868B;
    --background: #FFFFFF;
    --background-secondary: #F5F5F7;
    --border-color: #D2D2D7;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.08);
    --shadow-lg: 0 12px 40px rgba(0,0,0,0.12);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 18px;
    --radius-xl: 28px;
}

@media (prefers-color-scheme: dark) {
    :root {
        --text-primary: #F5F5F7;
        --text-secondary: #A1A1A6;
        --background: #000000;
        --background-secondary: #1D1D1F;
        --border-color: #424245;
    }
}

/* Manual Theme Override - 手动主题切换（高优先级） */
html[data-theme="dark"] {
    --text-primary: #F5F5F7;
    --text-secondary: #A1A1A6;
    --background: #000000;
    --background-secondary: #1D1D1F;
    --border-color: #424245;
}

html[data-theme="dark"] .nav-toc {
    background: rgba(0,0,0,0.85) !important;
}

html[data-theme="dark"] .mermaid {
    background: #1D1D1F !important;
}

html[data-theme="dark"] img {
    opacity: 0.9;
}

html[data-theme="dark"] .theme-toggle {
    background: linear-gradient(135deg, #0A84FF 0%, #5E5CE6 100%);
    border-color: #409CFF;
    box-shadow: 0 6px 20px rgba(10,132,255,0.6);
}

html[data-theme="dark"] .theme-toggle svg {
    color: #FFFFFF;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 17px;
    line-height: 1.6;
    color: var(--text-primary);
    background: var(--background);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Header Hero Section */
.hero {
    background: linear-gradient(135deg, var(--background-secondary) 0%, var(--background) 100%);
    padding: 80px 20px 60px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(0,122,255,0.03) 0%, transparent 70%);
    animation: pulse 20s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 0.8; }
}

.hero-content {
    position: relative;
    z-index: 1;
    max-width: 800px;
    margin: 0 auto;
}

.hero h1 {
    font-size: 48px;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 20px;
    background: linear-gradient(135deg, var(--text-primary) 0%, var(--primary-color) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-meta {
    display: flex;
    justify-content: center;
    gap: 24px;
    color: var(--text-secondary);
    font-size: 15px;
    flex-wrap: wrap;
}

.hero-meta span {
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Navigation */
.nav-toc {
    position: sticky;
    top: 0;
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border-color);
    z-index: 100;
    padding: 16px 0;
}

@media (prefers-color-scheme: dark) {
    .nav-toc {
        background: rgba(0,0,0,0.85);
    }
}

.nav-toc-content {
    max-width: 800px;
    margin: 0 auto;
    padding: 8px 20px;
    display: flex;
    gap: 6px;
    overflow-x: auto;
    scrollbar-width: none;
    justify-content: center;
}

.nav-toc-content::-webkit-scrollbar {
    display: none;
}

.nav-toc a {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 13px;
    font-weight: 500;
    padding: 6px 12px;
    border-radius: 16px;
    transition: all 0.2s ease;
    white-space: nowrap;
    flex-shrink: 0;
}

.nav-toc a:hover {
    color: var(--primary-color);
    background: var(--background-secondary);
}

.nav-toc a.active {
    color: var(--primary-color);
    background: rgba(0,122,255,0.1);
}

/* ToC mode: number (compact dots) */
.nav-toc[data-toc-mode=\"number\"] .nav-toc-content {
    gap: 10px;
}

.nav-toc[data-toc-mode=\"number\"] a {
    width: 34px;
    height: 34px;
    padding: 0;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-variant-numeric: tabular-nums;
    border: 1px solid transparent;
    background: transparent;
}

.nav-toc[data-toc-mode=\"number\"] a:hover {
    background: var(--background-secondary);
    border-color: var(--border-color);
    color: var(--text-primary);
}

.nav-toc[data-toc-mode=\"number\"] a.active {
    background: var(--primary-color);
    border-color: var(--primary-color);
    color: #FFFFFF;
}

/* ToC mode: phrase (slightly cleaner pills) */
.nav-toc[data-toc-mode=\"phrase\"] a {
    border: 1px solid transparent;
}

.nav-toc[data-toc-mode=\"phrase\"] a:hover {
    border-color: var(--border-color);
}

/* Main Content */
.main-content {
    max-width: 800px;
    margin: 0 auto;
    padding: 60px 20px;
}

/* Typography */
h2 {
    font-size: 32px;
    font-weight: 700;
    margin: 60px 0 24px;
    color: var(--text-primary);
    scroll-margin-top: 80px;
}

h3 {
    font-size: 24px;
    font-weight: 600;
    margin: 40px 0 16px;
    color: var(--text-primary);
    scroll-margin-top: 80px;
}

h4 {
    font-size: 20px;
    font-weight: 600;
    margin: 32px 0 12px;
    color: var(--text-primary);
}

p {
    margin-bottom: 20px;
    color: var(--text-primary);
}

/* Blockquotes - Term Definitions */
blockquote {
    background: linear-gradient(135deg, rgba(0,122,255,0.05) 0%, rgba(88,86,214,0.05) 100%);
    border-left: 4px solid var(--primary-color);
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 20px 24px;
    margin: 24px 0;
    font-size: 16px;
}

blockquote p {
    margin-bottom: 0;
}

blockquote strong {
    color: var(--primary-color);
    font-weight: 600;
}

/* Lists */
ul, ol {
    margin: 20px 0;
    padding-left: 28px;
}

li {
    margin-bottom: 12px;
    color: var(--text-primary);
}

/* Links */
a {
    color: var(--primary-color);
    text-decoration: none;
    transition: opacity 0.2s;
}

a:hover {
    opacity: 0.8;
    text-decoration: underline;
}

/* Tables */
table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin: 24px 0;
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-color);
}

th {
    background: var(--background-secondary);
    font-weight: 600;
    text-align: left;
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
}

td {
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
}

tr:last-child td {
    border-bottom: none;
}

tr:hover td {
    background: var(--background-secondary);
}

/* Code */
code {
    font-family: "SF Mono", Monaco, "Cascadia Code", "Courier New", monospace;
    font-size: 14px;
    background: var(--background-secondary);
    padding: 3px 8px;
    border-radius: 6px;
    color: var(--secondary-color);
}

pre {
    background: var(--background-secondary);
    padding: 20px;
    border-radius: var(--radius-md);
    overflow-x: auto;
    margin: 24px 0;
}

pre code {
    background: none;
    padding: 0;
    color: var(--text-primary);
}

/* Images - 统一占满内容宽度 */
img {
    width: 100%;
    height: auto;
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-md);
    margin: 24px 0;
    display: block;
}

/* Click-to-zoom for dense figures/tables (e.g., standard tables) */
.main-content img {
    cursor: zoom-in;
}

.img-modal {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.82);
    display: none;
    align-items: center;
    justify-content: center;
    padding: 24px;
    z-index: 3000;
}

.img-modal.open {
    display: flex;
}

.img-modal-inner {
    position: relative;
    width: min(96vw, 1400px);
    height: min(96vh, 900px);
    display: flex;
    align-items: center;
    justify-content: center;
}

.img-modal img {
    width: auto;
    height: auto;
    max-width: 96vw;
    max-height: 96vh;
    margin: 0;
    border-radius: 12px;
    box-shadow: 0 18px 60px rgba(0,0,0,0.5);
    cursor: zoom-out;
}

.img-modal-close {
    position: absolute;
    top: -12px;
    right: -12px;
    width: 40px;
    height: 40px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.25);
    background: rgba(0,0,0,0.6);
    color: #fff;
    font-size: 22px;
    line-height: 1;
    cursor: pointer;
    display: grid;
    place-items: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}

@media (max-width: 480px) {
    .img-modal {
        padding: 12px;
    }
    .img-modal-close {
        top: -8px;
        right: -8px;
    }
}

/* 图片容器 - 用于图表和插图的统一展示 */
figure {
    margin: 24px 0;
    text-align: center;
}

figure img {
    margin: 0;
}

figcaption {
    color: var(--text-secondary);
    font-size: 14px;
    margin-top: 12px;
    text-align: center;
}

/* Mermaid Diagrams */
.mermaid {
    background: var(--background-secondary);
    padding: 24px;
    border-radius: var(--radius-md);
    margin: 24px 0;
    text-align: center;
    overflow-x: auto;
}

/* Highlight / Strong */
strong {
    font-weight: 600;
    color: var(--text-primary);
}

/* Horizontal Rule */
hr {
    border: none;
    height: 1px;
    background: var(--border-color);
    margin: 48px 0;
}

/* Progress Indicator */
.reading-progress {
    position: fixed;
    top: 0;
    left: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    z-index: 1000;
    transition: width 0.1s;
    width: 0%;
}

/* Back to Top Button */
.back-to-top {
    position: fixed;
    bottom: 30px;
    right: 30px;
    width: 50px;
    height: 50px;
    background: var(--background);
    border: 1px solid var(--border-color);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: var(--shadow-md);
    opacity: 0;
    visibility: hidden;
    transition: all 0.3s ease;
    z-index: 999;
}

.back-to-top.visible {
    opacity: 1;
    visibility: visible;
}

.back-to-top:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

/* Theme Toggle Button - 主题切换按钮 */
.theme-toggle {
    position: fixed;
    bottom: 30px;
    left: 30px;
    width: 56px;
    height: 56px;
    background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
    border: 3px solid #0051D5;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 6px 20px rgba(0,122,255,0.5);
    transition: all 0.3s ease;
    z-index: 10000;
    opacity: 1;
    visibility: visible;
}

.theme-toggle:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.theme-toggle svg {
    width: 24px;
    height: 24px;
    color: #FFFFFF;
    transition: transform 0.3s ease;
    pointer-events: none;
}

.theme-toggle:hover svg {
    transform: rotate(15deg);
}

/* 隐藏/显示图标 */
.theme-toggle .sun-icon {
    display: none;
}

.theme-toggle .moon-icon {
    display: block;
}

html[data-theme="dark"] .theme-toggle .sun-icon {
    display: block;
}

html[data-theme="dark"] .theme-toggle .moon-icon {
    display: none;
}

@media (max-width: 768px) {
    .theme-toggle {
        bottom: 20px;
        left: 20px;
        width: 48px;
        height: 48px;
    }

    .theme-toggle svg {
        width: 22px;
        height: 22px;
    }
}

/* Footer */
.footer {
    background: var(--background-secondary);
    padding: 40px 20px;
    text-align: center;
    color: var(--text-secondary);
    font-size: 14px;
}

/* Responsive */
@media (max-width: 768px) {
    .hero {
        padding: 60px 16px 40px;
    }
    
    .hero h1 {
        font-size: 28px;
    }
    
    h2 {
        font-size: 24px;
    }
    
    h3 {
        font-size: 20px;
    }
    
    .hero-meta {
        flex-direction: column;
        gap: 8px;
    }
    
    .main-content {
        padding: 40px 16px;
    }
    
    .nav-toc-content {
        padding: 6px 16px;
        justify-content: flex-start;
    }
    
    .back-to-top {
        bottom: 20px;
        right: 20px;
        width: 44px;
        height: 44px;
    }
}

/* Print Styles */
@media print {
    .nav-toc,
    .back-to-top,
    .theme-toggle,
    .reading-progress {
        display: none;
    }
    
    body {
        font-size: 12pt;
    }
    
    .hero {
        padding: 40px 20px;
    }
}
</style>"""

    def get_inline_js(self):
        """获取内联 JavaScript"""
        return """<script>
// Reading Progress
document.addEventListener('scroll', function() {
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = (scrollTop / docHeight) * 100;
    document.querySelector('.reading-progress').style.width = progress + '%';
});

// Back to Top
const backToTop = document.querySelector('.back-to-top');
document.addEventListener('scroll', function() {
    if (window.scrollY > 500) {
        backToTop.classList.add('visible');
    } else {
        backToTop.classList.remove('visible');
    }
});

backToTop.addEventListener('click', function() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Active Nav Link
const sections = document.querySelectorAll('h2, h3');
const navLinks = document.querySelectorAll('.nav-toc a');

document.addEventListener('scroll', function() {
    let current = '';
    let minDistance = Infinity;
    
    sections.forEach(section => {
        const distance = Math.abs(section.getBoundingClientRect().top - 100);
        if (distance < minDistance) {
            minDistance = distance;
            current = section.querySelector('span')?.id || section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + current) {
            link.classList.add('active');
        }
    });
});

// Smooth Scroll for Nav Links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = this.getAttribute('href').substring(1);
        const target = document.querySelector(`span[id="${targetId}"]`) || 
                      document.getElementById(targetId);
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// Image Zoom (click-to-open modal)
(function() {
    const modal = document.getElementById('img-modal');
    const modalImg = document.getElementById('img-modal-img');
    const closeBtn = document.getElementById('img-modal-close');
    if (!modal || !modalImg || !closeBtn) return;

    function openModal(src, alt) {
        modalImg.src = src;
        modalImg.alt = alt || 'image';
        modal.classList.add('open');
        document.body.style.overflow = 'hidden';
    }

    function closeModal() {
        modal.classList.remove('open');
        modalImg.src = '';
        document.body.style.overflow = '';
    }

    document.querySelectorAll('.main-content img').forEach(img => {
        img.addEventListener('click', (e) => {
            e.preventDefault();
            openModal(img.getAttribute('src'), img.getAttribute('alt'));
        });
    });

    closeBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
})();

// Mermaid Diagrams - strict render + quality gate
let __mermaidRenderScheduled = false;

function __isDarkMode() {
    return document.documentElement.getAttribute('data-theme') === 'dark' ||
        (!document.documentElement.getAttribute('data-theme') &&
            window.matchMedia &&
            window.matchMedia('(prefers-color-scheme: dark)').matches);
}

function __decodeB64Utf8(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new TextDecoder('utf-8').decode(bytes);
}

function scheduleMermaidRender(reason) {
    if (__mermaidRenderScheduled) return;
    __mermaidRenderScheduled = true;
    setTimeout(() => {
        __mermaidRenderScheduled = false;
        renderMermaidDiagrams(reason || 'scheduled');
    }, 0);
}

function __basicMermaidHeuristics(code) {
    // Fail-closed on obviously risky diagrams (huge / too dense / likely overflow).
    if (!code) return { ok: false, reason: 'empty mermaid code' };
    if (code.length > 20000) return { ok: false, reason: 'diagram too large (>20k chars)' };
    const lines = code.split(/\\r?\\n/).filter(l => l.trim().length > 0);
    if (lines.length > 250) return { ok: false, reason: 'diagram too long (>250 lines)' };
    // Guard against mermaid pitfalls that often produce misleading charts or broken renders.
    if (/\\[\\s*\\d+[\\.|、]/.test(code)) return { ok: false, reason: 'node label looks like a markdown list (unsupported)' };
    return { ok: true };
}

async function renderMermaidDiagrams(reason) {
    if (typeof mermaid === 'undefined') {
        setTimeout(() => renderMermaidDiagrams(reason), 100);
        return;
    }

    // Always restore diagram source from data attribute before rendering.
    document.querySelectorAll('.mermaid[data-mermaid-base64]').forEach(div => {
        try {
            const b64 = div.getAttribute('data-mermaid-base64') || '';
            const code = __decodeB64Utf8(b64);
            div.textContent = code;
        } catch (e) {
            console.error('Failed to decode mermaid base64:', e);
        }
    });

    const dark = __isDarkMode();
    const theme = dark ? 'dark' : 'default';

    // Initialize Mermaid with HTML labels enabled so "<br/>" in labels works for wrapping.
    mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'loose',
        theme,
        flowchart: {
            htmlLabels: true,
            useMaxWidth: true,
            // More spacing reduces overlap/overflow in dense diagrams.
            nodeSpacing: 40,
            rankSpacing: 60
        },
        themeVariables: {
            fontFamily: '-apple-system,BlinkMacSystemFont,\"Segoe UI\",Helvetica,Arial,\"PingFang SC\",\"Hiragino Sans GB\",\"Noto Sans CJK SC\",sans-serif',
            fontSize: '14px',
            // Padding helps keep text inside node boxes.
            nodePadding: 12
        }
    });

    // Validate (syntax + guardrails). Fail closed: don't render risky diagrams.
    const mermaidDivs = Array.from(document.querySelectorAll('.mermaid'));
    for (const div of mermaidDivs) {
        div.classList.remove('mermaid-error');
        const code = (div.textContent || '').trim();
        const h = __basicMermaidHeuristics(code);
        if (!h.ok) {
            div.classList.add('mermaid-error');
            div.style.background = '#ffebee';
            div.style.padding = '16px';
            div.style.borderRadius = '8px';
            div.style.color = '#c62828';
            div.textContent = `Mermaid 图未通过质量闸门: ${h.reason}`;
            continue;
        }
        try {
            // Mermaid v10: parse may throw (or be async). Treat as strict gate.
            await mermaid.parse(code);
        } catch (e) {
            div.classList.add('mermaid-error');
            div.style.background = '#ffebee';
            div.style.padding = '16px';
            div.style.borderRadius = '8px';
            div.style.color = '#c62828';
            div.textContent = `Mermaid 语法错误，已阻止渲染（避免误导）: ${e && e.message ? e.message : String(e)}`;
        }
    }

    // Re-render only valid diagrams.
    document.querySelectorAll('.mermaid').forEach(el => el.removeAttribute('data-processed'));
    try {
        await mermaid.run({ querySelector: '.mermaid:not(.mermaid-error)' });
    } catch (e) {
        console.error('Mermaid run failed:', e);
    }
}

// DOM ready -> render diagrams
document.addEventListener('DOMContentLoaded', () => scheduleMermaidRender('dom'));
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    scheduleMermaidRender('readyState');
}

// Theme Toggle - 主题切换功能
(function() {
    const themeToggle = document.getElementById('theme-toggle');
    const html = document.documentElement;
    
    // 从 localStorage 读取保存的主题
    const savedTheme = localStorage.getItem('theme');
    const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    // 设置初始主题
    if (savedTheme === 'dark' || (!savedTheme && systemDark)) {
        html.setAttribute('data-theme', 'dark');
    }
    
    // 切换主题函数
    function toggleTheme() {
        const currentTheme = html.getAttribute('data-theme');
        if (currentTheme === 'dark') {
            html.removeAttribute('data-theme');
            localStorage.setItem('theme', 'light');
        } else {
            html.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
        }
        scheduleMermaidRender('theme-toggle');
    }
    
    // 绑定点击事件
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
    
    // 监听系统主题变化
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (!localStorage.getItem('theme')) {
            if (e.matches) {
                html.setAttribute('data-theme', 'dark');
            } else {
                html.removeAttribute('data-theme');
            }
            scheduleMermaidRender('system-theme-change');
        }
    });
})();
</script>"""

    def generate_toc(self, content):
        """生成目录导航

        Mode:
        - number: compact numeric labels (default)
        - phrase: short phrase labels
        """
        headers = re.findall(r"^## (.+)$", content, re.MULTILINE)
        toc_items = []

        for idx, header in enumerate(headers):
            anchor = self.slugify(header)
            full = _html.escape(header.strip(), quote=True)
            if self.toc_mode == "number":
                label = str(idx + 1)
            else:
                # 将长标题转换为简短短语
                needs_numbering = self.detect_needs_numbering(headers)
                label = self.get_short_label(header, idx, needs_numbering)
            label = _html.escape(label, quote=True)
            toc_items.append(
                f'<a href="#{anchor}" title="{full}" aria-label="{full}">{label}</a>'
            )

            # 最多显示12个章节，避免导航过长
            if len(toc_items) >= 12:
                break

        return "\n            ".join(toc_items)

    def detect_needs_numbering(self, headers):
        """检测文章是否需要自动添加章节编号"""
        chinese_nums = [
            "引言",
            "前言",
            "一",
            "二",
            "三",
            "四",
            "五",
            "六",
            "七",
            "八",
            "九",
            "十",
        ]
        chapter_patterns = ["第", "章", "部分", "Part", "Chapter"]

        has_numbering = 0
        for header in headers[:5]:  # 检查前5个标题
            # 检查是否以中文数字开头
            if any(header.startswith(num) for num in chinese_nums):
                has_numbering += 1
                continue
            # 检查是否包含章节标识
            if any(pattern in header for pattern in chapter_patterns):
                has_numbering += 1
                continue
            # 检查是否以阿拉伯数字开头
            if re.match(r"^\d+[.．、\s]", header):
                has_numbering += 1
                continue

        # 如果前5个标题中少于2个有编号，则认为需要自动编号
        return has_numbering < 2

    def get_short_label(self, header, index=0, needs_numbering=False):
        """将完整标题转换为简短的导航标签

        Args:
            header: 章节标题
            index: 章节索引（用于自动编号）
            needs_numbering: 是否需要自动编号
        """
        header = header.strip()

        # 匹配"第X章：..."格式（中文数字）
        chapter_match = re.match(r"(第[一二三四五六七八九十]+章)[：:：]", header)
        if chapter_match:
            return chapter_match.group(1)

        # 匹配"第X章 ..."格式（阿拉伯数字）
        chapter_num_match = re.match(r"(第\s*\d+\s*章)", header)
        if chapter_num_match:
            return chapter_num_match.group(1).replace(" ", "")

        # 匹配"第X部分：..."格式
        part_match = re.match(r"(第[一二三四五六七八九十]+部分)[：:：]", header)
        if part_match:
            return part_match.group(1)

        # 匹配"第X部分 ..."格式（阿拉伯数字）
        part_num_match = re.match(r"(第\s*\d+\s*部分)", header)
        if part_num_match:
            return part_num_match.group(1).replace(" ", "")

        # 匹配单个中文数字开头："一、..." "二、..."
        chinese_num_match = re.match(r"([一二三四五六七八九十])[、.．\s]", header)
        if chinese_num_match:
            return f"{chinese_num_match.group(1)}"

        # 匹配阿拉伯数字开头："1. ..." "2. ..."
        arabic_num_match = re.match(r"(\d+)[.．、\s]", header)
        if arabic_num_match:
            return f"{arabic_num_match.group(1)}"

        # 匹配"引言：..."格式
        if header.startswith("引言") or header.startswith("前言"):
            return "引言"

        # 匹配"结语..."或"总结..."格式
        if (
            header.startswith("结语")
            or header.startswith("总结")
            or header.startswith("结论")
        ):
            return "结语"

        # 匹配"Chapter X: ..."格式
        chapter_en_match = re.match(r"Chapter\s+(\d+)", header, re.IGNORECASE)
        if chapter_en_match:
            return f"第{chapter_en_match.group(1)}章"

        # 匹配"Part X: ..."格式
        part_en_match = re.match(r"Part\s+(\d+|[IVX]+)", header, re.IGNORECASE)
        if part_en_match:
            return f"Part {part_en_match.group(1)}"

        # 如果需要自动编号，返回中文数字编号
        if needs_numbering:
            chinese_nums = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
            if index < len(chinese_nums):
                return chinese_nums[index]
            else:
                return f"{index + 1}"

        # 如果无法识别，返回前6个字符（避免过长）
        return header[:6] + "..." if len(header) > 6 else header

    def slugify(self, text):
        """将标题转换为 URL 友好的锚点"""
        return re.sub(r"[^\w\s-]", "", text).strip().replace(" ", "-")

    def convert_markdown_to_html(self, content):
        """转换 Markdown 为 HTML"""
        md = markdown.Markdown(extensions=["fenced_code", "tables", "toc", "nl2br"])

        # 预处理：确保标题有 ID
        content = self.add_header_ids(content)

        # 转换
        html = md.convert(content)

        # 后处理
        html = self.post_process_html(html)

        return html

    def add_header_ids(self, content):
        """为标题添加 ID 属性"""

        def replace_header(match):
            level = len(match.group(1))
            title = match.group(2)
            anchor = self.slugify(title)
            return f'{"#" * level} <span id="{anchor}">{title}</span>'

        return re.sub(r"^(#{2,4}) (.+)$", replace_header, content, flags=re.MULTILINE)

    def get_image_mime_type(self, filepath):
        """根据文件扩展名获取 MIME 类型"""
        ext = Path(filepath).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".webp": "image/webp",
        }
        return mime_types.get(ext, "image/jpeg")

    def file_to_base64(self, filepath):
        """将文件转换为 base64 编码"""
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode("utf-8")
        except Exception as e:
            print(f"警告: 无法读取文件 {filepath}: {e}")
            return None

    def embed_images_as_base64(self, html):
        """将 HTML 中的本地图片引用转换为 base64 编码内嵌"""
        base_dir = self.markdown_file.parent
        root_dir = self.find_paper_reading_root(base_dir)

        def replace_img_src(match):
            full_tag = match.group(0)
            src_match = re.search(r'src=["\']([^"\']+)["\']', full_tag)
            if not src_match:
                return full_tag

            src = src_match.group(1)

            # 跳过已经是 data URL 或绝对 URL 的图片
            if (
                src.startswith("data:")
                or src.startswith("http://")
                or src.startswith("https://")
            ):
                return full_tag

            # If src is an absolute filesystem path, use it directly.
            try:
                src_path = Path(src)
                if src_path.is_absolute() and src_path.exists():
                    img_path = src_path
                else:
                    img_path = None
            except Exception:
                img_path = None

            # 构建图片的绝对路径
            if img_path is None:
                if src.startswith("/"):
                    # Treat leading slash as "root-relative" within the Paper-Reading root
                    # (not the OS root), because many markdown exporters do that.
                    img_path = base_dir / src.lstrip("/")
                else:
                    img_path = base_dir / src

            # 如果文件不存在，尝试其他可能的路径
            if not img_path.exists():
                # 尝试直接相对于当前目录
                alt_path = base_dir / Path(src).name
                if alt_path.exists():
                    img_path = alt_path
                else:
                    # 尝试在 images 子目录中查找
                    images_dir = base_dir / "images"
                    if images_dir.exists():
                        alt_path = images_dir / Path(src).name
                        if alt_path.exists():
                            img_path = alt_path
                    # Try again from Paper-Reading root if we detected one.
                    if not img_path.exists() and root_dir is not None:
                        alt_path = root_dir / src.lstrip("/")
                        if alt_path.exists():
                            img_path = alt_path
                    if not img_path.exists():
                        print(f"警告: 图片文件不存在: {img_path}")
                        return full_tag

            # 转换为 base64
            base64_data = self.file_to_base64(img_path)
            if base64_data:
                mime_type = self.get_image_mime_type(img_path)
                data_url = f"data:{mime_type};base64,{base64_data}"
                new_tag = full_tag.replace(f'src="{src}"', f'src="{data_url}"').replace(
                    f"src='{src}'", f'src="{data_url}"'
                )
                print(f"✓ 已内嵌图片: {src} ({len(base64_data)} bytes base64)")
                return new_tag

            return full_tag

        # 匹配所有 img 标签
        return re.sub(r"<img[^>]+>", replace_img_src, html)

    def post_process_html(self, html):
        """HTML 后处理"""
        # 为图片添加懒加载
        html = re.sub(r"<img ", r'<img loading="lazy" ', html)

        # 将本地图片转换为 base64 内嵌（生成完全独立的单文件）
        html = self.embed_images_as_base64(html)

        # 转换 Mermaid 代码块为可渲染的 div
        # 将 <pre><code class="language-mermaid">...</code></pre> 转换为 <div class="mermaid">...</div>

        def autowrap_mermaid_labels(code: str) -> str:
            """Best-effort Mermaid label wrapping to avoid text overflowing node shapes.

            This is intentionally conservative: it only wraps long, plain labels
            inside common node shapes ([...], (...), {...]) and skips anything that
            already contains explicit line breaks or HTML tags.
            """
            if not code or len(code) < 16:
                return code

            # Avoid touching code that already looks heavily formatted.
            if "<br" in code or "\\n" in code:
                return code

            def wrap_text(t: str) -> str:
                t = t.strip()
                if len(t) <= 24:
                    return t
                # Prefer wrapping at spaces / punctuation, fallback to fixed width.
                # For CJK, fixed width works reasonably.
                chunks = []
                cur = ""
                # Try to wrap around ~12-14 chars for readability.
                limit = 14
                for ch in t:
                    cur += ch
                    if len(cur) >= limit:
                        # If next boundary exists soon, don't split mid-word for Latin.
                        chunks.append(cur.strip())
                        cur = ""
                if cur.strip():
                    chunks.append(cur.strip())
                # Keep at most 3 lines to avoid huge nodes.
                chunks = chunks[:3]
                return "<br/>".join(chunks)

            # Replace node labels in square brackets / parentheses / braces.
            def repl_sq(m):
                inner = m.group(1)
                if len(inner) <= 24:
                    return m.group(0)
                if "<" in inner or ">" in inner or "$" in inner:
                    return m.group(0)
                return "[" + wrap_text(inner) + "]"

            def repl_paren(m):
                inner = m.group(1)
                if len(inner) <= 24:
                    return m.group(0)
                if "<" in inner or ">" in inner or "$" in inner:
                    return m.group(0)
                return "(" + wrap_text(inner) + ")"

            def repl_brace(m):
                inner = m.group(1)
                if len(inner) <= 24:
                    return m.group(0)
                if "<" in inner or ">" in inner or "$" in inner:
                    return m.group(0)
                return "{" + wrap_text(inner) + "}"

            # Match literal node labels like A[...], B(...), C{...}
            # Note: the Mermaid source here is plain text; do NOT over-escape.
            code2 = re.sub(r"\[([^\[\]]{25,})\]", repl_sq, code)
            code2 = re.sub(r"\(([^()]{25,})\)", repl_paren, code2)
            code2 = re.sub(r"\{([^{}]{25,})\}", repl_brace, code2)
            return code2

        def replace_mermaid_block(match):
            # Extract Mermaid code (still HTML-escaped at this stage) then decode entities.
            code_content = match.group(1)
            code_content = code_content.replace("&lt;", "<").replace("&gt;", ">")
            code_content = code_content.replace("&amp;", "&").replace("&quot;", '"')
            code_content = code_content.replace("&#39;", "'").replace("&apos;", "'")

            # Quality assist: wrap long labels to avoid overflow.
            code_content = autowrap_mermaid_labels(code_content).strip()

            # IMPORTANT:
            # - Do NOT inline the Mermaid code as HTML content, because we may inject "<br/>"
            #   which the browser would treat as a real <br> tag and destroy the diagram source.
            # - Store the source as base64, then JS will set `textContent` before rendering.
            code_b64 = base64.b64encode(code_content.encode("utf-8")).decode("ascii")
            return f'<div class="mermaid" data-mermaid-base64="{code_b64}"></div>'

        # 匹配 mermaid 代码块
        html = re.sub(
            r'<pre><code class="language-mermaid">(.*?)</code></pre>',
            replace_mermaid_block,
            html,
            flags=re.DOTALL,
        )

        return html

    def generate_html_template(self, content, metadata):
        """生成完整独立 HTML 模板"""
        title = metadata["title"]
        paper_title = metadata.get("paper_title", "")
        authors = metadata.get("authors", "")
        publication = metadata.get("publication", "")
        institutions = metadata.get("institutions", "")
        reading_time = metadata["reading_time"]

        # 生成作者和发表信息的 HTML（如果有）
        meta_html = ""
        if authors:
            meta_html += f"<span>👥 {authors}</span>"
        if publication:
            meta_html += f"<span>📅 {publication}</span>"

        # 生成导航目录
        toc_html = self.generate_toc(content)

        # 转换 Markdown 为 HTML
        html_content = self.convert_markdown_to_html(content)

        # 内联 CSS 和 JS
        inline_css = self.get_inline_css()
        inline_js = self.get_inline_js()

        return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="{title} - Steven 论文解读">
    <title>{title}</title>
    {inline_css}
    <!-- Mermaid (official) for diagrams -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    
    <!-- MathJax for LaTeX Math Support -->
    <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true,
        processEnvironments: true
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
      }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    
    <style>
    /* MathJax 样式优化 */
    mjx-container {{
        font-size: 1.1em !important;
        margin: 0.5em 0;
    }}
    mjx-container[display="true"] {{
        display: block;
        text-align: center;
        margin: 1.5em 0;
        overflow-x: auto;
    }}
    mjx-container:not([display="true"]) {{
        display: inline;
    }}
    /* 行内公式垂直对齐 */
    mjx-container[display="false"] {{
        vertical-align: middle;
    }}
    /* 暗黑模式适配 */
    @media (prefers-color-scheme: dark) {{
        mjx-container {{
            color: var(--text-primary) !important;
        }}
    }}
    html[data-theme="dark"] mjx-container {{
        color: var(--text-primary) !important;
    }}
    </style>
</head>
<body>
    <div class="reading-progress"></div>
    
    <header class="hero">
        <div class="hero-content">
            <h1>{title}</h1>
            <div class="hero-meta">
                <span>📄 Steven 论文解读</span>
                {meta_html}
                <span>⏱️ {reading_time}</span>
            </div>
        </div>
    </header>
    
    <nav class="nav-toc" data-toc-mode="{self.toc_mode}">
        <div class="nav-toc-content">
            {toc_html}
        </div>
    </nav>
    
    <main class="main-content">
{html_content}
    </main>
    
    <footer class="footer">
        <p>Generated by Steven 论文解读 | 用通俗的语言理解复杂的论文</p>
    </footer>
    
    <!-- Theme Toggle Button - 主题切换按钮 -->
    <button id="theme-toggle" class="theme-toggle" aria-label="切换主题">
        <!-- Moon Icon (for light mode) -->
        <svg class="moon-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
        <!-- Sun Icon (for dark mode) -->
        <svg class="sun-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="5"/>
            <line x1="12" y1="1" x2="12" y2="3"/>
            <line x1="12" y1="21" x2="12" y2="23"/>
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
            <line x1="1" y1="12" x2="3" y2="12"/>
            <line x1="21" y1="12" x2="23" y2="12"/>
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
        </svg>
    </button>
    
    <button class="back-to-top" aria-label="回到顶部">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 15l-6-6-6 6"/>
        </svg>
    </button>

    <!-- Image modal (for zooming dense tables/figures) -->
    <div id="img-modal" class="img-modal" role="dialog" aria-modal="true" aria-label="Image preview">
        <div class="img-modal-inner">
            <button id="img-modal-close" class="img-modal-close" aria-label="关闭">×</button>
            <img id="img-modal-img" src="" alt="">
        </div>
    </div>
    
    {inline_js}
</body>
</html>
'''

    def save_file(self, html_content):
        """保存独立 HTML 文件"""
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return self.output_file

    def generate(self):
        """主生成流程"""
        # 读取 Markdown
        content = self.read_markdown()

        # 提取元数据
        metadata = self.extract_metadata(content)

        # 生成 HTML
        html = self.generate_html_template(content, metadata)

        # 保存文件
        output_path = self.save_file(html)

        file_size = output_path.stat().st_size
        print(f"✅ Steven 风格独立 HTML 生成完成: {output_path}")
        print(f"   文件大小: {file_size / 1024:.1f} KB")
        print(f"   风格特点: 通俗易懂、深入浅出、像朋友聊天")

        return output_path


def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_html.py <markdown_file> <output_file>")
        print("Example: python generate_html.py paper.md output.html")
        sys.exit(1)

    markdown_file = sys.argv[1]
    output_file = sys.argv[2]

    generator = AppleStyleStandaloneHTMLGenerator(markdown_file, output_file)
    generator.generate()


if __name__ == "__main__":
    main()
