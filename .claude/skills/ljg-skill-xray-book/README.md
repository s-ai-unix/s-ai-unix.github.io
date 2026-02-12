# ljg-skill-xray-book

深度拆书机 (Deep Book X-Ray) — 一个 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) Skill，基于 Epiplexity 原理，通过三轮认知压缩最大化提取书籍中的可学习结构。

## 功能

- 支持书名、书籍内容或链接输入
- 三轮认知压缩：骨架扫描 → 血肉解剖 → 灵魂提取
- 餐巾纸极限压缩：公式 + 草图 + 一句话
- 生成 Org-mode 格式报告，含 ASCII 结构图

## 安装

```bash
git clone https://github.com/lijigang/ljg-skill-xray-book.git ~/.claude/skills/ljg-xray-book
```

## 使用

在 Claude Code 中输入：

```
/ljg-xray-book 《思考，快与慢》
```

## 输出示例

生成的 Org-mode 报告包含：

- **NAPKIN** — 餐巾纸公式、一句话、草图
- **ROUND 1: SKELETON** — 骨架扫描，全局结构
- **ROUND 2: DISSECTION** — 血肉解剖，论证链条
- **ROUND 3: SOUL** — 灵魂提取，跨域迁移
- **STRUCTURE MAP** — ASCII 全书结构图

## License

MIT
