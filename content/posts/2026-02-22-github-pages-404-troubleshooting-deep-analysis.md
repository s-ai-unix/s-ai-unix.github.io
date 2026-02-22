---
title: "GitHub Pages 404 故障排查：深度分析与根因定位"
date: 2026-02-22T18:45:00+08:00
draft: false
description: "从症状到根源：一次 GitHub Pages 404 故障的完整排查过程，展示如何通过 git 历史分析和系统性思维快速定位问题根因"
categories: ["系统管理"]
tags: ["GitHub", "DevOps", "调试方法", "故障排除"]
cover:
    image: "images/covers/github-pages-404-故障排查深度分析与根因定位-cover.png"
    alt: "GitHub Pages 404 故障排查"
    caption: "从症状到根源的系统性故障排查"
math: true
---

## 引言：当博客突然消失

![引言：当博客突然消失](/images/illustrations/2026-02-22-github-pages-404故障排查深度分析与根因定位-01.png)

2026年2月22日中午，我的技术博客 `https://s-ai-unix.github.io/` 突然显示 404 错误。上午还正常，中午就挂了。我没有在 github上做任何特殊操作，只是正常提交了几篇文章。

这场故障排查揭示了一个核心问题：**不同的问题解决方法，效果天差地别**。

---

## 第一章：问题诊断的两条路径

### 1.1 症状

访问 `https://s-ai-unix.github.io/` 显示 404 Not Found。GitHub Actions 失败日志：

```
Run #333 (UTC 2026-02-22 05:15:32Z)
Status: failed
Error: Failed to create deployment (status: 404)
Ensure GitHub Pages has been enabled
```

### 1.2 两条诊断路径

**路径 A（症状分析）：**

```
看到错误日志 → 判断 Pages 配置异常 → 建议用户手动修改 Settings
```

结论：需要用户手动操作。问题：**即使执行了所有操作，404 依然存在**。

**路径 B（根源分析）：**

```
看到错误日志 → 追溯 git 历史 → 查看 commit diff → 发现配置被改错 → 恢复配置
```

结论：**问题自动解决，无需任何手动操作**。

---

## 第二章：根源分析的核心步骤

### 2.1 追溯 git 历史

```bash
git log --oneline -10
```

输出：

```
6e4d588a Fix blog deployment mode and restore latest post assets/formats
91c88ce6 Update content
0a07fead Add 比安基恒等式 article and images
```

**关键发现**：commit `6e4d588a` 标题是 "Fix blog deployment mode"——这很可疑！

### 2.2 查看 commit diff

```bash
git diff 91c88ce6 6e4d588a -- .github/workflows/hugo.yaml
```

**之前的工作 workflow（正确）：**

```yaml
# GitHub Pages Infrastructure 模式
deploy:
  uses: actions/deploy-pages@v4  # ✅ 正确
```

**被错误修改为：**

```yaml
# gh-pages 分支模式
deploy:
  uses: peaceiris/actions-gh-pages@v4  # ❌ 错误
  with:
    publish_branch: gh-pages
```

### 2.3 根本原因

**GitHub Pages 有两种部署模式，必须与 Settings 匹配：**

| 模式 | Workflow Action | Settings 配置 |
| --- | --- | --- |
| **GitHub Pages Infrastructure** | `actions/deploy-pages@v4` | Source: GitHub Actions |
| **gh-pages 分支模式** | `peaceiris/actions-gh-pages@v4` | Source: Deploy from a branch |

**Commit 6e4d588a 把正确的模式改成了错误的模式**，导致与 GitHub Settings 不匹配！

### 2.4 解决方案

恢复正确的 workflow 配置，触发重新部署：

```bash
git commit --allow-empty -m "Trigger GitHub Actions deployment"
git push origin main
```

**结果**：1-3 分钟后，博客自动恢复正常。

---

## 第三章：为什么有的模型成功，有的失败？

### 3.1 核心差异

| 维度 | 失败方法 | 成功方法 |
| --- | --- | --- |
| **诊断方法** | 查看错误日志 | 分析 git commit 历史 + diff |
| **问题判断** | 配置异常（需手动修复） | Workflow 配置错误（可代码修复） |
| **用户操作** | 需要用户手动修改 | 完全自动，无需操作 |
| **最终结果** | 未解决 | 完全解决 |

### 3.2 思维模式差异

**线性思维（失败）：**

```
问题 → 症状 → 表面解决方案
 ↓        ↓         ↓
 404   日志错误   gh-pages模式
```

**系统性思维（成功）：**

```
问题 → 历史分析 → 模式识别 → 代码修复
 ↓        ↓          ↓         ↓
 404   commit 6e4...  两种模式   恢复配置
```

### 3.3 工具使用差异

| 工具 | 失败方法 | 成功方法 |
| --- | --- | --- |
| `git log` | 未使用 | 查看最近 commits |
| `git diff` | 未使用 | 查看 workflow 改动 |
| GitHub Actions log | 查看失败 | 分析 workflow 配置 |

**关键差异**：失败方法只看**错误日志**（症状），成功方法分析了**代码变更历史**（根因）。

---

## 第四章：方法论总结

### 4.1 智慧公式

```
问题解决 = 溯源历史 × 模式识别 × 自动修复
```

### 4.2 核心原则

1. **症状 ≠ 根因**：错误信息只是症状，必须追溯问题如何被引入
2. **历史很重要**：当前状态是历史变更的结果
3. **模式匹配**：部署模式必须与 Settings 匹配
4. **自动化 > 手动**：能通过代码修复，不让用户手动操作

### 4.3 调试路径

**错误路径**：

```
查看错误 → 搜索类似问题 → 给出解决方案 → 让用户执行
```

**正确路径**：

```
查看错误 → 分析 git 历史 → 定位根本原因 → 修复根本原因 → 验证修复
```

---

## 第五章：启示

### 5.1 为什么有些模型会犯错？

1. **只看症状，不看根因**：错误日志只是表象，代码变更历史才是根源
2. **缺少系统性知识**：不知道 GitHub Pages 有两种部署模式
3. **手动修复思维**：优先让用户手动操作，而非通过代码自动化修复

### 5.2 为什么有的模型能成功？

1. **深度分析能力**：不仅看症状，还看历史
2. **系统性知识**：理解部署模式与 Settings 的对应关系
3. **自动化优先思维**：通过代码修复而非手动操作

---

## 结语：从症状到根源的跃迁

> 🎯 **深度分析 > 表面观察**
> 🎯 **根源治理 > 症状缓解**
> 🎯 **自动化 > 手动操作**

在技术调试中，**git log 和 git diff 是追溯历史的根本工具**。错误日志告诉你"什么失败了"，git 历史告诉你"为什么失败"——这就是"治标"与"治本"的区别。

---

## 参考资源

- [GitHub Pages Documentation](https://docs.github.com/en/pages/)
- [GitHub Actions Deployment](https://docs.github.com/en/actions/deployment)
- [Git Log Documentation](https://git-scm.com/docs/git-log)
- [Git Diff Documentation](https://git-scm.com/docs/git-diff)
