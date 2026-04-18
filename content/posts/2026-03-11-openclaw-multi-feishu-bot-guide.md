---
title: "OpenClaw 多飞书 Bot 实战：从零打造你的 AI 特工队"
date: 2026-03-11T14:30:00+08:00
draft: false
description: "手把手教你配置 OpenClaw 多飞书 Bot，从踩坑到实战，让你的 AI 助手各司其职、协同作战。"
categories: ["技术", "AI", "自动化"]
tags: ["OpenClaw", "飞书", "Bot", "Agent", "配置教程"]
cover:
    image: "images/agents-naming/feishu-agent-bot-vs-agent.jpg"
    alt: "飞书中的智能体与 Bot 界面示例"
    caption: "多角色 Bot 协作的一个实际界面"
math: true
---

## 前言：一个 Bot 不够玩

第一次把 OpenClaw 接上飞书时，我兴奋地问它："你能做什么？"

它回答："我是你的 AI 助手，可以帮你写代码、查资料、解答问题..."

听起来不错。用了一周后却发现，一个号称万能的助手，往往意味着什么都不精通。聊 EU AI Act 合规，它给不了专业建议；让它记住我的工作习惯，它又显得心不在焉。

所以我想，不如搞一支 AI 特工队。

主管 Bot 处理通用问题、任务分派。专家 Bot 负责 TIC 行业合规咨询。再加个助理 Bot 管日程、整理信息。每个角色有自己的知识库和回答风格。

听起来很酷？配置过程让我踩了一堆坑。这篇文章记录从零到成功的完整过程。

> OpenClaw 是开源 AI Agent 管理平台，支持 Claude、GPT 等大模型以独立 Agent 形式运行，可接入飞书、Telegram、Discord 等渠道。

---

## 第一章：踩坑实录

### 坑一：Bindings 直接写 appId

我的直觉是：加几个 Agent，在 bindings 里配一下路由就行。

```json
{
  "id": "expert",
  "name": "行业专家",
  "agentDir": "~/.openclaw/agents/expert/agent"
}
```

Bindings 配置：

```json
"bindings": [
  {
    "agentId": "main",
    "match": {
      "channel": "feishu",
      "accountId": "cli_xxxxx"
    }
  }
]
```

结果只有 main bot 能收到消息，其他全静默。

原因是 OpenClaw 2026.3.8 的 `channels.feishu` 默认只支持单个 `appId`。所有 bindings 都指向同一个 channel，但 channel 只连了第一个 bot。

### 坑二：用不存在的命令

网上教程说可以用 `openclaw channels add --agent expert --channel feishu` 添加渠道。我信心满满地敲下命令，终端提示命令不存在。尴尬。

网络教程可能是旧版本，动手前先验证命令是否存在。

### 坑三：正确的 accounts 结构

翻遍文档才找到正确姿势：

```json
"channels": {
  "feishu": {
    "enabled": true,
    "connectionMode": "websocket",
    "accounts": {
      "main": {
        "enabled": true,
        "appId": "<main-app-id>",
        "appSecret": "<main-secret>",
        "groupPolicy": "allowlist",
        "groupAllowFrom": ["<group-id>"],
        "requireMention": true
      },
      "expert": {
        "enabled": true,
        "appId": "<expert-app-id>",
        "appSecret": "<expert-secret>",
        "groupPolicy": "allowlist",
        "groupAllowFrom": ["<group-id>"],
        "requireMention": true
      }
    }
  }
}
```

这里有几个坑要注意：

1. **`accounts` 下的 key**（如 `"main"`、`"expert"`）是内部标识，不是飞书的 `appId`
2. **Bindings 里要用完整格式**：`"type": "route"` + `match` 对象
3. **`accountId` 对应 accounts 的 key**，不是飞书 AppId
4. **`default` 账户的特殊性**：如果顶层 `appId` 与某个 account 的 `appId` 相同，消息会路由到 `default` 账户。必须给 `default` 也配置 `groupAllowFrom`，或在 bindings 中添加 `default -> main` 路由

### 2.4 正确的 default 账户配置

如果你的 Main Bot 的 `appId` 同时出现在顶层和 `accounts.main`，需要额外处理：

```json
"channels": {
  "feishu": {
    "appId": "<main-app-id>",        // 顶层配置
    "accounts": {
      "default": {
        "groupAllowFrom": ["<group-id>"]   // 必须配置！
      },
      "main": {
        "appId": "<main-app-id>",          // 与顶层相同
        "groupAllowFrom": ["<group-id>"]
      },
      "expert": {
        "appId": "<expert-app-id>",        // 不同，正常路由
        "groupAllowFrom": ["<group-id>"]
      }
    }
  }
}
```

或者在 bindings 中添加双重路由：

```json
"bindings": [
  {"type": "route", "agentId": "main", "match": {"channel": "feishu", "accountId": "default"}},
  {"type": "route", "agentId": "main", "match": {"channel": "feishu", "accountId": "main"}},
  {"type": "route", "agentId": "expert", "match": {"channel": "feishu", "accountId": "expert"}}
]
```

---

## 第二章：完整配置指南

### 2.1 准备工作

在飞书开放平台（open.feishu.cn）完成：

创建企业自建应用（每个 Bot 一个）。获取凭证：App ID、App Secret、Verification Token、Encrypt Key。开启权限：获取用户基本信息、获取与发送单聊和群组消息、接收群聊中 @我或单聊消息。配置订阅：使用长链接接收事件，添加接收消息事件。发布应用。

### 2.2 创建 Agent 目录

```bash
mkdir -p ~/.openclaw/agents/expert/agent
mkdir -p ~/.openclaw/agents/assistant/agent
```

每个 Agent 需要两个核心文件。

SOUL.md（人格定义）：

```markdown
# Expert - 行业专家

## 角色
10年+ 行业经验，专注合规领域。

## 风格
- 专业严谨，有观点，直给可落地建议
- 默认中文，专业术语保留英文
```

AGENTS.md（工作规范）：

```markdown
# Expert 工作规范

## 专长
- 功能安全标准
- 合规咨询

## 知识库
- ~/docs/industry/standards
```

### 2.3 配置 openclaw.json

```json
{
  "agents": {
    "list": [
      {
        "id": "main",
        "default": true,
        "name": "Main Agent"
      },
      {
        "id": "expert",
        "name": "行业专家",
        "agentDir": "~/.openclaw/agents/expert/agent"
      },
      {
        "id": "assistant",
        "name": "个人助理",
        "agentDir": "~/.openclaw/agents/assistant/agent"
      }
    ]
  },
  "channels": {
    "feishu": {
      "enabled": true,
      "connectionMode": "websocket",
      "dmPolicy": "pairing",
      "accounts": {
        "main": {
          "enabled": true,
          "appId": "<your-main-app-id>",
          "appSecret": "<your-main-secret>",
          "groupPolicy": "allowlist",
          "groupAllowFrom": ["<your-group-id>"],
          "requireMention": true
        },
        "expert": {
          "enabled": true,
          "appId": "<your-expert-app-id>",
          "appSecret": "<your-expert-secret>",
          "groupPolicy": "allowlist",
          "groupAllowFrom": ["<your-group-id>"],
          "requireMention": true
        },
        "assistant": {
          "enabled": true,
          "appId": "<your-assistant-app-id>",
          "appSecret": "<your-assistant-secret>",
          "groupPolicy": "allowlist",
          "groupAllowFrom": ["<your-group-id>"],
          "requireMention": true
        }
      }
    }
  },
  "bindings": [
    {
      "type": "route",
      "agentId": "main",
      "match": {
        "channel": "feishu",
        "accountId": "main"
      }
    },
    {
      "type": "route",
      "agentId": "expert",
      "match": {
        "channel": "feishu",
        "accountId": "expert"
      }
    },
    {
      "type": "route",
      "agentId": "assistant",
      "match": {
        "channel": "feishu",
        "accountId": "assistant"
      }
    }
  ],
  "session": {
    "dmScope": "per-account-channel-peer",
    "reset": {
      "mode": "daily",
      "atHour": 4,
      "idleMinutes": 1440
    }
  },
  "tools": {
    "agentToAgent": {
      "enabled": true,
      "allow": ["main", "expert", "assistant"]
    },
    "sessions": {
      "visibility": "all"
    }
  },
  "agents": {
    "defaults": {
      "memorySearch": {
        "extraPaths": ["~/.openclaw/shared"]
      }
    }
  }
}
```

几个关键配置要注意：`session.dmScope` 设置为 `per-account-channel-peer` 实现会话隔离。`tools.agentToAgent` 启用 Agent 间协作，main 可以给其他 Agent 派活。`memorySearch.extraPaths` 设置共享知识库路径，实现跨场景记忆。`requireMention` 让群聊中必须 @bot 才响应，避免混乱。

---

## 第三章：进阶——跨场景记忆

### 3.1 问题：DM 和群聊是"两个世界"

配置好多 Bot 后，我发现一个尴尬的问题。

在私聊里告诉专家 Bot "我在做 EU AI Act 合规评估"，到群里 @它时，它却问 "请问您需要什么帮助？"

原因是 OpenClaw 的 session key 包含 `peerType`（direct vs group），DM 和群聊的上下文默认不共享。

### 3.2 解决方案：共享目录 + MEMORY.md

建立三层记忆体系：

```
~/.openclaw/
├── workspace/MEMORY.md      # 长期记忆（用户档案、项目背景）
├── shared/                   # 跨场景共享目录
│   ├── board.md             # 公告栏
│   ├── tasks.md             # 任务看板
│   ├── notes/               # 文档资料
│   └── projects/            # 项目资料
└── agents/<name>/agent/     # Agent 私有目录
```

SOUL.md 中配置记忆策略：

```markdown
## 跨场景记忆（DM 与群聊共享）

同一个 Bot 在 DM 和群聊中的上下文默认不共享。

解决方案：

共享目录（`~/.openclaw/shared/`）存放项目文档、待办事项、调研报告。每次回复前搜索 `shared/` 了解历史背景。

MEMORY.md 存放用户档案、长期偏好、不变的项目信息。适合存放缓慢变化的信息。

信息同步：DM 中获得的关键信息主动同步到 shared，群聊中引用 shared 中的项目背景。
```

MEMORY.md 模板：

```markdown
# MEMORY.md - 长期记忆档案

## 用户档案

基本信息
- 工作地：<城市>
- 职业：<职业>

工作背景
- 公司：<公司>
- 专注领域：<领域>

沟通偏好
- 框架先行，再细化，再落地
- 偏好比较研究
- 喜欢实操型答案

## 当前进行中的项目

项目名称
- 描述：...
- 状态：进行中

## 常用知识库

- ~/docs/project-a
- ~/.openclaw/shared/notes/
```

### 3.3 信息同步示例

DM 场景：
```
用户（DM）：我正在做 X 项目合规评估，下周要交
Bot：收到，记录在 shared/projects/x-project.md
```

群聊场景（后续）：
```
用户（群聊）：@Bot 帮我看看那个合规评估的进展
Bot：（搜索 shared/projects/x-project.md）
      根据记录，X 项目合规评估已完成风险分类，
      正在进行 Article 9 的符合性评估...
```

---

## 第四章：验证和测试

### 4.1 重启 Gateway

```bash
openclaw gateway restart
```

检查状态：

```bash
openclaw channels status --probe
```

预期看到：
```
- Feishu main: enabled, configured, running, works
- Feishu expert: enabled, configured, running, works
- Feishu assistant: enabled, configured, running, works
```

### 4.2 用户配对

第一次发消息时，Bot 会回复配对码：

```
OpenClaw: access not configured.
Pairing code: XXXXXX
Ask the bot owner to approve with:
openclaw pairing approve feishu XXXXXX
```

执行批准（每个 Bot 都需要）：

```bash
openclaw pairing approve feishu <pairing-code>
```

### 4.3 功能测试

分别问三个 Bot "你是谁"，确认回复的身份正确。

在群里测试 @不同的 Bot，确认只有被 @的 Bot 响应。

Agent 协作测试：
```
用户: @Main-Bot 帮我调研 XX 最新变化
Main Bot: （调用 Expert Agent）
Expert Bot: XX 的最新变化包括...
```

跨场景记忆测试：
1. DM 里告诉某个 Bot 重要信息
2. 群里 @同一个 Bot 询问相关信息
3. 确认 Bot 能回忆起 DM 中的内容

---

## 第五章：排坑指南（血泪教训）

### 坑四：Main Bot 不响应，日志显示 "blocked by group-level policy"

这是最隐蔽的坑，我花了整整一个下午才找到根因。

**症状**：所有新增的 Bot（wukong、yy 等）都正常，唯独 Main Bot 在群里 @它没反应。Gateway 日志显示：

```
feishu[default]: group oc_xxx blocked by group-level policy
```

**根本原因**：OpenClaw 的路由机制有一个"默认账户"概念。

当 `channels.feishu.appId`（顶层配置）与 `accounts.main.appId` 相同时，消息会被路由到 `default` 账户，而不是 `main` 账户。而 `default` 账户默认没有 `groupAllowFrom`，导致群消息被阻止。

**配置结构示意**：

```json
{
  "channels": {
    "feishu": {
      "appId": "cli_aaa",           // 顶层 appId
      "accounts": {
        "default": {},               // 继承顶层 appId，但没配 groupAllowFrom！
        "main": {
          "appId": "cli_aaa",        // 与顶层相同
          "groupAllowFrom": ["oc_xxx"]
        },
        "wukong": {
          "appId": "cli_bbb",        // 不同，正常路由
          "groupAllowFrom": ["oc_xxx"]
        }
      }
    }
  }
}
```

**解决方案一**：给 `default` 账户配置 `groupAllowFrom`

```json
"accounts": {
  "default": {
    "groupAllowFrom": ["<your-group-id>"]
  },
  "main": {
    "appId": "cli_aaa",
    "groupAllowFrom": ["<your-group-id>"]
  }
}
```

**解决方案二**：在 bindings 中添加 default -> main 路由

```json
"bindings": [
  {
    "type": "route",
    "agentId": "main",
    "match": {
      "channel": "feishu",
      "accountId": "default"    // 让 default 也路由到 main
    }
  },
  {
    "type": "route",
    "agentId": "main",
    "match": {
      "channel": "feishu",
      "accountId": "main"
    }
  }
]
```

**验证方法**：查看日志中的 session key

```bash
tail -f /tmp/openclaw/openclaw-*.log | grep -E "route|account|blocked"
```

正常应该看到：
```
feishu[main]: dispatching to agent:main:feishu:main:group:oc_xxx
```

异常时会看到：
```
feishu[default]: group oc_xxx blocked by group-level policy
```

### 坑五：群聊中 Bot 身份错乱

**症状**：@YY Bot 时，它自称 "AI_Main_Openclaw" 而不是 YY。

**原因**：群聊 session 使用的是 `workspace-yy/` 目录下的文件，而不是 `agents/yy/agent/` 目录。如果 workspace 目录没有正确同步 SOUL.md，Bot 会使用默认身份。

**解决方案**：同步身份文件到 workspace 目录

```bash
cp ~/.openclaw/agents/yy/agent/SOUL.md ~/.openclaw/workspace-yy/
cp ~/.openclaw/agents/yy/agent/AGENTS.md ~/.openclaw/workspace-yy/
```

### Bot 不回复（通用排查）

排查步骤：

```bash
# 检查配置格式
openclaw config validate

# 查看 Gateway 状态
openclaw channels status --probe

# 检查日志（关键！）
tail -f /tmp/openclaw/openclaw-*.log | grep -E "blocked|route|dispatching"

# 确认用户已批准
openclaw pairing list
```

### 三个 Bot 同时回复

`requireMention` 未启用。确保每个 account 都有 `"requireMention": true`。

### 会话上下文混乱

`dmScope` 配置不正确。设置为 `"per-account-channel-peer"`。

### Agent 间无法通信

`agentToAgent` 未启用，或 `sessions.visibility` 不是 `"all"`。

---

## 第六章：架构原理简述

### 消息流转

```
用户消息
    ↓
飞书服务器
    ↓
OpenClaw Gateway (WebSocket)
    ↓
Channel Router (根据 accountId 匹配)
    ↓
Binding Matcher (根据 bindings 配置)
    ↓
Agent Dispatcher (路由到对应 Agent)
    ↓
Agent 处理 (加载 SOUL.md + AGENTS.md + MEMORY.md)
    ↓
回复消息
```

### 会话隔离

Session key 格式：

```
agent:{agentId}:{channel}:{accountId}:{peerType}:{peerId}
```

示例：
- DM: `agent:main:feishu:main:direct:ou_xxx`
- 群聊: `agent:expert:feishu:expert:group:oc_xxx`

`per-account-channel-peer` 确保每个 Bot 的 session key 唯一，互不干扰。

---

## 结语

多 Bot 配置的核心在于理解路由和隔离。

`accounts` 配置多账号，`bindings` 实现路由。`dmScope` 控制会话隔离级别。`shared/` 目录 + `MEMORY.md` 实现跨场景记忆。`agentToAgent` 让 Bot 之间可以协作。

现在，去组建你的 AI 特工队吧。

---

## 参考

- OpenClaw 官方文档：https://docs.openclaw.ai
- 飞书开放平台：https://open.feishu.cn
