---
name: universally-updated-skill
description: Skill synchronization tool for keeping skills in sync across multiple directories. Use when: (1) Creating a new skill that should be available in multiple environments (Claude Code, OpenCode, Hugo Blog), (2) Updating an existing skill and syncing changes to all target directories, (3) Listing available skills that can be synced, (4) Ensuring skill consistency across ~/.claude/skills, ~/.config/opencode/skill, ~/.config/opencode/skills, and Hugo Blog .claude/skills directories.
---

# universally-updated-skill

Synchronize skills across multiple directories to ensure they are available in all your environments.

## Quick Start

```bash
# List all available skills
scripts/sync-skill.sh --list

# Sync an existing skill to all directories
scripts/sync-skill.sh --name <skill-name>

# Create a new skill and sync it
scripts/sync-skill.sh --new <skill-name>
```

## Target Directories

Skills are synchronized to these locations:

| Directory | Environment |
|-----------|-------------|
| `~/.claude/skills` | Claude Code (primary) |
| `~/.config/opencode/skill` | OpenCode (legacy path) |
| `~/.config/opencode/skills` | OpenCode (new path) |
| `/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/.claude/skills` | Hugo Blog |

## Usage

### List Available Skills

Show all skills in `~/.claude/skills` with their descriptions:

```bash
scripts/sync-skill.sh --list
```

### Sync Existing Skill

Copy a skill from `~/.claude/skills` to all target directories:

```bash
scripts/sync-skill.sh --name my-skill
```

This will:
- Create target directories if they don't exist
- Use rsync to sync the skill (deletes files in target that aren't in source)
- Show progress for each directory

### Create New Skill

Create a new skill with a template SKILL.md and sync it:

```bash
scripts/sync-skill.sh --new my-new-skill
```

This will:
- Create the skill directory structure
- Generate a template SKILL.md
- Create `scripts/`, `references/`, and `assets/` subdirectories
- Sync to all target directories

## How It Works

1. **Source of Truth**: `~/.claude/skills` is the primary location
2. **One-way Sync**: Skills are copied FROM source TO targets (not bidirectional)
3. **Rsync**: Uses `rsync -av --delete` for exact replication
4. **Idempotent**: Running sync multiple times is safe

## Workflow

When creating or updating a skill:

1. Edit the skill in `~/.claude/skills/<skill-name>/`
2. Run `scripts/sync-skill.sh --name <skill-name>`
3. Verify the skill appears in all target directories

## Troubleshooting

**Skill not found**: Ensure the skill exists in `~/.claude/skills/`. Use `--list` to verify.

**Permission denied**: Some target directories may require sudo. Check directory permissions.

**Sync incomplete**: Check the output for specific errors. Ensure target directories are accessible.
