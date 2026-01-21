#!/bin/bash
# Skill Sync Script - Sync skills to multiple directories
# Usage: sync-skill.sh [--name <skill-name>] [--new <skill-name>] [--list] [--sync-only]

set -e

# Configuration - Target directories
TARGET_DIRS=(
    "$HOME/.claude/skills"
    "$HOME/.config/opencode/skill"
    "$HOME/.config/opencode/skills"
    "/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/.claude/skills"
)

# Source directory (primary skills location)
SOURCE_DIR="$HOME/.claude/skills"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# List available skills
list_skills() {
    print_info "Available skills in $SOURCE_DIR:"
    echo ""

    for skill_dir in "$SOURCE_DIR"/*/; do
        if [ -d "$skill_dir" ]; then
            skill_name=$(basename "$skill_dir")
            skill_file="$skill_dir/SKILL.md"

            if [ -f "$skill_file" ]; then
                # Extract description from SKILL.md
                description=$(grep "^description:" "$skill_file" | sed 's/description: //' | sed 's/^["'\'']\|["'\'']$//g')
                printf "  ${GREEN}%-30s${NC} %s\n" "$skill_name" "$description"
            else
                printf "  ${YELLOW}%-30s${NC} (No SKILL.md)\n" "$skill_name"
            fi
        fi
    done
    echo ""
}

# Check if skill exists
skill_exists() {
    local skill_name="$1"
    [ -d "$SOURCE_DIR/$skill_name" ]
}

# Create target directory if not exists
ensure_dir() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_info "Created directory: $dir"
    fi
}

# Sync skill to target directory
sync_skill_to_dir() {
    local skill_name="$1"
    local target_dir="$2"
    local source_path="$SOURCE_DIR/$skill_name"
    local target_path="$target_dir/$skill_name"

    ensure_dir "$target_dir"

    # Use rsync to sync the skill directory
    rsync -av --delete "$source_path/" "$target_path/" 2>/dev/null

    if [ $? -eq 0 ]; then
        print_success "Synced to: $target_dir"
        return 0
    else
        print_error "Failed to sync to: $target_dir"
        return 1
    fi
}

# Sync skill to all target directories
sync_skill() {
    local skill_name="$1"

    if ! skill_exists "$skill_name"; then
        print_error "Skill '$skill_name' not found in $SOURCE_DIR"
        echo ""
        print_info "Use --list to see available skills"
        exit 1
    fi

    print_info "Syncing skill: $skill_name"
    print_info "Source: $SOURCE_DIR/$skill_name"
    echo ""

    local success_count=0
    local total_count=${#TARGET_DIRS[@]}

    for target_dir in "${TARGET_DIRS[@]}"; do
        if sync_skill_to_dir "$skill_name" "$target_dir"; then
            ((success_count++))
        fi
    done

    echo ""
    print_success "Sync complete: $success_count/$total_count directories"
}

# Create new skill
create_skill() {
    local skill_name="$1"

    if skill_exists "$skill_name"; then
        print_warning "Skill '$skill_name' already exists"
        read -p "Do you want to sync it instead? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sync_skill "$skill_name"
        fi
        exit 0
    fi

    print_info "Creating new skill: $skill_name"

    local skill_path="$SOURCE_DIR/$skill_name"
    mkdir -p "$skill_path"

    # Create SKILL.md with template
    cat > "$skill_path/SKILL.md" << EOF
---
name: $skill_name
description: A new skill created on $(date +%Y-%m-%d)
---

# $skill_name

## Overview

This skill was created on $(date +%Y-%m-%d).

## Usage

Add your skill instructions here.
EOF

    # Create subdirectories
    mkdir -p "$skill_path/scripts"
    mkdir -p "$skill_path/references"
    mkdir -p "$skill_path/assets"

    print_success "Skill structure created at: $skill_path"
    print_info "Please edit SKILL.md to add your skill content"

    # Now sync to all directories
    echo ""
    sync_skill "$skill_name"
}

# Show usage
show_usage() {
    cat << EOF
Skill Sync Tool - Sync skills to multiple directories

Usage:
    sync-skill.sh [options]

Options:
    --name <skill-name>      Sync existing skill to all target directories
    --new <skill-name>       Create new skill and sync to all directories
    --list                   List all available skills
    --help                   Show this help message

Target directories:
    - ~/.claude/skills
    - ~/.config/opencode/skill
    - ~/.config/opencode/skills
    - ~/Gitlab/Personal/Hugo_Blog/blog/.claude/skills

Examples:
    sync-skill.sh --list
    sync-skill.sh --name my-skill
    sync-skill.sh --new my-new-skill
EOF
}

# Main
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi

    local action=""
    local skill_name=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --name)
                action="sync"
                skill_name="$2"
                shift 2
                ;;
            --new)
                action="create"
                skill_name="$2"
                shift 2
                ;;
            --list)
                list_skills
                exit 0
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    case "$action" in
        sync)
            sync_skill "$skill_name"
            ;;
        create)
            create_skill "$skill_name"
            ;;
        *)
        print_error "Please specify --name or --new"
        show_usage
        exit 1
        ;;
    esac
}

main "$@"
