#!/usr/bin/env bash
#
# list.sh - List available documentation
#
# Usage:
#   ./scripts/docs/list.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Documentation Index ==="
echo ""

# Check docs directory
if [ -d "$REPO_ROOT/docs" ]; then
    echo "📁 docs/"
    for doc in "$REPO_ROOT/docs"/*.md; do
        if [ -f "$doc" ]; then
            filename=$(basename "$doc")
            # Try to get first heading
            title=$(grep -m1 "^# " "$doc" 2>/dev/null | sed 's/^# //' || echo "$filename")
            echo "   📄 $filename - $title"
        fi
    done
    echo ""
fi

# Check AGENTS.md
if [ -f "$REPO_ROOT/AGENTS.md" ]; then
    echo "📄 AGENTS.md - Agent Guidelines (canonical)"
fi

# Check CLAUDE.md
if [ -f "$REPO_ROOT/CLAUDE.md" ]; then
    echo "📄 CLAUDE.md - Claude Code entrypoint (shim)"
fi

# Check .agent directory
if [ -d "$REPO_ROOT/.agent" ]; then
    echo ""
    echo "📁 .agent/"
    for doc in "$REPO_ROOT/.agent"/*.md; do
        if [ -f "$doc" ]; then
            filename=$(basename "$doc")
            echo "   📄 $filename"
        fi
    done

    # Check MCP servers
    if [ -d "$REPO_ROOT/.agent/mcp/servers" ]; then
        echo ""
        echo "📁 .agent/mcp/servers/"
        for doc in "$REPO_ROOT/.agent/mcp/servers"/*.md; do
            if [ -f "$doc" ]; then
                filename=$(basename "$doc")
                echo "   📄 $filename"
            fi
        done
    fi
fi

echo ""
echo "=== Quick Access ==="
echo ""
echo "View a document:  cat docs/<filename>"
echo "View agent docs:  cat .agent/<filename>"
echo "View this list:   ./scripts/docs/list.sh"
