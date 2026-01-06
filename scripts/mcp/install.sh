#!/usr/bin/env bash
#
# install.sh - Install MCP server definitions to user's ~/.mcp/
#
# Usage:
#   ./scripts/mcp/install.sh           # Install with symlinks (macOS/Linux)
#   ./scripts/mcp/install.sh --force   # Overwrite existing
#   ./scripts/mcp/install.sh --copy    # Copy instead of symlink
#   ./scripts/mcp/install.sh --dry-run # Show what would be done
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check for servers directory - support both kit and applied repo structures
if [ -d "$REPO_ROOT/templates/.agent/mcp/servers" ]; then
    SERVERS_DIR="$REPO_ROOT/templates/.agent/mcp/servers"
elif [ -d "$REPO_ROOT/.agent/mcp/servers" ]; then
    SERVERS_DIR="$REPO_ROOT/.agent/mcp/servers"
else
    echo -e "${RED}Error: No MCP servers directory found${NC}"
    echo "Expected at: $REPO_ROOT/templates/.agent/mcp/servers"
    echo "         or: $REPO_ROOT/.agent/mcp/servers"
    exit 1
fi

MCP_DIR="$HOME/.mcp"

# Defaults
FORCE=false
COPY_MODE=false
DRY_RUN=false

# Detect Windows (copy mode required)
if [[ "${OS:-}" == "Windows_NT" ]] || [[ "$(uname -s)" == MINGW* ]] || [[ "$(uname -s)" == CYGWIN* ]]; then
    COPY_MODE=true
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE=true
            shift
            ;;
        --copy)
            COPY_MODE=true
            shift
            ;;
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Install MCP server definitions to ~/.mcp/"
            echo ""
            echo "Options:"
            echo "  --force, -f     Overwrite existing files"
            echo "  --copy          Copy files instead of symlink"
            echo "  --dry-run, -n   Show what would be done"
            echo "  --help, -h      Show this help"
            echo ""
            echo "By default, uses symlinks on macOS/Linux and copy on Windows."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "MCP Server Installation"
echo "======================"
echo ""
echo "Source: $SERVERS_DIR"
echo "Target: $MCP_DIR"
echo "Mode:   $([ "$COPY_MODE" = true ] && echo "copy" || echo "symlink")"
echo ""

# Create ~/.mcp if needed
if [ ! -d "$MCP_DIR" ]; then
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would create: $MCP_DIR"
    else
        mkdir -p "$MCP_DIR"
        echo -e "${GREEN}Created:${NC} $MCP_DIR"
    fi
fi

# Track results
INSTALLED=0
SKIPPED=0
UPDATED=0

# Install each server definition
for src in "$SERVERS_DIR"/*.md; do
    if [ ! -f "$src" ]; then
        continue
    fi

    filename=$(basename "$src")
    dst="$MCP_DIR/$filename"

    # Check if target exists
    if [ -e "$dst" ] || [ -L "$dst" ]; then
        if [ "$FORCE" = true ]; then
            if [ "$DRY_RUN" = true ]; then
                echo "[DRY RUN] Would update: $filename"
            else
                rm -f "$dst"
                if [ "$COPY_MODE" = true ]; then
                    cp "$src" "$dst"
                else
                    ln -s "$src" "$dst"
                fi
                echo -e "${YELLOW}Updated:${NC} $filename"
                ((UPDATED++))
            fi
        else
            echo -e "${YELLOW}Skipped:${NC} $filename (exists, use --force to overwrite)"
            ((SKIPPED++))
        fi
    else
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would install: $filename"
        else
            if [ "$COPY_MODE" = true ]; then
                cp "$src" "$dst"
            else
                ln -s "$src" "$dst"
            fi
            echo -e "${GREEN}Installed:${NC} $filename"
            ((INSTALLED++))
        fi
    fi
done

echo ""
echo "Summary"
echo "-------"
echo "Installed: $INSTALLED"
echo "Updated:   $UPDATED"
echo "Skipped:   $SKIPPED"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "(This was a dry run. No changes were made.)"
fi

echo ""
echo "MCP servers are now available at: $MCP_DIR"
echo ""
echo "To use with Claude, configure your MCP client to read from ~/.mcp/"
