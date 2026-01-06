#!/usr/bin/env bash
#
# verify.sh - Verify agent-ergonomics compliance (for target repo)
#
# This is a copy of the verify script that lives in target repos.
# It verifies the current repo conforms to agent-ergonomics standards.
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Verifying agent-ergonomics compliance in: $REPO_ROOT"
echo ""

ERRORS=0
WARNINGS=0

check_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

check_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((ERRORS++))
}

check_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

# Expected CLAUDE.md shim content (exact match required)
EXPECTED_SHIM='# Claude Code entrypoint
@AGENTS.md
@.agent/skills.md
@.agent/sandbox.md
@.agent/auth.md
@.agent/mcp.md
@.agent/codex.md
@.agent/inference_speed.md'

echo "=== Core Files ==="
echo ""

# Check CLAUDE.md exists and matches shim exactly
if [ -f "$REPO_ROOT/CLAUDE.md" ]; then
    ACTUAL_SHIM=$(cat "$REPO_ROOT/CLAUDE.md")
    if [ "$ACTUAL_SHIM" = "$EXPECTED_SHIM" ]; then
        check_pass "CLAUDE.md exists and matches shim template"
    else
        check_fail "CLAUDE.md exists but does not match shim template"
    fi
else
    check_fail "CLAUDE.md not found"
fi

# Check AGENTS.md exists
if [ -f "$REPO_ROOT/AGENTS.md" ]; then
    check_pass "AGENTS.md exists"
else
    check_fail "AGENTS.md not found"
fi

echo ""
echo "=== .agent/ Directory ==="
echo ""

# Check .agent/ directory and files
AGENT_FILES=(
    ".agent/skills.md"
    ".agent/sandbox.md"
    ".agent/auth.md"
    ".agent/mcp.md"
    ".agent/codex.md"
    ".agent/inference_speed.md"
)

for file in "${AGENT_FILES[@]}"; do
    if [ -f "$REPO_ROOT/$file" ]; then
        check_pass "$file exists"
    else
        check_fail "$file not found"
    fi
done

echo ""
echo "=== Summary ==="
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}Passed with $WARNINGS warning(s)${NC}"
    exit 0
else
    echo -e "${RED}Failed with $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "To fix, run: ./scripts/agent/bootstrap.sh"
    exit 1
fi
