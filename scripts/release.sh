#!/usr/bin/env bash
#
# Release script for transcriber
# Usage: ./scripts/release.sh [--patch|--minor|--major] [--yes]
#
# Automatically determines version bump from conventional commits:
#   fix:              → patch (0.0.X)
#   feat:             → minor (0.X.0)
#   BREAKING CHANGE:  → major (X.0.0)
#   feat!: / fix!:    → major (X.0.0)
#
# Users install via:
#   uv tool install git+https://github.com/Deloitte-Nordics/transcriber.git
#   uv tool install "transcriber[local] @ git+https://github.com/Deloitte-Nordics/transcriber.git"
#   uv add "transcriber @ git+https://github.com/Deloitte-Nordics/transcriber.git@v1.0.0"
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Parse command line arguments
FORCE_BUMP=""
AUTO_YES=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --patch) FORCE_BUMP="patch"; shift ;;
        --minor) FORCE_BUMP="minor"; shift ;;
        --major) FORCE_BUMP="major"; shift ;;
        --yes|-y) AUTO_YES=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--patch|--minor|--major] [--yes|-y]"
            echo ""
            echo "Automatically determines version bump from conventional commits,"
            echo "or use flags to force a specific bump type."
            echo ""
            echo "Options:"
            echo "  --yes, -y    Skip all confirmation prompts (for CI/automation)"
            exit 0
            ;;
        *) error "Unknown option: $1" ;;
    esac
done

# Ensure we're in the project root
if [ ! -f "pyproject.toml" ]; then
    error "Must run from project root (pyproject.toml not found)"
fi

# Check for required tools
command -v uv >/dev/null 2>&1 || error "uv is not installed"
command -v git >/dev/null 2>&1 || error "git is not installed"
command -v git-cliff >/dev/null 2>&1 || error "git-cliff is not installed (see https://github.com/orhun/git-cliff)"

# Check git working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    error "Git working directory is not clean. Commit or stash changes first."
fi

# Check we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    warn "Not on main branch (currently on: $CURRENT_BRANCH)"
    if $AUTO_YES; then
        info "--yes passed, continuing on non-main branch"
    else
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
info "Current version: $CURRENT_VERSION"

# Parse current version
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

# Get the last tag (if any)
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")

if [ -n "$LAST_TAG" ]; then
    info "Last tag: $LAST_TAG"
    COMMIT_RANGE="${LAST_TAG}..HEAD"
else
    warn "No previous tags found, analyzing all commits"
    COMMIT_RANGE="HEAD"
fi

# Analyze commits for conventional commit types
if [ -n "$FORCE_BUMP" ]; then
    BUMP_TYPE="$FORCE_BUMP"
    info "Forced bump type: $BUMP_TYPE"
else
    info "Analyzing commits since $LAST_TAG..."
    
    # Get commit messages
    if [ -n "$LAST_TAG" ]; then
        COMMITS=$(git log "$COMMIT_RANGE" --pretty=format:"%s" 2>/dev/null || echo "")
    else
        COMMITS=$(git log --pretty=format:"%s" 2>/dev/null || echo "")
    fi
    
    if [ -z "$COMMITS" ]; then
        error "No commits found since last release"
    fi
    
    # Determine bump type from commits
    BUMP_TYPE="patch"  # default
    HAS_BREAKING=false
    HAS_FEAT=false
    HAS_FIX=false
    
    while IFS= read -r commit; do
        # Check for breaking changes
        if [[ "$commit" =~ ^[a-z]+!: ]] || [[ "$commit" =~ BREAKING[[:space:]]CHANGE ]]; then
            HAS_BREAKING=true
        fi
        # Check for features
        if [[ "$commit" =~ ^feat(\(.+\))?!?: ]]; then
            HAS_FEAT=true
        fi
        # Check for fixes
        if [[ "$commit" =~ ^fix(\(.+\))?!?: ]]; then
            HAS_FIX=true
        fi
    done <<< "$COMMITS"
    
    if $HAS_BREAKING; then
        BUMP_TYPE="major"
    elif $HAS_FEAT; then
        BUMP_TYPE="minor"
    elif $HAS_FIX; then
        BUMP_TYPE="patch"
    fi
    
    # Show commits that influenced the decision
    echo ""
    echo -e "${BLUE}Commits since last release:${NC}"
    if [ -n "$LAST_TAG" ]; then
        git log "$COMMIT_RANGE" --pretty=format:"  %s" | head -20
    else
        git log --pretty=format:"  %s" | head -20
    fi
    echo ""
    echo ""
    
    if $HAS_BREAKING; then
        info "Detected: BREAKING CHANGE → major bump"
    elif $HAS_FEAT; then
        info "Detected: feat: commits → minor bump"
    elif $HAS_FIX; then
        info "Detected: fix: commits → patch bump"
    else
        warn "No conventional commits found, defaulting to patch bump"
    fi
fi

# Calculate new version
case $BUMP_TYPE in
    major)
        NEW_VERSION="$((MAJOR + 1)).0.0"
        ;;
    minor)
        NEW_VERSION="${MAJOR}.$((MINOR + 1)).0"
        ;;
    patch)
        NEW_VERSION="${MAJOR}.${MINOR}.$((PATCH + 1))"
        ;;
esac

echo ""
info "Version bump: $CURRENT_VERSION → $NEW_VERSION ($BUMP_TYPE)"
echo ""
if ! $AUTO_YES; then
    read -p "Proceed with release? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Aborted."
        exit 0
    fi
fi

# Run linting
info "Running linting..."
uv run ruff check . || error "Linting failed"
uv run ruff format --check . || error "Format check failed"
uv run pyright || error "Type checking failed"

# Run tests
info "Running tests..."
uv run pytest tests/ -v || error "Tests failed"

# Update version in pyproject.toml
info "Updating version in pyproject.toml..."
sed -i '' "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml

# Verify the update
UPDATED_VERSION=$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
if [ "$UPDATED_VERSION" != "$NEW_VERSION" ]; then
    error "Failed to update version in pyproject.toml"
fi

# Generate changelog
info "Generating CHANGELOG.md..."
git-cliff --tag "v$NEW_VERSION" -o CHANGELOG.md

# Commit the version bump and changelog
info "Committing version bump and changelog..."
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release v$NEW_VERSION"

# Create git tag
info "Creating git tag v$NEW_VERSION..."
git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"

# Confirm before pushing
echo ""
warn "Ready to push v$NEW_VERSION to remote."
if ! $AUTO_YES; then
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        warn "Aborting. Rolling back changes..."
        git tag -d "v$NEW_VERSION"
        git reset --soft HEAD~1
        git checkout pyproject.toml
        exit 1
    fi
fi

# Push to remote
info "Pushing to remote..."
git push origin "$CURRENT_BRANCH"
git push origin "v$NEW_VERSION"

echo ""
info "Successfully released v$NEW_VERSION!"
echo ""
echo "Install with:"
echo "  uv tool install git+https://github.com/Deloitte-Nordics/transcriber.git                          # CLI (cloud only)"
echo "  uv tool install \"transcriber[local] @ git+https://github.com/Deloitte-Nordics/transcriber.git\"   # CLI (with local Whisper)"
echo "  uv add \"transcriber @ git+https://github.com/Deloitte-Nordics/transcriber.git@v$NEW_VERSION\"     # library (pinned)"
