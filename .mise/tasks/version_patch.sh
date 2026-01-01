#!/bin/bash
#MISE description = "Bump patch version (X.Y.Z+1 -> X.Y.Z+1)"

set -e

echo "🔍 Bumping patch version..."

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo "Current version: $CURRENT_VERSION"

# Parse version components
MAJOR=$(echo $CURRENT_VERSION | cut -d. -f1)
MINOR=$(echo $CURRENT_VERSION | cut -d. -f2)
PATCH=$(echo $CURRENT_VERSION | cut -d. -f3)

# Increment patch version
NEW_PATCH=$((PATCH + 1))
NEW_VERSION="$MAJOR.$MINOR.$NEW_PATCH"

echo "New version: $NEW_VERSION"

# Update pyproject.toml
sed -i "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml

# Update mise.toml if it exists and has a version
if [ -f ".mise/tasks/version.sh" ]; then
    sed -i "s/VERSION=.*/VERSION=\"$NEW_VERSION\"/" .mise/tasks/version.sh
fi

# Commit version bump
git add pyproject.toml
if [ -f ".mise/tasks/version.sh" ]; then
    git add .mise/tasks/version.sh
fi

git commit -m "version: bump to $NEW_VERSION"

echo "✅ Patch version bumped to $NEW_VERSION"