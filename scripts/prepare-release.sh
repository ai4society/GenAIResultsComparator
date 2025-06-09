#!/bin/bash
set -e # Exit on error

# 1. Get current version
CURRENT_VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
echo "Current version in pyproject.toml: $CURRENT_VERSION"

# 2. Prompt for new version
read -p "Enter new version (e.g., 0.1.5): " NEW_VERSION

if [ -z "$NEW_VERSION" ]; then
    echo "No version entered. Aborting."
    exit 1
fi

# Validate version format (basic example)
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([a-zA-Z0-9.-]*)$ ]]; then
    echo "Invalid version format. Expected X.Y.Z or X.Y.Z-suffix. Aborting."
    exit 1
fi

TAG_NAME="v$NEW_VERSION"

echo ""
echo "You are about to:"
echo "1. Update pyproject.toml to version $NEW_VERSION (MANUAL STEP REQUIRED)"
echo "2. Commit pyproject.toml with message: 'Bump version to $NEW_VERSION'"
echo "3. Create tag $TAG_NAME"
echo "4. Push main/master branch and tag $TAG_NAME (this will trigger the PyPI publish workflow)"
echo ""
read -p "Have you ALREADY manually updated pyproject.toml to $NEW_VERSION? (y/n) " CONFIRM_PYPROJECT_UPDATE

if [ "$CONFIRM_PYPROJECT_UPDATE" != "y" ]; then
    echo "Please update pyproject.toml to version $NEW_VERSION first, then re-run."
    exit 1
fi

# Verify the version in pyproject.toml matches NEW_VERSION
ACTUAL_PYPROJECT_VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
if [ "$ACTUAL_PYPROJECT_VERSION" != "$NEW_VERSION" ]; then
    echo "Error: pyproject.toml version ($ACTUAL_PYPROJECT_VERSION) does not match entered new version ($NEW_VERSION)."
    echo "Please correct pyproject.toml and re-run."
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "You have uncommitted changes. Please commit or stash them before releasing."
    read -p "Continue anyway? (y/n) " CONTINUE_UNCOMMITTED
    if [ "$CONTINUE_UNCOMMITTED" != "y" ]; then
        exit 1
    fi
fi


echo "Proceeding with release..."

# 3. Commit pyproject.toml
git add pyproject.toml
git commit -m "Bump version to $NEW_VERSION"

# 4. Create tag
echo "Creating tag $TAG_NAME..."
git tag "$TAG_NAME"

# 5. Push changes and tag
echo "Pushing commit and tag..."
# Determine current branch (e.g. main or master)
CURRENT_GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push origin "$CURRENT_GIT_BRANCH"
git push origin "$TAG_NAME"

echo ""
echo "Version $NEW_VERSION commit and tag $TAG_NAME pushed."
echo "The GitHub Actions workflow should now start publishing to PyPI."
echo "Monitor it here: https://github.com/ai4society/GenAIResultsComparator/actions"
