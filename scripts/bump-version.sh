#!/bin/bash
set -e

# Determine bump type from input (patch, minor, major)
BUMP_TYPE="${1:-patch}"
FILE_PKG="package.json"
FILE_PLUGIN="openclaw.plugin.json"

# Read current version from package.json
CURRENT_VERSION=$(node -p "require('./$FILE_PKG').version")
echo "Current version: $CURRENT_VERSION"

# Bump version using node
NEW_VERSION=$(node -e "
  const v = '$CURRENT_VERSION'.split('.').map(Number);
  const parts = { major: 0, minor: 1, patch: 2 };
  const i = parts['$BUMP_TYPE'];
  v[i]++;
  for (let j = i+1; j < 3; j++) v[j] = 0;
  console.log(v.join('.'));
")
echo "New version: $NEW_VERSION"

# Update package.json
node -e "
  const fs = require('fs');
  const pkg = JSON.parse(fs.readFileSync('$FILE_PKG', 'utf8'));
  pkg.version = '$NEW_VERSION';
  fs.writeFileSync('$FILE_PKG', JSON.stringify(pkg, null, 2) + '\n');
"

# Update openclaw.plugin.json
node -e "
  const fs = require('fs');
  const pkg = JSON.parse(fs.readFileSync('$FILE_PLUGIN', 'utf8'));
  pkg.version = '$NEW_VERSION';
  fs.writeFileSync('$FILE_PLUGIN', JSON.stringify(pkg, null, 2) + '\n');
"

echo "✅ Version bumped to $NEW_VERSION in both files"