#!/bin/bash
set -e

BUMP_TYPE="${1:-patch}"
FILE_PKG="package.json"
FILE_PLUGIN="openclaw.plugin.json"

CURRENT_VERSION=$(node -p "require('./$FILE_PKG').version")
echo "Current version: $CURRENT_VERSION"

NEW_VERSION=$(node -e "
  const v = '$CURRENT_VERSION'.split('.').map(Number);
  const parts = { major: 0, minor: 1, patch: 2 };
  const i = parts['$BUMP_TYPE'];
  v[i]++;
  for (let j = i+1; j < 3; j++) v[j] = 0;
  console.log(v.join('.'));
")

node -e "
  const fs = require('fs');
  const pkg = JSON.parse(fs.readFileSync('$FILE_PKG', 'utf8'));
  pkg.version = '$NEW_VERSION';
  fs.writeFileSync('$FILE_PKG', JSON.stringify(pkg, null, 2) + '\n');
"

node -e "
  const fs = require('fs');
  const pkg = JSON.parse(fs.readFileSync('$FILE_PLUGIN', 'utf8'));
  pkg.version = '$NEW_VERSION';
  fs.writeFileSync('$FILE_PLUGIN', JSON.stringify(pkg, null, 2) + '\n');
"

echo "New version: $NEW_VERSION"
echo "$NEW_VERSION"