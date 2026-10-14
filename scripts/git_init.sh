#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d .git ]; then
  echo 'Repository is already initialized.'
  exit 0
fi
git init -b main
git add .
echo 'Files staged. Review with git diff --cached, then commit and add your GitHub remote.'
