#!/usr/bin/env bash
# Run all frontend code quality checks.
set -e

FRONTEND_DIR="$(cd "$(dirname "$0")/frontend" && pwd)"

echo "==> Installing frontend dev dependencies..."
cd "$FRONTEND_DIR"
npm install --silent

echo ""
echo "==> Prettier format check..."
npx prettier --check .

echo ""
echo "==> ESLint..."
npx eslint script.js

echo ""
echo "All frontend quality checks passed."
