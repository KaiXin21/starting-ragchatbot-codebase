# Frontend Code Quality Changes

## What Was Added

### Prettier (auto-formatter — front-end equivalent of black)

- **`frontend/.prettierrc`** — Formatting rules:
  - 100-character line width
  - 4-space indentation
  - Single quotes
  - Trailing commas (ES5 style)
  - LF line endings

- **`frontend/package.json`** — npm scripts:
  - `npm run format` — auto-format all frontend files in-place
  - `npm run format:check` — check formatting without changing files (CI-friendly)
  - `npm run lint` — run ESLint on `script.js`
  - `npm run lint:fix` — auto-fix ESLint issues where possible
  - `npm run check` — run both format check and lint (full quality gate)

### ESLint (JavaScript linter)

- **`frontend/.eslintrc.json`** — Rules enforced:
  - `no-eval` / `no-implied-eval` — blocks XSS-prone code patterns
  - `eqeqeq` — requires `===` over `==`
  - `no-var` — enforces `let`/`const`
  - `prefer-const` — flags variables never reassigned
  - `curly` — requires braces on all control flow blocks
  - `no-console` (warn) — flags leftover debug logs, allows `warn`/`error`

### Quality Check Script

- **`check-frontend.sh`** (project root) — single command to run all frontend checks:
  ```bash
  ./check-frontend.sh
  ```
  Installs dependencies, runs Prettier check, then ESLint. Exits non-zero on any failure.

## Formatting Applied to Existing Files

### `frontend/script.js`

Reformatted to match Prettier rules:
- Removed stale inline comments and dead-code comments
- Removed `console.log` debug calls (`loadCourseStats` had two)
- Added curly braces to single-line `if` bodies
- Normalized trailing commas in object/array literals
- Consistent blank lines between logical sections (one, not two)
- Wrapped long lines (`addMessage` welcome text, `.catch` callback) to stay within 100 chars

`index.html` and `style.css` were already consistently formatted and required no changes.

## How to Use

```bash
# Install deps once
cd frontend && npm install

# Check formatting
npm run format:check

# Auto-fix formatting
npm run format

# Lint JS
npm run lint

# Run everything (use in CI)
npm run check

# Or from project root
./check-frontend.sh
```
