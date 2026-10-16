#!/bin/bash
# Format source files. Skips any formatter not on PATH (and says which).
#   whole tree:   ./format/format.sh
#   paths/globs:  ./format/format.sh src/ lib/util.py *.rs      (shell expands globs; a single file works too)
set -euo pipefail

FORMAT_DIR="$(cd "$(dirname "$0")" && pwd)"
has() { command -v "$1" >/dev/null 2>&1; }

# Run one formatter, or say it is not on PATH.
run_fmt() {
  local tool="$1"
  shift
  if has "$tool"; then
    "$@"
  else
    echo "[format] skipping $tool: not on PATH" >&2
  fi
}

# Format one file, dispatched by extension.
fmt_file() {
  local f="$1"
  [ -f "$f" ] || return 0
  case "${f##*.}" in
    rs) run_fmt rustfmt rustfmt --config-path "$FORMAT_DIR" --config skip_children=true "$f" ;;
    py) run_fmt ruff ruff format -q --config "$FORMAT_DIR/pyproject.toml" "$f" ;;
    sh) run_fmt shfmt shfmt -w -i 2 -ci "$f" ;;
    cc | cpp | h | hpp) run_fmt clang-format clang-format --style="file:$FORMAT_DIR/.clang-format" -i "$f" ;;
    md | json | yaml | yml | js | ts) run_fmt prettier prettier --write --log-level warn --config "$FORMAT_DIR/.prettierrc.json" "$f" ;;
  esac
  return 0
}

# Directory names every tree walk skips: VCS, build output, deps, Python venv, scratch.
PRUNE=(! -path "*/.git/*" ! -path "*/target/*" ! -path "*/node_modules/*" ! -path "*/.venv/*" ! -path "./work/*")
# Project's own skips.
PRUNE+=()

# Format all matching files under one or more directory roots.
fmt_tree() {
  local roots=("$@")
  if has rustfmt; then
    echo "[format] rust"
    find "${roots[@]}" -name "*.rs" "${PRUNE[@]}" -print0 |
      xargs -0 -r rustfmt --config-path "$FORMAT_DIR" --config skip_children=true
  else
    echo "[format] skipping rustfmt: not on PATH" >&2
  fi
  if has ruff; then
    echo "[format] python"
    find "${roots[@]}" -name "*.py" "${PRUNE[@]}" -print0 |
      xargs -0 -r ruff format -q --config "$FORMAT_DIR/pyproject.toml"
  else
    echo "[format] skipping ruff: not on PATH" >&2
  fi
  if has shfmt; then
    echo "[format] bash"
    find "${roots[@]}" -name "*.sh" "${PRUNE[@]}" -print0 | xargs -0 -r shfmt -w -i 2 -ci
  else
    echo "[format] skipping shfmt: not on PATH" >&2
  fi
  if has prettier; then
    echo "[format] markdown/json/yaml"
    find "${roots[@]}" \( -name "*.md" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" \) "${PRUNE[@]}" -print0 |
      xargs -0 -r prettier --write --log-level warn --config "$FORMAT_DIR/.prettierrc.json"
  else
    echo "[format] skipping prettier: not on PATH" >&2
  fi
  if has clang-format; then
    mapfile -t cpp_files < <(find "${roots[@]}" \( -name "*.cpp" -o -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) "${PRUNE[@]}")
    if [ "${#cpp_files[@]}" -gt 0 ]; then
      echo "[format] c++"
      clang-format --style="file:$FORMAT_DIR/.clang-format" -i "${cpp_files[@]}"
    fi
  else
    echo "[format] skipping clang-format: not on PATH" >&2
  fi
}

# Explicit paths/globs: format just those files and directories, no clean guard.
if [ "$#" -gt 0 ]; then
  roots=()
  for p in "$@"; do
    if [ -d "$p" ]; then
      roots+=("$p")
    elif [ -f "$p" ]; then
      fmt_file "$p"
    else
      echo "${0##*/}: no such file or directory: $p" >&2
    fi
  done
  [ "${#roots[@]}" -gt 0 ] && fmt_tree "${roots[@]}"
  exit 0
fi

# Whole tree: clean-tree guard, then format everything.
REPO_ROOT="$(cd "$FORMAT_DIR/.." && pwd)"
cd "$REPO_ROOT"
if [ -z "${FORMAT_SKIP_CLEAN_CHECK:-}" ]; then
  unstaged="$(git status --porcelain | grep '^.[^ ]' || true)"
  if [ -n "$unstaged" ]; then
    echo "${0##*/}: unstaged changes present (stage or stash first, or set FORMAT_SKIP_CLEAN_CHECK=1):" >&2
    echo "$unstaged" >&2
    exit 1
  fi
fi
fmt_tree .
