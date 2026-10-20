#!/bin/bash
# Format source files. Skips any formatter not on PATH (and says which).
# Tree walks list files with git, so ignored files are never touched.
# Named files are formatted directly, even when git ignores them.
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

# List git's unignored files under the roots matching extension regex $1.
list_files() {
  local f
  git ls-files -z -co --exclude-standard -- "${roots[@]}" |
    while IFS= read -r -d '' f; do
      # Regular files only: skip deleted entries and submodule dirs.
      [[ -f $f && $f =~ \.($1)$ ]] || continue
      printf '%s\0' "$f"
    done
}

# Format the listed files with extension regex $2 in one batch.
fmt_batch() {
  local label="$1" exts="$2"
  shift 2
  echo "[format] $label"
  list_files "$exts" | xargs -0 -r "$@"
}

# Format all matching files under one or more directory roots.
fmt_tree() {
  local roots=("$@")
  run_fmt rustfmt fmt_batch rust rs rustfmt --config-path "$FORMAT_DIR" --config skip_children=true
  run_fmt ruff fmt_batch python py ruff format -q --config "$FORMAT_DIR/pyproject.toml"
  run_fmt shfmt fmt_batch bash sh shfmt -w -i 2 -ci
  run_fmt prettier fmt_batch markdown/json/yaml 'md|json|yaml|yml' prettier --write --log-level warn --config "$FORMAT_DIR/.prettierrc.json"
  run_fmt clang-format fmt_batch c++ 'cpp|cc|h|hpp' clang-format --style="file:$FORMAT_DIR/.clang-format" -i
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
