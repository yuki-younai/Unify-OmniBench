#!/usr/bin/env bash
# AS-Bench 一键 push 到远程（github.com/yuki-younai/AS-Bench）
#
# 行为：
#   1. 确保 .gitignore 忽略 results/ 与 __pycache__/（首次运行会自动补齐 + 解除已 tracked）
#   2. 走 star-proxy 代理（GitHub HTTPS）
#   3. 自动 add / commit / push 到当前分支
#
# 用法：
#   bash start_push.sh                  # 用默认 commit message（当前时间）
#   bash start_push.sh "fix hunyuan"    # 自定义 commit message
set -euo pipefail

cd "$(dirname "$0")"

MSG="${1:-update: $(date '+%Y-%m-%d %H:%M:%S')}"
BRANCH="$(git branch --show-current)"

# ── 1. 代理（GitHub HTTPS 走 star-proxy）──
export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-$http_proxy}"

git remote set-url origin https://github.com/yuki-younai/Unify-OmniBench


# ── 2. 确保 .gitignore 含 results/ 与 __pycache__/ ──
touch .gitignore
need_clean=0
for pat in 'results/' '__pycache__/' '*.pyc' '.pytest_cache/' '.DS_Store'; do
  if ! grep -qxF "$pat" .gitignore; then
    echo "$pat" >> .gitignore
    need_clean=1
  fi
done

# 首次添加 ignore 规则时，把已 tracked 的 results/__pycache__ 从索引剔除（保留磁盘文件）
if [[ $need_clean -eq 1 ]]; then
  echo ">>> 首次配置 .gitignore，从索引移除 results/ 与 __pycache__/ ..."
  git rm -r --cached --ignore-unmatch results/ >/dev/null 2>&1 || true
  git ls-files | grep -E '(^|/)__pycache__/' | xargs -r git rm --cached >/dev/null 2>&1 || true
  git ls-files | grep -E '\.pyc$' | xargs -r git rm --cached >/dev/null 2>&1 || true
fi

# ── 3. add ──
git add -A

# ── 3.5. 跳过超过 10MB 的单个文件 ──
echo ">>> 检查大文件（>10MB）..."
large_files=$(git diff --cached --name-only | while IFS= read -r f; do
  if [ -f "$f" ]; then
    size=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$size" -gt 10485760 ]; then
      echo "$f"
    fi
  fi
done)

if [ -n "$large_files" ]; then
  echo "$large_files" | while IFS= read -r f; do
    echo "  SKIP (size > 10MB): $f"
    git reset HEAD -- "$f" >/dev/null 2>&1 || true
  done
fi

# ── 4. commit / push ──

if git diff --cached --quiet; then
  echo ">>> 无变更，直接 push（确保远程同步）"
else
  echo ">>> commit: $MSG"
  git commit -m "$MSG"
fi

echo ">>> push origin $BRANCH ..."
git push origin "$BRANCH"
echo ">>> 完成"
