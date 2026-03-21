#!/bin/sh
set -eu

LOCK_HASH_FILE="/app/node_modules/.package-lock.hash"
CURRENT_HASH="$(cat /app/package.json /app/package-lock.json | sha256sum | awk '{print $1}')"

if [ ! -d /app/node_modules ] || [ ! -f "$LOCK_HASH_FILE" ] || [ "$(cat "$LOCK_HASH_FILE")" != "$CURRENT_HASH" ]; then
  echo "Installing dependencies inside Docker volume..."
  cd /app
  npm ci --include=optional
  printf '%s' "$CURRENT_HASH" > "$LOCK_HASH_FILE"
fi

exec "$@"
