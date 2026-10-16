#!/usr/bin/env bash
# Build every box that is missing from Docker Hub, tagged exactly as the
# VisionIST fleet compose file expects (sipgisr/visionist-<name>:latest).
# Image names carry no version - the manifest's `version` is the recipe's,
# not the image's (see tools/_registry.py IMAGE_TAG).
#
# Usage:  ./build_all.sh                 # build the 8 missing fleet boxes
#         ./build_all.sh clip sbert      # only these boxes
#         SKIP_HUB=0 ./build_all.sh      # build all, even ones on the Hub
#         PAR=8 ./build_all.sh           # run 8 docker builds concurrently
#
# Re-runnable: completed layers are cached, so reruns after a failure are fast.
# Builds run concurrently (PAR, default 4) — docker's shared layer cache makes
# this safe; base images are pulled in parallel instead of serially.
#
# Per-box logs:  build-logs/<box>.log
# Summary:       build-logs/SUMMARY.txt

set -u
cd "$(dirname "$0")"

BOXES=${*:-$(ls boxes)}
SKIP_HUB=${SKIP_HUB:-1}   # 1 = skip boxes that are pullable from Docker Hub
PAR=${PAR:-4}             # concurrent builds (lower if RAM/disk I/O is tight)
LOGDIR=build-logs
mkdir -p "$LOGDIR"
STATDIR="$LOGDIR/.status"
mkdir -p "$STATDIR"; rm -f "$STATDIR"/*

hub_has() {  # repo -> 0 if it exists on Docker Hub
  curl -sf -o /dev/null --max-time 15 "https://hub.docker.com/v2/repositories/sipgisr/$1/"
}

record() {  # record KIND NAME — one file per box, safe across concurrent subshells
  printf '%s\n' "$2" > "$STATDIR/$1.$2"
}

do_build() {
  local name=$1
  local dir="boxes/$name"
  if [ ! -f "$dir/box.yaml" ]; then
    echo "!! $name: no boxes/$name/box.yaml — skipping" >&2
    record missing "$name"
    return
  fi

  local img_repo="visionist-${name//_/-}"
  local tag="sipgisr/$img_repo:latest"

  if [ "$SKIP_HUB" = 1 ] && hub_has "sipgisr/$img_repo"; then
    echo "== $name: on Docker Hub as sipgisr/$img_repo — skipping"
    record skip "$name"
    return
  fi

  echo "== [$name] building -> $tag"
  local log="$LOGDIR/$name.log"
  local code=0
  set -o pipefail
  docker build --progress=plain \
       --tag "$tag" \
       --build-arg SERVICE_NAME="$name" \
       -f "$dir/docker/Dockerfile" "$dir" \
       2>&1 | tee "$log" || code=${PIPESTATUS[0]}
  set +o pipefail

  if [ "$code" -eq 0 ]; then
    record ok "$name"
  else
    echo "   [$name] FAILED (exit $code) — full log: $log" >&2
    record fail "$name"
  fi
}

t_start=$(date +%s)
n_running=0
for name in $BOXES; do
  ( do_build "$name" ) &
  n_running=$((n_running + 1))
  if [ "$n_running" -ge "$PAR" ]; then
    wait -n || true   # free a slot once any build finishes
    n_running=$((n_running - 1))
  fi
done
wait

t_end=$(date +%s)
ok=""; fail=""; skip=""; missing=""
for f in "$STATDIR"/*; do
  [ -f "$f" ] || continue
  kind=$(basename "$f" | cut -d. -f1)
  name=$(basename "$f" | cut -d. -f2-)
  case $kind in
    ok)      ok="$ok $name" ;;
    fail)    fail="$fail $name" ;;
    skip)    skip="$skip $name" ;;
    missing) missing="$missing $name (missing)" ;;
  esac
done

{
  echo "build_all summary — $(date -Is)"
  echo "elapsed: $((t_end - t_start))s (PAR=$PAR)"
  echo "OK:   ${ok#}"
  echo "SKIP: ${skip#}"
  echo "FAIL: ${fail#}"
  [ -n "$missing" ] && echo "MISSING: ${missing#}"
  docker images --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}' | grep "sipgisr/visionist" || true
} | tee "$LOGDIR/SUMMARY.txt"

[ -z "${fail// /}" ] && [ -z "${missing// /}" ] || exit 1
