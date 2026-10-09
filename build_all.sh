#!/usr/bin/env bash
# Build every box that is missing from Docker Hub, tagged exactly as the
# VisionIST fleet compose file expects (sipgisr/visionist-<name>:latest).
# Image names carry no version - the manifest's `version` is the recipe's,
# not the image's (see tools/_registry.py IMAGE_TAG).
#
# Usage:  ./build_all.sh                 # build the 8 missing fleet boxes
#         ./build_all.sh clip sbert      # only these boxes
#         SKIP_HUB=0 ./build_all.sh      # build all, even ones on the Hub
#
# Re-runnable: completed layers are cached, so reruns after a failure are fast.
#
# Per-box logs:  build-logs/<box>.log
# Summary:       build-logs/SUMMARY.txt

set -u
cd "$(dirname "$0")"

BOXES=${*:-"clip d4rt lang_sam lightglue sbert tapnext unimatch yolo sfm"}
SKIP_HUB=${SKIP_HUB:-1}   # 1 = skip boxes that are pullable from Docker Hub
LOGDIR=build-logs
mkdir -p "$LOGDIR"

hub_has() {  # repo -> 0 if it exists on Docker Hub
  curl -sf -o /dev/null "https://hub.docker.com/v2/repositories/sipgisr/$1/"
}

ok=(); fail=(); skip=()
t_start=$(date +%s)

for name in $BOXES; do
  dir="boxes/$name"
  [ -f "$dir/box.yaml" ] || { echo "!! $name: no boxes/$name/box.yaml — skipping"; fail+=("$name (missing)"); continue; }

  img_repo="visionist-${name//_/-}"
  tag="sipgisr/$img_repo:latest"

  if [ "$SKIP_HUB" = 1 ] && hub_has "sipgisr/$img_repo"; then
    echo "== $name: on Docker Hub as sipgisr/$img_repo — skipping"
    skip+=("$name (on hub)")
    continue
  fi

  echo "== building $name -> $tag"
  # tee: full build output goes to your terminal AND to the log file
  set -o pipefail
  if docker build --progress=plain \
       --tag "$tag" \
       --build-arg SERVICE_NAME="$name" \
       -f "$dir/docker/Dockerfile" "$dir" \
       2>&1 | tee "$LOGDIR/$name.log"; then
    ok+=("$name")
  else
    status=${PIPESTATUS[0]}
    echo "   FAILED (exit $status) — full log: $LOGDIR/$name.log"
    fail+=("$name")
  fi
  set +o pipefail
done

t_end=$(date +%s)
{
  echo "build_all summary — $(date -Is)"
  echo "elapsed: $((t_end - t_start))s"
  echo "OK:   ${ok[*]:-}"
  echo "SKIP: ${skip[*]:-}"
  echo "FAIL: ${fail[*]:-}"
  docker images --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}' | grep "sipgisr/visionist" || true
} | tee "$LOGDIR/SUMMARY.txt"

[ ${#fail[@]} -eq 0 ] || exit 1
