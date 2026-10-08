#!/usr/bin/env bash
# Safety net for unattended pods: stop (not delete) this pod if nobody has copied the results within 40 min of DONE.
# /workspace is a volume disk, so the results survive the stop. Touch /workspace/FETCHED after copying them back.
O=${1:-/workspace/out}
# ssh sessions don't carry the container's environment: take RunPod's pod id and pod-scoped API key from PID 1
while IFS= read -r -d '' kv; do case "$kv" in RUNPOD_POD_ID=*|RUNPOD_API_KEY=*) export "$kv";; esac; done < /proc/1/environ
runpodctl config --apiKey "$RUNPOD_API_KEY" > /dev/null 2>&1
[ "${2:-}" = "--check" ] && { runpodctl get pod "$RUNPOD_POD_ID" 2>&1 | head -3; exit 0; }
for i in $(seq 1 80); do [ -e /workspace/FETCHED ] && exit 0; sleep 30; done
echo "[$(date -u +%H:%M:%S)] self-stop (results not fetched within 40 min)" >> $O/STATUS
runpodctl stop pod "$RUNPOD_POD_ID" >> $O/STATUS 2>&1 || runpodctl pod stop "$RUNPOD_POD_ID" >> $O/STATUS 2>&1
