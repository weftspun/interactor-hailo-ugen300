#!/bin/sh
# Wait for the UGen300 to enumerate, then run the frame ladder and report the port current.
CLI="/c/Program Files/HailoRT/bin/hailortcli.exe"
PY="/c/weftspun-keypoints/6-datasource/anny-render-corpus/.pixi/envs/restyle/python.exe"
for i in $(seq 1 120); do
  if "$CLI" scan 2>/dev/null | grep -q "Device:"; then
    echo "device appeared after ${i} polls (~$((i*5))s)"
    "$CLI" scan 2>&1 | head -3
    sleep 2
    rm -f hailort.log
    "$PY" bench_frames.py 2>&1 | tail -12
    echo
    echo "=== advertised port current ==="
    grep -o "electrical current advertised: [0-9.]*A" hailort.log 2>/dev/null | sort -u \
      || echo "NO CURRENT WARNING LOGGED -- the port did not trip the 1.5A notice"
    exit 0
  fi
  sleep 5
done
echo "timed out after 10 minutes; device never enumerated"
exit 1
