#!/bin/bash
cd "$(dirname "$0")/.."
[ -f logs/workers.pid ] && xargs -r kill < logs/workers.pid 2>/dev/null; rm -f logs/workers.pid
