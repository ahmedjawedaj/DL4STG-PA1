#!/usr/bin/env bash
# usage: bash run_queue.sh JOBS_FILE PARALLEL THREADS
# each line of JOBS_FILE is the argument list for `python -m pa1q2.train`
set -u
jobs_file=$1; par=${2:-2}; thr=${3:-2}
mkdir -p logs
cat "$jobs_file" | grep -v '^\s*$' | xargs -P "$par" -I{} bash -c \
  'args="{}"; tag=$(echo "$args" | tr " -" "__"); T2_THREADS='"$thr"' python -m pa1q2.train $args > logs/$tag.log 2>&1; echo "done: $args"'
