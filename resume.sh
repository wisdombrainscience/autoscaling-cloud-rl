#!/bin/sh
# re-run any job from jobs.txt whose result JSON is missing
cd /home/claude/work/code
while read v s; do
  [ -f ../results/runs/${v}_s${s}.json ] || echo "$v $s"
done < jobs.txt > jobs_left.txt
cat jobs_left.txt | xargs -P 2 -L 1 sh -c 'python3 run_control.py $0 $1 > ../results/logs/$0_$1.log 2>&1; echo "finished $0 $1"'
