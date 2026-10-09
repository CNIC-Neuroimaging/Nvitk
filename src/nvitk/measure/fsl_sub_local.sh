#!/bin/bash

# Copyright (C) 2007-2017 University of Oxford
# Authors: Dave Flitney, Stephen Smith, Matthew Webster and Duncan Mortimer

#   Part of FSL - FMRIB's Software Library
#   http://www.fmrib.ox.ac.uk/fsl
#   fsl@fmrib.ox.ac.uk
#
#   Developed at FMRIB (Oxford Centre for Functional Magnetic Resonance
#   Imaging of the Brain), Department of Clinical Neurology, Oxford
#   University, Oxford, UK
#
#
#   LICENCE
#
#   FMRIB Software Library, Release 6.0 (c) 2018, The University of
#   Oxford (the "Software")
#
#   The Software remains the property of the Oxford University Innovation
#   ("the University").
#
#   The Software is distributed "AS IS" under this Licence solely for
#   non-commercial use in the hope that it will be useful, but in order
#   that the University as a charitable foundation protects its assets for
#   the benefit of its educational and research purposes, the University
#   makes clear that no condition is made or to be implied, nor is any
#   warranty given or to be implied, as to the accuracy of the Software,
#   or that it will be suitable for any particular purpose or for use
#   under any specific conditions. Furthermore, the University disclaims
#   all responsibility for the use which is made of the Software. It
#   further disclaims any liability for the outcomes arising from using
#   the Software.
#
#   The Licensee agrees to indemnify the University and hold the
#   University harmless from and against any and all claims, damages and
#   liabilities asserted by third parties (including claims for
#   negligence) which arise directly or indirectly from the use of the
#   Software or the sale of any products based on the Software.
#
#   No part of the Software may be reproduced, modified, transmitted or
#   transferred in any form or by any means, electronic or mechanical,
#   without the express permission of the University. The permission of
#   the University is not required if the said reproduction, modification,
#   transmission or transference is done without financial return, the
#   conditions of this Licence are imposed upon the receiver of the
#   product, and all original and amended source code is included in any
#   transmitted product. You may be held legally responsible for any
#   copyright infringement that is caused or encouraged by your failure to
#   abide by these terms and conditions.
#
#   You are not permitted under this Licence to use this Software
#   commercially. Use for which any financial return is received shall be
#   defined as commercial use, and includes (1) integration of all or part
#   of the source code or the Software into a product for sale or license
#   by or on behalf of Licensee to third parties or (2) use of the
#   Software or any derivative of it for research with the final aim of
#   developing software products for sale or license to a third party or
#   (3) use of the Software or any derivative of it for research with the
#   final aim of developing non-software products for sale or license to a
#   third party, or (4) use of the Software to provide any service to an
#   external organisation for which payment is received. If you are
#   interested in using the Software commercially, please contact Oxford
#   University Innovation ("OUI"), the technology transfer company of the
#   University, to negotiate a licence. Contact details are:
#   fsl@innovation.ox.ac.uk quoting Reference Project 9564, FSL.

###########################################################################
# nvitk's local fsl_sub.
#
# Derived from the bash fsl_sub V1.1 that shipped with FSL 6.0 (its NONE /
# FSLPARALLEL branch), modified to run on this machine only. nvitk puts it in
# front of FSL's own fsl_sub -- in a throwaway FSLDIR overlay, never in the FSL
# installation -- when it runs randomise_parallel locally, on a workstation or
# inside a cluster job:
#
#   * FSL's Python fsl_sub follows site configuration (SGE/SLURM plugins,
#     ~/.fsl_sub.yml). On a configured machine a "local" randomise_parallel
#     would queue its fragments and return before a single map existed.
#   * Its shell plugin, like FSLPARALLEL in the old script, runs one task per
#     core by default. Every randomise fragment holds the whole 4D stack in
#     memory, so on a 32-core workstation that is 32 copies of it at once.
#
# How many tasks of a task file (-t) run at once -- the first one set wins:
#
#   FSLSUB_PARALLEL   the variable FSL's own fsl_sub reads (0 = every core)
#   FSLPARALLEL       the old fsl_sub's name for it (0 = every core)
#   NSLOTS            the slots SGE granted the enclosing job
#   1                 otherwise, with a note saying how to raise it
#
# Each task runs single-threaded (OMP/BLAS pinned to 1): the cap is in cores.
#
# Unlike the old script a failed task is not ignored. fsl_sub prints its job
# id as always, exits non-zero, and records the failure; a later
# "fsl_sub -j <that id>" then refuses to run. randomise_parallel ignores the
# exit code of its fragment step, so without this its defragment step would
# merge an incomplete set of permutations into plausible-looking p-values.
###########################################################################
export LC_ALL=C

say() {
  echo "fsl_sub (local): $*" >&2
}

usage() {
  cat >&2 <<EOF

fsl_sub (nvitk local runner) - runs FSL jobs on this machine, never on a queue

Usage: fsl_sub [options] <command> [args...]
       fsl_sub [options] -t <taskfile>

  -t <filename>   Task file: one command per line, run in parallel (see below)
  -N <jobname>    Job name, used for the log file names
  -l <logdir>     Where to write the logs [default: current directory]
  -j <jid>[,..]   Run only if these jobs (from this fsl_sub) did not fail
  -s <pe>,<n>     Run a single command with OMP_NUM_THREADS=<n>
  -z <output>     Do nothing if <output> already exists
  -v              Verbose: echo each command before running it

Queue options (-T -q -a -p -M -R -m -n -F) are accepted and ignored.

Tasks run at once: \$FSLSUB_PARALLEL, else \$FSLPARALLEL, else \$NSLOTS, else 1.
EOF
  exit 1
}

# Failure markers live here, keyed by job id, so that -j can see them.
state_dir=${FSL_SUB_LOCAL_STATE:-${TMPDIR:-/tmp}}
mkdir -p "$state_dir" 2>/dev/null
jid=$$
rm -f "$state_dir/fsl_sub_local.$jid.failed"

mark_failed() {
  echo "$*" > "$state_dir/fsl_sub_local.$jid.failed"
}

[ $# -eq 0 ] && usage

taskfile=
job_name=
log_dir=
holds=
threads=
verbose=${FSLSUBVERBOSE:-0}

need_value() {
  if [ $# -lt 2 ]; then
    say "option $1 needs a value"
    usage
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    -t) need_value "$@"; taskfile=$2; shift 2 ;;
    -N) need_value "$@"; job_name=$2; shift 2 ;;
    -l) need_value "$@"; log_dir=$2; shift 2 ;;
    -j) need_value "$@"; holds=$2; shift 2 ;;
    -s) need_value "$@"; threads=${2##*,}; shift 2 ;;
    -z)
      need_value "$@"
      if [ -e "$2" ] || [ "$("${FSLDIR:-}/bin/imtest" "$2" 2>/dev/null)" = 1 ]; then
        exit 0
      fi
      shift 2 ;;
    # Queue, architecture, priority, mail and RAM have nothing to choose between locally.
    -T|-q|-a|-p|-M|-R|-m) need_value "$@"; shift 2 ;;
    -n|-F) shift ;;
    -v) verbose=1; shift ;;
    --) shift; break ;;
    -*) say "ignoring unknown option $1"; shift ;;
    *) break ;;
  esac
done

# A job that waits on a failed one must not run: its inputs are incomplete.
if [ -n "$holds" ]; then
  for hold in $(echo "$holds" | tr ',' ' '); do
    if [ -e "$state_dir/fsl_sub_local.$hold.failed" ]; then
      say "not running ${job_name:-${taskfile:-$1}}: job $hold, which it waits for, failed"
      mark_failed "held on failed job $hold"
      echo "$jid"
      exit 1
    fi
  done
fi

log_prefix=
case "$log_dir" in
  "") ;;
  /dev/null*) log_prefix=/dev/null ;;
  *)
    if [ -f "$log_dir" ]; then
      say "log destination $log_dir is a file (should be a folder)"
      exit 1
    fi
    mkdir -p "$log_dir" || exit 1
    log_prefix="${log_dir%/}/" ;;
esac

# log_file <o|e> <suffix>
log_file() {
  if [ "$log_prefix" = /dev/null ]; then
    echo /dev/null
  else
    echo "${log_prefix}${job_name}.$1${jid}$2"
  fi
}

###########################################################################
# A single command
###########################################################################
if [ -z "$taskfile" ]; then
  if [ $# -eq 0 ]; then
    say "supply a command to run or a task file (-t)"
    usage
  fi
  if [ ! -x "$1" ] && ! command -v "$1" >/dev/null 2>&1; then
    say "the command you have requested cannot be found or is not executable: $1"
    exit 1
  fi
  [ -n "$job_name" ] || job_name=$(basename "$1")
  [ -n "$threads" ] && export OMP_NUM_THREADS=$threads
  [ "$verbose" = 1 ] && say "executing: $*"

  out=$(log_file o "")
  err=$(log_file e "")
  "$@" > "$out" 2> "$err"
  rc=$?
  if [ $rc -ne 0 ]; then
    [ "$err" != /dev/null ] && cat "$err" >&2
    mark_failed "exit $rc"
    echo "$jid"
    exit $rc
  fi
  echo "$jid"
  exit 0
fi

###########################################################################
# A task file: one command per line, a bounded number at a time
###########################################################################
if [ $# -gt 0 ]; then
  say "spurious input after the options: \"$*\" -- give a task file or a command, not both"
  exit 1
fi
if [ ! -f "$taskfile" ]; then
  say "task file ($taskfile) does not exist"
  exit 1
fi
ntasks=$(awk 'END { print NR }' "$taskfile")
if [ "$ntasks" -eq 0 ]; then
  say "task file $taskfile is empty; it should list the commands to run"
  exit 1
fi
[ -n "$job_name" ] || job_name=$(basename "$taskfile")

ncpu=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
limit=
from=
for var in FSLSUB_PARALLEL FSLPARALLEL NSLOTS; do
  if [ -n "${!var:-}" ]; then
    limit=${!var}
    from=$var
    break
  fi
done
case "$limit" in
  "")
    limit=1
    [ "$ntasks" -gt 1 ] && say "running tasks one at a time; set FSLSUB_PARALLEL=N to run N at once" ;;
  *[!0-9]*)
    say "ignoring $from=$limit (not a whole number); running tasks one at a time"
    limit=1 ;;
  0) limit=$ncpu ;;
esac
[ "$limit" -gt "$ntasks" ] && limit=$ntasks
# GPU binaries share one device; running them side by side only splits its memory.
case "$(sed -n 1p "$taskfile")" in
  *_gpu*)
    [ "$limit" -gt 1 ] && say "tasks run a GPU binary (_gpu); running them one at a time"
    limit=1 ;;
esac

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MKL_DOMAIN_NUM_THREADS=1 GOTO_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

say "$ntasks task(s) from $(basename "$taskfile"), $limit at a time${from:+ ($from)}"

running=""   # "task:pid" pairs
failed=""
finished=0

# Stop every running task and its children. A task line is "a; b; randomise ...",
# so randomise is a child of the sh running the line: killing only the child lets
# sh go on to its next command, and killing only sh orphans the child. Freeze sh
# first so it cannot start anything new, then terminate it and its children.
stop_all() {
  for entry in $running; do
    pid=${entry#*:}
    kill -STOP "$pid" 2>/dev/null
    kids=$(pgrep -P "$pid" 2>/dev/null || ps -o pid= --ppid "$pid" 2>/dev/null)
    kill -TERM "$pid" $kids 2>/dev/null
    kill -CONT "$pid" 2>/dev/null
  done
}
trap 'say "interrupted; stopping running tasks"; stop_all; mark_failed interrupted; echo "$jid"; exit 130' INT TERM

# Collect every task that has finished since the last call.
reap() {
  live=" $(jobs -pr | tr '\n' ' ') "
  keep=""
  for entry in $running; do
    pid=${entry#*:}
    case "$live" in
      *" $pid "*) keep="$keep $entry"; continue ;;
    esac
    task=${entry%%:*}
    wait "$pid"
    rc=$?
    finished=$((finished + 1))
    if [ $rc -ne 0 ]; then
      failed="$failed $task"
      say "task $task failed (exit $rc) -- $finished/$ntasks done"
    else
      say "task $task finished -- $finished/$ntasks done"
    fi
  done
  running=$keep
}

n=0
while IFS= read -r line || [ -n "$line" ]; do
  n=$((n + 1))
  while :; do
    reap
    set -- $running
    [ $# -lt "$limit" ] && break
    sleep 0.5
  done
  [ "$verbose" = 1 ] && say "executing task $n: $line"
  /bin/sh -c "$line" > "$(log_file o ".$n")" 2> "$(log_file e ".$n")" < /dev/null &
  running="$running $n:$!"
done < "$taskfile"

while [ -n "$running" ]; do
  reap
  [ -n "$running" ] && sleep 0.5
done

if [ -n "$failed" ]; then
  set -- $failed
  say "$# of $ntasks task(s) failed:$failed"
  for task in $failed; do
    err=$(log_file e ".$task")
    if [ "$err" != /dev/null ] && [ -s "$err" ]; then
      say "task $task stderr ($err):"
      tail -n 5 "$err" >&2
    fi
  done
  mark_failed "tasks:$failed"
  echo "$jid"
  exit 1
fi

echo "$jid"
exit 0
