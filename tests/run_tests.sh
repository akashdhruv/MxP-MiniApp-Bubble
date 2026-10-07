#!/bin/sh
# make check harness.
#
# usage: sh tests/run_tests.sh <objdir> [test binaries...]
#
# Each test source may declare the rank counts it must be run on with a line
#     // ranks: 1 2 4
# near the top; the default is 1 rank.  A test passes when its mpirun exit
# status is 0 on every declared rank count.  Every run is wrapped in a timeout
# so a deadlocked halo exchange fails the round instead of hanging it.

OBJDIR="$1"
shift

TIMEOUT="timeout 900"   # test_conservation runs 2 x 20478 steps at 256^2
MPIRUN="mpirun"
# allow running as root / oversubscribed inside containers
MPIFLAGS="--oversubscribe"

npass=0
nfail=0
failed=""

if [ $# -eq 0 ]; then
    echo "make check: no tests present yet (0 tests)"
    exit 0
fi

for bin in "$@"; do
    name=$(basename "$bin")
    src="tests/$name.cpp"
    ranks=$(sed -n 's,^//[[:space:]]*ranks:[[:space:]]*,,p' "$src" | head -n 1)
    [ -n "$ranks" ] || ranks=1
    for np in $ranks; do
        printf '  %-24s np=%-2s ... ' "$name" "$np"
        out=$($TIMEOUT $MPIRUN $MPIFLAGS -np "$np" "$bin" 2>&1)
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "PASS"
            npass=$((npass + 1))
        else
            echo "FAIL (exit $rc)"
            echo "$out" | sed 's,^,      ,'
            nfail=$((nfail + 1))
            failed="$failed $name(np=$np)"
        fi
    done
done

echo "make check: $npass passed, $nfail failed"
if [ $nfail -ne 0 ]; then
    echo "failed:$failed"
    exit 1
fi
exit 0
