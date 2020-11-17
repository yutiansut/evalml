#!/bin/bash -f
usage() {
    echo "Usage: $0 evalml_debug_logfile [mprofile-filename]"
    exit 1
}

if [ foo"$1" = foo -a foo"$2" = foo ]; then
    usage
fi

logfile=$1
if [ ! -r $logfile ]; then
    echo "Not found: $logfile"
    usage
fi

if [ foo"$2" = foo ]; then
    echo "mprofile-filename missing from command line; only doing time-related calculations"
    mprofile="time_only"
else
    if [ ! -r "$2" ]; then
	echo "Not found: $2"
	usage
    fi

    mprofile=`echo "$2" | sed s/\.dat//`
fi


# get the dir this script is in
pushd $(dirname "${0}") > /dev/null
thisdir=$(pwd -L)
popd > /dev/null

# TODO: add a way to set a start time for time-only analysis
if [ "$mprofile" != "time_only" ]; then
    start=`head -2 $mprofile.dat | tail -1 | awk ' { print $3 }'`
    start_sec=`echo $start | awk ' { print int($3) }'`
    echo "profile start timestamp:" `date -r $start_sec`
fi

cat "$logfile" | grep 'at time' | awk -v start=$start '($10 >= start) { print $10 " " $13; }' | sed s/,// > "$mprofile".evalml_debug.timestamps

if [ "$mprofile" != "time_only" ]; then
    cat "$mprofile".dat | awk ' { print $3 " " $2 " " $1 }' > "$mprofile".timestamps
    cat evalml_debug_timestamps "$mprofile".timestamps | sort -n > "$mprofile".interleaved.timestamps
    echo "$mprofile.interleaved.timestamps - test function-entry times and memory profile snapshots, interleaved and ordered by time"
else
    cat evalml_debug_timestamps | sort -n > "$mprofile".interleaved.timestamps
    echo "$mprofile.interleaved.timestamps - test function-entry times and memory profile snapshots, interleaved and ordered by time"
fi

cat "$mprofile".interleaved.timestamps | awk -f "$thisdir"/run_durations_by_time.awk > "$mprofile".run_durations_by_time
echo "$mprofile.run_durations_by_time - individual test run durations, ordered by start time"

cat "$mprofile".run_durations_by_time | awk ' { print $2 " " $1 }' | sort -nr > "$mprofile".slowest_runs
echo "$mprofile.slowest_runs - individual test run durations, ordered by duration"

cat "$mprofile".interleaved.timestamps | awk -f "$thisdir"/run_durations_grouped.awk | sort -nr > "$mprofile".slowest_tests
echo "$mprofile.slowest_tests - total duration for all runs of each test function, ordered by duration"

if [ "$mprofile" != "time_only" ]; then
   cat "$mprofile".interleaved.timestamps | awk -f "$thisdir"/heap_usage.awk | sort -nr > "$mprofile".worst_heap_usage
   echo "$mprofile.worst_heap_usage - memory usage for all runs of each test function, ordered by max usage"
fi
