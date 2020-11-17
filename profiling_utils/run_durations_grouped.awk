BEGIN {
  f = ""
}

($2 ~ "\\\(\\\)$") {
  duration = ($1 - start_time)
  if (f != "") {
    if (! f in collated_durations)
      collated_durations[f] = 0
    collated_durations[f] += duration
  }
  start_time = $1
  f = $2
}

END {
  for (f in collated_durations) {
    printf "%3.6f %s\n", collated_durations[f], f
  }
}
