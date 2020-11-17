# start_mems[f] is not set if we haven't seen f before
# elseit is set to 0 if we've seen the function start line but no MEM line
# else it is set to the first memory sample after the start line
BEGIN {
  f = ""
}

($2 ~ "\\\(\\\)$") {
  if (f != $2) {
    # new function
    if (0 != start_mems[f]) # we've seen at least one MEM between functions
      if ("" == f)
	printf "%6.1f %s\n", last_mem - start_mems[f], "test harness initialization"
      else
	printf "%6.1f %s\n", last_mem - start_mems[f], f

    start_mems[f] = 0
  }

  f = $2
  next
}

($3 == "MEM") {
  if (0 == start_mems[f]) 
    start_mems[f] = $2

  last_mem = $2 # always set, in case we only have one MEM for a function
  next
}


