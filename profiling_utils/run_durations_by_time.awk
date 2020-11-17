BEGIN {
  f = ""
}

($2 ~ "\\\(\\\)$") {
  # time, function
  if (f != "") {
    printf "%s %3.6f\n", f, ($1 - start_time) # time for this test
  }
  start_time = $1
  f = $2
  next
}

