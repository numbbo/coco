##
## Generate test cases to compare the BBOB 2009 functions in the old
## and new framework.
##

library("soobench")

encode_fid <- function(fid, iid, d) {
  low_iid <- (iid - 1) %% 5
  high_iid <- (iid - 1) %/% 5
  res <- low_iid
  res <- res + (fid - 1) * 5
  res <- res + (which(d == c(2, 3, 5, 10, 20, 40)) - 1) * (5 * 24)
  res <- res + high_iid * (5 * 24 * 6)
  res
}

catf <- function(fmt, ...) cat(sprintf(fmt, ...))

generate_testvectors <- function(n) {
  m <- runif(40*n, min=-5, max=5)
  dim(m) <- c(n, 40)
  m
}

generate_bbob_testcase <- function(function_id, instance_id, dimension,
                                   testvectors) {
  function_index <- encode_fid(function_id, instance_id, dimension);
  f <- bbob2009_function(dimension, function_id, instance_id)
  for (i in 1:nrow(testvectors)) {
    par <- testvectors[i,1:dimension]
    value <- f(par)
    catf("%i %i %.6e\n", function_index, i - 1, value)
  }
}

testvectors <- generate_testvectors(100)

catf("bbob2009\n%i\n", nrow(testvectors))
for (i in 1:nrow(testvectors)) {
  catf("%s\n", paste(sprintf("%.14e", testvectors[i,]), collapse=" "))
}

set.seed(42)

res <- NULL
for (high_instance_id in 0:2) {
  for (dimension in c(2, 3, 5, 10, 20, 40)) {
    for (function_id in c(1:18)) {
      for (low_instance_id in 1:5) {
        instance_id <- low_instance_id + 5 * high_instance_id
        generate_bbob_testcase(function_id, instance_id, dimension, testvectors)
      }
    }
  }
}
