library("coco")

my_optimizer <- function(f, lower, upper) {
  n <- length(lower)
  delta <- upper - lower
  for (i in 1:100000) {
    f(lower + runif(n) * delta)
  }
}

coco_benchmark("suite_toy", "observer_toy", my_optimizer)

