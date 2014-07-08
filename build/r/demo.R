library("numbbo")

my_optimizer <- function(f, lower, upper) {
  n <- length(lower)
  delta <- upper - lower
  for (i in 1:100000) {
    f(lower + runif(n) * delta)
  }
}

numbbo_benchmark("toy_suit", "toy_observer", my_optimizer)

