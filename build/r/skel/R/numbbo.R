#' @useDynLib numbbo do_get_problem
#' @useDynLib numbbo do_evaluate_function
#' @useDynLib numbbo do_lower_bounds
#' @useDynLib numbbo do_upper_bounds
{}

#' @export
lower_bounds <- function(f) {
  problem <- environment(f)$problem
  .Call(do_lower_bounds, problem)
}

#' @export
upper_bounds <- function(f) {  
  problem <- environment(f)$problem
  .Call(do_upper_bounds, problem)
}

#' @export
numbbo_get_generator <- function(benchmark) {
  force(benchmark)
  function(function_id, result_directory) {
    problem <- .Call(do_get_problem,
                     as.character(benchmark),
                     as.integer(function_id))
    if (is.null(problem))
      return(NULL)
    print(problem)
    res <- function(x) {
      .Call(do_evaluate_function, problem, as.numeric(x))
    }
    class(res) <- "numbbo_problem"
    res
  }
}

#' @export
numbbo_benchmark <- function(benchmark,
                             result_directory,
                             optimize) {
  generator <- numbbo_get_generator(benchmark)
  i <- 0
  repeat {
    f <- generator(i, as.character(result_directory))
    if (is.null(f))
      break
    optimize(f, lower_bounds(f), upper_bounds(f))
    i <- i + 1
  }
}
