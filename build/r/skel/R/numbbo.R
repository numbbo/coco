#' @useDynLib coco do_get_problem
#' @useDynLib coco do_evaluate_function
#' @useDynLib coco do_lower_bounds
#' @useDynLib coco do_upper_bounds
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
coco_get_generator <- function(benchmark) {
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
coco_benchmark <- function(benchmark,
                           result_directory,
                           optimize) {
  generator <- coco_get_generator(benchmark)
  function_id <- 0
  repeat {
    f <- generator(function_id, as.character(result_directory))
    if (is.null(f))
      break
    optimize(f, lower_bounds(f), upper_bounds(f))
    function_id <- function_id + 1
  }
}
