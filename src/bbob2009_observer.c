#include "coco.h"

#include "bbob2009_logger.c"

/* TODO:
 * o parse options that look like "folder:foo; verbos...:bar" (use coco_strfind and/or sscanf and/or ??)
 *   Ideally, valid options should be
 *      "my_folder_name verbose : 3",
 *      "folder: my_folder_name",
 *      "verbose : 4 folder:another_folder"
 *      "folder:yet_another ; verbose: -2 "
 *   This should be done in a bbob2009_logger_read_options(logger, options)
 *   function with code snippets like (approximately)
     idx = coco_strfind(options, "folder");
     if (idx >= 0) {
        / * (re-)allocate logger->folder if needed, in which case assert is sufficient * /
        if (strlen(options) > strlen(logger->folder))
            coco_error(...);
        sscanf(options[idx], "folder : %s", logger->folder);
    }
 * 
 */   
static coco_problem_t *bbob2009_observer(coco_problem_t *problem,
                                  const char *options) {
  if (problem == NULL)
    return problem;
  /* TODO: " */
  coco_create_path(options);
  problem = bbob2009_logger(problem, options);
  return problem;
}
