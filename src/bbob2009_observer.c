#include "coco.h"

#include "bbob2009_logger.c"

/* TODO:
 *
 * o here needs to go the docstring for this function
 * 
 * o parse options that look like "folder:foo; verbose:bar" (use coco_strfind and/or sscanf and/or ??)
 *   Ideally, valid options should be
 *      "my_folder_name verbose : 3",
 *      "folder: my_folder_name",
 *      "verbose : 4 folder:another_folder"
 *      "folder:yet_another verbose: -2 "
 *   This could be done with a coco_get_option(options, name, format, pointer)
 *   function with code snippets like (approximately)

        logger->folder = coco_allocate_memory(sizeof(char) * (strlen(options) + 1));
        if (!coco_options_read(options, "folder", " %s", logger->folder))
            sscanf(options, " %s", logger->folder);
        coco_options_read(options, "verbose", " %i", &(logger->verbose));

    with 
        
        # caveat: "folder: name; " might fail, use spaces for separations
        int coco_options_read(const char *options, const char *name, const char *format, void *pointer) {
            long i1 = coco_strfind(options, name);
            long i2;
            
            if (i1 < 0)
                return 0;
            i2 = i1 + coco_strfind(&options[i1], ":") + 1;
            if (i2 <= i1)
                return 0;
            return sscanf(&options[i2], format, pointer);
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
