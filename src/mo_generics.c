#include <stdlib.h>
#include <stdio.h>
#include "coco.h"

/**
 * Allocates memory for a matrix with ${number_of_rows} rows and ${number_of_columns} columns of strings of
 * maximal size ${max_length_of_string}.
 */
static char ***mococo_allocate_matrix_of_strings(const size_t number_of_rows, const size_t number_of_columns,
    const size_t max_length_of_string) {

  char ***matrix_of_strings;
  size_t i, j;

  matrix_of_strings = malloc(number_of_rows * sizeof(char**));
  if (matrix_of_strings == NULL)
    coco_error("mococo_allocate_matrix_of_strings() failed");

  for(i = 0; i < number_of_rows; i++) {
    matrix_of_strings[i] = malloc(number_of_columns * sizeof(char*));
      for(j = 0; j < number_of_columns; j++) {
        matrix_of_strings[i][j] = malloc(max_length_of_string * sizeof(char));
      }
  }

  return matrix_of_strings;
}

/**
 * Frees the memory occupied by the ${matrix_of_strings} matrix of string with ${number_of_rows} rows and
 * ${cnumber_of_columns} columns.
 */
static void mococo_free_matrix_of_strings(char ***matrix_of_strings, const size_t number_of_rows,
    const size_t number_of_columns) {

  size_t i, j;

  for(i = 0; i < number_of_rows; i++) {
      for(j = 0; j < number_of_columns; j++) {
        free(matrix_of_strings[i][j]);
      }
      free(matrix_of_strings[i]);
  }
  free(matrix_of_strings);
}

/**
 * Counts and returns the number of lines in the already opened file pointed to by ${fp}.
 */
static size_t mococo_get_file_line_number(FILE *fp) {

  int ch;
  size_t number_of_lines = 0;

  /* Count the number of lines */
  do {
    ch = fgetc(fp);
    if (ch == '\n')
      number_of_lines++;
  } while (ch != EOF);
  /* Add 1 if the last line doesn't end with \n */
  if(ch != '\n' && number_of_lines != 0)
      number_of_lines++;

  /* Return to the beginning of the file */
  rewind(fp);

  return number_of_lines;
}

/**
 * Assumes the input file contains lines with single double values. Reads and returns those values.
 */
static double *mococo_get_doubles_from_file(const char *filename) {

  size_t i, number_of_lines;
  double *result;

  FILE *fp = fopen(filename, "r");
  if (fp == 0) {
    coco_error("failed to open input file %s", filename);
  }
  number_of_lines = mococo_get_file_line_number(fp);

  /* Prepare the vector */
  result = coco_allocate_vector(number_of_lines);
  for (i = 0; i < number_of_lines; i++) {
    fscanf(fp, "%lf", &result[i]);
  }
  fclose(fp);

  return result;
}

/**
 * Assumes the input file contains pairs of strings separated by tabs and that each line is of maximal length
 * ${max_length_of_string}. Returns the strings in the form of a N x 2 matrix, where N is the number of lines
 * in the file.
 */
static char ***mococo_get_string_pairs_from_file(const char *filename, const size_t max_length_of_string) {

  size_t i, number_of_lines;
  FILE *fp = fopen(filename, "r");
  char ***matrix_of_strings;

  if (fp == 0) {
    coco_error("failed to open input file %s", filename);
  }
  number_of_lines = mococo_get_file_line_number(fp);

  /* Prepare the matrix */
  matrix_of_strings = mococo_allocate_matrix_of_strings(number_of_lines, 2, max_length_of_string);
  for (i = 0; i < number_of_lines; i++) {
    fscanf(fp, "%s\t%[^\n]", matrix_of_strings[i][0], matrix_of_strings[i][1]);
  }
  fclose(fp);

  return matrix_of_strings;
}

/**
 * Converts string (char *) to double. Does not check for underflow or overflow, ignores any trailing characters.
 */
static double mococo_string_to_double(const char *string) {
  double result;
  char *err;

  result = strtod(string, &err);
  if (result == 0 && string == err) {
    coco_error("mococo_string_to_double() failed");
  }

  return result;
}

/**
 * Scans the input matrix to find the first value that matches the given ${key} (looks for the key in ${key_column} column and
 * for the value in ${value_column} column). If the key is not found, it returns ${default_value}.
 */
static double mococo_get_matching_double_value(const char ***matrix_of_strings, const char *key, const size_t number_of_rows,
    const size_t key_column, const size_t value_column, double default_value) {

  size_t i;

  for (i = 0; i < number_of_rows; i++) {
    /* The given key matches the key in the matrix */
    if (strcmp(matrix_of_strings[i][key_column], key) == 0) {
      /* Return the value in the same row */
      return mococo_string_to_double(matrix_of_strings[i][value_column]);
    }
  }

  /* The key was not found, therefore the default value is returned */
  return default_value;
}

/**
 * Appends ${string} to the given file.
 */
static void mococo_append_to_file(const char *filename, const char *string) {

  FILE *fp = fopen(filename, "a");
  fprintf(fp, string);
  fprintf(fp, "\n");
  fclose(fp);
}

/**
 * Checks the dominance relation in the unconstrained minimization case between
 * ${objectives1} and ${objectives2} and returns:
 *  1 if ${objectives1} dominates ${objectives2}
 *  0 if ${objectives1} and ${objectives2} are non-dominated
 * -1 if ${objectives2} dominates ${objectives1}
 * -2 if ${objectives1} is identical to ${objectives2}
 */
static int mococo_get_dominance(const double *objectives1, const double *objectives2, const size_t num_obj) {
  /* TODO: Should we care about comparison precision? */
  size_t i;

  int flag1 = 0;
  int flag2 = 0;

  for (i = 0; i < num_obj; i++) {
    if (objectives1[i] < objectives2[i]) {
      flag1 = 1;
    } else if (objectives1[i] > objectives2[i]) {
      flag2 = 1;
    }
  }

  if (flag1 && !flag2) {
    return 1;
  } else if (!flag1 && flag2) {
    return -1;
  } else if (flag1 && flag2) {
    return 0;
  } else { /* (!flag1 && !flag2) */
    return -2;
  }
}
