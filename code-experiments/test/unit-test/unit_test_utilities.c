#include <math.h>

static int about_equal_value(const double a, const double b) {
  /* Copied from an integration test.
   *
   * Shortcut to avoid the case where a - b is tiny and both a and b
   * are close to or equal to 0.
   *
   * Example: a = +EPS and b = -EPS then the relative error is 2 but
   * in fact the two numbers are both for all practical purposes 0.
   */
  if (a == b)
    return 1;
  {
    const double absolute_error = fabs(a - b);
    const double larger = fabs(a) > fabs(b) ? a : b;
    const double relative_error = fabs((a - b) / larger);

    if (absolute_error < 2 * DBL_MIN)
      return 1;
    return relative_error < 4e-6;
  }
}

static int about_equal_vector(const double *a, const double *b, const size_t dimension) {

  size_t i;

  for (i = 0; i < dimension; i++) {
    if (!about_equal_value(a[i], b[i]))
      return 0;
  }
  return 1;
}


static int about_equal_2d(const double *a, const double b1, const double b2) {

  return (about_equal_value(a[0], b1) && about_equal_value(a[1], b2));

}
