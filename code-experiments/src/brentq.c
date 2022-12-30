/*
    This is a modification of the original file:
    Written by Charles Harris charles.harris@sdl.usu.edu

    Edited by Paul Dufosse paul.dufosse@inria.fr
    - make it standalone
      - removing SciPy-dependent stuff
      - merging with zeros.h declaration file
    - wrapping in brentinv(f, y, ...)
      - automatic range interval computation [xa, xb]
      - assume f has fixed point in 0

    Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>

typedef double (*callback_type)(double, void*);

static double brentq(callback_type f, double y, double xa, double xb, double xtol,
                     double rtol, int iter, void *func_data);

static double brentinv(callback_type f, double y, void *func_data);

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*
  Note: signbit is in the C99 math library and we compile with C89 standard.
*/
#ifndef signbit
#define signbit(x)((x) < 0 ? 1 : 0)
#endif

/*
  At the top of the loop the situation is the following:

    1. the root is bracketed between xa and xb
    2. xa is the most recent estimate
    3. xp is the previous estimate
    4. |fp| < |fb|

  The order of xa and xp doesn't matter, but assume xp < xb. Then xa lies to
  the right of xp and the assumption is that xa is increasing towards the root.
  In this situation we will attempt quadratic extrapolation as long as the
  condition

  *  |fa| < |fp| < |fb|

  is satisfied. That is, the function value is decreasing as we go along.
  Note the 4 above implies that the right inequality already holds.

  The first check is that xa is still to the left of the root. If not, xb is
  replaced by xp and the interval reverses, with xb < xa. In this situation
  we will try linear interpolation. That this has happened is signaled by the
  equality xb == xp;

  The second check is that |fa| < |fb|. If this is not the case, we swap
  xa and xb and resort to bisection.

*/

double brentq(callback_type f, double y, double xa, double xb, double xtol, double rtol,
       int iter, void *func_data)  {

    double xpre = xa, xcur = xb;
    double xblk = 0., fpre, fcur, fblk = 0., spre = 0., scur = 0., sbis;
    /* the tolerance is 2*delta */
    double delta;
    double stry, dpre, dblk;
    int i;

    fpre = (*f)(xpre, func_data) - y;
    fcur = (*f)(xcur, func_data) - y;
    if (fpre*fcur > 0) {
        /* Sign error */
        return NAN;
    }
    if (fpre == 0) {
        /* Converged */
        return xpre;
    }
    if (fcur == 0) {
        /* Converged*/
        return xcur;
    }

    for (i = 0; i < iter; i++) {
        if (fpre != 0 && fcur != 0 &&
	    (signbit(fpre) != signbit(fcur))) {
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre;
        }
        if (fabs(fblk) < fabs(fcur)) {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;

            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        delta = (xtol + rtol*fabs(xcur))/2;
        sbis = (xblk - xcur)/2;
        if (fcur == 0 || fabs(sbis) < delta) {
            /* Converged*/
            return xcur;
        }

        if (fabs(spre) > delta && fabs(fcur) < fabs(fpre)) {
            if (xpre == xblk) {
                /* interpolate */
                stry = -fcur*(xcur - xpre)/(fcur - fpre);
            }
            else {
                /* extrapolate */
                dpre = (fpre - fcur)/(xpre - xcur);
                dblk = (fblk - fcur)/(xblk - xcur);
                stry = -fcur*(fblk*dblk - fpre*dpre)
                    /(dblk*dpre*(fblk - fpre));
            }
            if (2*fabs(stry) < MIN(fabs(spre), 3*fabs(sbis) - delta)) {
                /* good short step */
                spre = scur;
                scur = stry;
            } else {
                /* bisect */
                spre = sbis;
                scur = sbis;
            }
        }
        else {
            /* bisect */
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur; fpre = fcur;
        if (fabs(scur) > delta) {
            xcur += scur;
        }
        else {
            xcur += (sbis > 0 ? delta : -delta);
        }

        fcur = (*f)(xcur, func_data) - y;
    }
    /* Iterations exceeded*/
    return xcur;
}

/*
  Inverse of one-dimensional function x such that y = T(x).

  The transformation T is assumed to have fixed point at 0.
  The method uses Brent's method (brentq) and the initial interval [a, b], a < b,
  is computed such that:
    - either a > 0 or b < 0
    - either a=y or b=y
  proceeding with halving/doubling.
*/

double brentinv(callback_type f, double y, void *func_data) {
    double xres, xmin, xmax, fval;

    fval = (*f)(y, func_data);
    xmax = y;
    xmin = y;

    if (y > 0) {
        if (fval > y) {
            while (fval > y) {
                xmin = xmin / 2;
                fval = (*f)(xmin, func_data);
            }
        } else {
            while (fval < y) {
                xmax = xmax * 2;
                fval = (*f)(xmax, func_data);
            }
        }
    } else {
        if (fval > y) {
            while (fval > y) {
                xmin = xmin * 2;
                fval = (*f)(xmin, func_data);
            }
        } else {
            while (fval < y) {
                xmax = xmax / 2;
                fval = (*f)(xmax, func_data);
            }
        }
    }

    xres = brentq(f, y, xmin, xmax, 1E-14, 1E-10, 200, func_data);
    return xres;
}
