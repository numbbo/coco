## Copyright (C) 2015 Asma
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} my_optimizer (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Asma <asma@pc5-136.lri.fr>
## Created: 2015-01-30

function my_optimizer (f, lower_bounds, upper_bounds, budget)
    disp("In optimizer...")
    n = columns(lower_bounds)
    delta = upper_bounds - lower_bounds
    x = lower_bounds + stdnormal_rnd(n, 1) * delta
    for i = 1:budget
        y = f(x)
    end

endfunction
