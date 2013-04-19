#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Various tools. 

"""
from __future__ import absolute_import

import os, sys, time
import numpy as np
import warnings

def print_done(message='  done'):
    """prints a message with time stamp"""
    print message, '(' + time.asctime() + ').'

def equals_approximately(a, b, eps=1e-12):
    if a < 0:
        a, b = -1 * a, -1 * b
    return a - eps < b < a + eps or (1 - eps) * a < b < (1 + eps) * a
 
def prepend_to_file(filename, lines, maxlines=1000, warn_message=None):
    """"prepend lines the tex-command filename """
    try:
        lines_to_append = list(open(filename, 'r'))
    except IOError:
        lines_to_append = []
    f = open(filename, 'w')
    for line in lines:
        f.write(line + '\n')
    for i, line in enumerate(lines_to_append):
        f.write(line)
        if i > maxlines:
            print warn_message
            break
    f.close()
        
def truncate_latex_command_file(filename, keeplines=200):
    """truncate file but keep in good latex shape"""
    open(filename, 'a').close()
    lines = list(open(filename, 'r'))
    f = open(filename, 'w')
    for i, line in enumerate(lines):
        if i > keeplines and line.startswith('\providecommand'):
            break
        f.write(line)
    f.close()
    
def strip_pathname(name):
    """remove ../ and ./ and leading/trailing blanks and path separators from input string ``name``"""
    return name.replace('..' + os.sep, '').replace('.' + os.sep, '').strip().strip(os.sep)

def strip_pathname2(name):
    """remove ../ and ./ and leading/trailing blanks and path separators from input string ``name``
    and keep only the last two parts of the path"""
    return os.sep.join(name.replace('..' + os.sep, '').replace('.' + os.sep, '').strip().strip(os.sep).split(os.sep)[-2:])

def str_to_latex(string):
    """do replacements in ``string`` such that it most likely compiles with latex """
    return string.replace('\\', r'\textbackslash{}').replace('_', '\\_').replace(r'^', r'\^\,').replace(r'%', r'\%').replace(r'~', r'\ensuremath{\sim}').replace(r'#', r'\#')

def number_of_digits(val, precision=1e-13):
    """returns the number of non-zero digits of a number, e.g. two for 1200 or three for 2.03.
    
    """  
    raise NotImplementedError()

def num2str(val, significant_digits=2, force_rounding=False, 
            max_predecimal_digits=5, max_postdecimal_leading_zeros=1, 
            remove_trailing_zeros=True):
    """returns the shortest string representation with either ``significant_digits`` 
    digits shown or its true value, whichever is shorter.
    
    ``force_rounding`` shows no more than the desired number of significant digits, 
    which means, e.g., ``12345``  becomes ``12000``. 
    
    ``remove_trailing_zeros`` removes zeros, if and only if the value is exactly. 
     
    >>> from bbob_pproc import toolsdivers as as bb
    >>> print [td.num2str(val) for val in [12345, 1234.5, 123.45, 12.345, 1.2345, .12345, .012345, .0012345]
    ['12345', '1.2e3', '1.2e2', '12', '1.2', '0.12', '0.012', '1.2e-3']
    
    """
    if val == 0:
        return '0'
    assert significant_digits > 0
    is_negative = val < 0
    original_value = val
    val = float(np.abs(val))

    order_of_magnitude = int(np.floor(np.log10(val)))
    # number of digits before decimal point == order_of_magnitude + 1
    fac = 10**(significant_digits - 1 - order_of_magnitude)
    val_rounded = np.round(fac * val) / fac

    # the strategy is now to produce two string representations 
    # cut each down to the necessary length and return the better 
    
    # the first is %f format
    if order_of_magnitude + 1 >= significant_digits:
        s = str(int(val_rounded if force_rounding else np.round(val)))
    else:
        s = str(val_rounded)
        idx1 = 0  # first non-zero index
        while idx1 < len(s) and s[idx1] in ('-', '0', '.'):
            idx1 += 1  # find index of first significant number
        idx2 = idx1 + significant_digits + (s.find('.') > idx1)
        # print val, val_rounded, s, len(s), idx1, idx2
        # pad some zeros in the end, in case
        if val != val_rounded:
            if len(s) < idx2:
                s += '0' * (idx2 - len(s))
        # remove zeros from the end, in case
        if val == val_rounded and remove_trailing_zeros:
            while s[-1] == '0': 
                s = s[0:-1]
        if s[-1] == '.':
            s = s[0:-1]
    s_float = ('-' if is_negative else '') + s

    # now the second, %e format
    s = ('%.' + str(significant_digits - 1) + 'e') % val
    if eval(s) == val and s.find('.') > 0:
        while s.find('0e') > 0:
            s = s.replace('0e', 'e')
    s = s.replace('.e', 'e')
    s = s.replace('e+', 'e')
    while s.find('e0') > 0:
        s = s.replace('e0', 'e')
    while s.find('e-0') > 0:
        s = s.replace('e-0', 'e-')
    if s[-1] == 'e':
        s = s[:-1]
    s_exp = ('-' if is_negative else '') + s 
    
    # print s_float, s_exp
    
    # now return the better (most of the time the shorter) representation
    if (len(s_exp) < len(s_float) or 
        s_float.find('0.' + '0' * (max_postdecimal_leading_zeros + 1)) > -1 or
        np.abs(val_rounded) >= 10**(max_predecimal_digits + 1)
        ):
        return s_exp
    else:
        return s_float

def number_to_latex(number_as_string):
    """usage as ``number_to_latex(num2str(1.023e-12)) == "'-1.0\\times10^{-12}'"``"""
    s = number_as_string
    if s.find('e') > 0:
        if s.startswith('1e') or s.startswith('-1e'):
            s = s.replace('1e', '10^{')
        else:
            s = s.replace('e', '\\times10^{')
        s += '}'
    return s
