#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for writing TeX for tables."""

import os
import sys
import string
import numpy

from bbob_pproc import toolsstats
from pdb import set_trace

#GLOBAL VARIABLES DEFINITION
alphabet = string.ascii_letters
#conversion of matplotlib elements to LaTeX
latex_marker_map = {'o': r'$\circ$',
              'd': r'$\diamondsuit$',
              's': r'$\Box$',
              'v': r'$\triangledown$',
              '*': r'$\star$',
              'h': r'$\varhexagon$', # need \usepackage{wasysymb}
              '^': r'$\triangle$',
              'p': r'$\pentagon$', # need \usepackage{wasysymb}
              'H': r'$\hexagon$', # need \usepackage{wasysymb}
              '<': r'$\triangleleft$',
              'D': r'$\Diamond$',
              '>': r'$\triangleright$',
              '1': r'$\downY$', # need \usepackage{MnSymbol}
              '2': r'$\upY$', # need \usepackage{MnSymbol}
              '3': r'$\rightY$', # need \usepackage{MnSymbol}
              '4': r'$\leftY$'} # need \usepackage{MnSymbol}
latex_color_map_old = {'g': 'green!45!black',
             'r': 'red',
             'c': 'cyan',
             'm': 'magenta',
             'y': 'yellow',
             'k': 'black',
             'b': 'blue'}
latex_color_map = {'#000080': 'NavyBlue',
             'r': 'red',
             '#ffd700': 'Goldenrod',
             '#d02090': 'VioletRed',
             'k': 'Black',
             '#6495ed': 'CornflowerBlue',
             '#ff4500': 'OrangeRed',
             '#ffff00': 'Yellow',
             '#ff00ff': 'Magenta',
             '#bebebe': 'Gray',
             '#87ceeb': 'SkyBlue',
             '#ffa500': 'Orange',
             '#ffc0cb': 'Lavender',
             '#4169e1': 'RoyalBlue',
             '#228b22': 'ForestGreen',
             '#32cd32': 'LimeGreen',
             '#9acd32': 'YellowGreen',
             '#adff2f': 'GreenYellow'}



#CLASS DEFINITION
class Error(Exception):
    """ Base class for errors. """
    pass

class WrongInputSizeError(Error):
    """Error if an array has the wrong size for the following operation.

    :returns: message containing the size of the array and the required
              size.

    """
    def __init__(self,arrName, arrSize, reqSize):
        self.arrName = arrName
        self.arrSize = arrSize
        self.reqSize = reqSize

    def __str__(self):
        message = 'The size of %s is %s. One dimension must be of length %s!' %\
                  (self.arrName,str(self.arrSize), str(self.reqSize))
        return repr(message)


#TOP LEVEL METHODS
def color_to_latex(color):
    try:
        res = '\color{%s}' % latex_color_map[color]
    except KeyError, err:
        try:
            float(color)
            res = '\color[gray]{%s}' % color
        except ValueError:
            raise err
    return res

def marker_to_latex(marker):
    return latex_marker_map[marker]

def numtotext(n):
    """Returns a text from a positive integer.

    Is to be used for generating command names: they cannot include number
    characters.

    WARNING: n can only be smaller than 51

    """
    if n > 51:
        raise Exception('Cannot handle a number of algorithms that large.')

    return alphabet[n]

def writeLabels(label):
    """Format text to be output by LaTeX."""
    #return label.replace('_', r'\_')

def writeFEvals(fevals, precision='.2'):
    """Returns string representation of a number of function evaluations."""

    if numpy.isinf(fevals):
        return r'$\infty$'

    tmp = (('%' + precision + 'g') % fevals)
    res = tmp.split('e')
    if len(res) > 1:
        res[1] = '%d' % int(res[1])
        res = '%s' % 'e'.join(res)
        pr2 = str(float(precision) + .2)
        #res2 = (('%' + pr2 + 'g') % fevals)
        res2 = (('%' + pr2 + 'g') % float(tmp))
        # To have the same number of significant digits.
        if len(res) >= len(res2):
            res = res2
    else:
        res = res[0]
    return res

def writeFEvals2(fevals, precision=2, maxdigits=None, isscientific=False):
    """Returns string representation of a number of function evaluations.

    This method is supposed to be used for filling up a LaTeX tabular.

    To address the eventual need to keep their string representation
    short, the method here proposes the shortest representation between
    the full representation and a modified scientific representation.

    :param float fevals:
    :param int precision: number of significant digits
    :param int maxdigits:
    :param bool isscientific:

    Examples:
    
    ======   =========   =====================
    Number   Precision   Output Representation
    ======   =========   =====================
    102345   2 digits    1.0e5
    ======   =========   =====================

    """

    #Printf:
    # %[flags][width][.precision][length]specifier

    assert not numpy.isnan(fevals)

    if numpy.isinf(fevals):
        return r'$\infty$'

    if maxdigits is None:
        precision = int(precision)

        #repr1 is the alternative scientific notation
        #repr2 is the full notation but with a number of significant digits given
        #by the variable precision.

        res = (('%.' + str(precision-1) + 'e') % fevals)
        repr1 = res
        tmp = repr1.split('e')
        tmp[1] = '%d' % int(tmp[1]) # Drop the eventual plus sign and trailing zero
        repr1 = 'e'.join(tmp)

        repr2 = (('%.' + str(precision+1) + 'f') % float(res)).rstrip('0').rstrip('.')
        #set_trace()
        if len(repr1) > len(repr2) and not isscientific:
            return repr2

        return repr1

    else:
        # takes precedence, in this case we expect a positive integer
        if not isinstance(fevals, int):
            return '%d' % fevals

        repr2 = '%.0f' % fevals
        if len(repr2) > maxdigits:
            precision = maxdigits - 4
            # 1) one symbol for the most significant digit
            # 2) one for the dot, 3) one for the e, 4) one for the exponent
            if numpy.log10(fevals) > 10:
                precision -= 1
            if precision < 0:
                precision = 0
            repr1 = (('%.' + str(precision) + 'e') % fevals).split('e')
            repr1[1] = '%d' % int(repr1[1]) # drop the sign and trailing zero
            repr1 = 'e'.join(repr1)
            return repr1

        return repr2

def writeFEvalsMaxSymbols(fevals, maxsymbols, isscientific=False):
    """Return the smallest string representation of a number.

    This method is only concerned with the maximum number of significant
    digits.

    Two alternatives:

    1) modified scientific notation (without the trailing + and zero in
       the exponent) 
    2) float notation

    :returns: string representation of a number of function evaluations
              or ERT.

    """

    #Compared to writeFEvals2?
    #Printf:
    # %[flags][width][.precision][length]specifier

    assert not numpy.isnan(fevals)

    if numpy.isinf(fevals):
        return r'$\infty$'

    #repr1 is the alternative scientific notation
    #repr2 is the full notation but with a number of significant digits given
    #by the variable precision.

    # modified scientific notation:
    #smallest representation of the decimal part
    #drop + and starting zeros of the exponent part
    repr1 = (('%.' + str(maxsymbols) + 'e') % fevals)
    size1 = len(repr1)
    tmp = repr1.split('e', 1)
    tmp2 = tmp[-1].lstrip('+-0')
    if float(tmp[-1]) < 0:
        tmp2 = '-' + tmp2
    tmp[-1] = tmp2
    remainingsymbols = max(maxsymbols - len(tmp2) - 2, 0)
    tmp[0] = (('%.' + str(remainingsymbols) + 'f') % float(tmp[0]))
    repr1 = 'e'.join(tmp)
    #len(repr1) <= maxsymbols is not always the case but should be most usual

    tmp = '%.0f' % fevals
    remainingsymbols = max(maxsymbols - len(tmp), 0)
    repr2 = (('%.' + str(remainingsymbols) + 'f') % fevals)
    tmp = repr2.split('.', 1)
    if len(tmp) > 1:
        tmp[-1] = tmp[-1].rstrip('0')
    repr2 = '.'.join(tmp)
    repr2 = repr2.rstrip('.')
    #set_trace()

    if len(repr1)-repr1.count('.') < len(repr2)-repr2.count('.') or isscientific:
        return repr1

    #tmp1 = '%4.0f' % bestalgdata[-1]
    #tmp2 = ('%2.2g' % bestalgdata[-1]).split('e', 1)
    #if len(tmp2) > 1:
    #    tmp2[-1] = tmp2[-1].lstrip('+0')
    #    tmp2 = 'e'.join(tmp2)
    #    tmp = tmp1
    #    if len(tmp1) >= len(tmp2):
    #        tmp = tmp2
    #    curline.append(r'\multicolumn{2}{c|}{%s}' % tmp)

    return repr2

def writeFEvalsMaxPrec(entry, SIG, maxfloatrepr=1e5):
    """Return a string representation of a number.

    Two alternatives:

    1) float notation with a precision smaller or equal to SIG (if the
       entry is one, then the result is 1).
    2) if the number is larger or equal to maxfloatrepr, a modified
       scientific notation (without the trailing + and zero in the
       exponent)

    :returns: string representation of a number of function evaluations
              or ERT.

    """
    #CAVE: what if entry is smaller than 10**(-SIG)?
    #Printf:
    # %[flags][width][.precision][length]specifier

    assert not numpy.isnan(entry)

    if numpy.isinf(entry):
        return r'$\infty$'

    if entry == 1.:
        res = '1'
    elif entry < maxfloatrepr:
        # the full notation but with given maximum precision
        corr = 1 if abs(entry) < 1 else 0
        tmp = '%.0f' % entry
        remainingsymbols = max(SIG - len(tmp) + corr, 0)
        res = (('%.' + str(remainingsymbols) + 'f') % entry)
    else:
        # modified scientific notation:
        #smallest representation of the decimal part
        #drop + and starting zeros of the exponent part
        res = (('%.' + str(max([0, SIG - 1])) + 'e') % entry)
        size1 = len(res)
        tmp = res.split('e', 1)
        tmp2 = tmp[-1].lstrip('+-0')
        if float(tmp[-1]) < 0:
            tmp2 = '-' + tmp2
        tmp[-1] = tmp2
        if len(tmp) > 1 and tmp[-1]:
            res = 'e'.join(tmp)
        else:
            res = tmp[0]

    return res

def tableLaTeX(table, spec, extraeol=()):
    """Generates a tabular from a sequence of sequence of strings.

    :param seq table: sequence of sequence of strings
    :param string spec: string for table specification, see
                        http://en.wikibooks.org/wiki/LaTeX/Tables#The_tabular_environment 
    :param seq extraeol: sequence of string the same length as the table
                         (same number of lines) which are added at the
                         end of each line.
    :returns: sequence of strings of a LaTeX tabular.

    """

    if not extraeol:
        extraeol = len(table) * ['']

    # TODO: check that spec and extraeol have the right format? 

    res = [r'\begin{tabular}{%s}' % spec]
    for i, line in enumerate(table[:-1]):
        curline = ' & '.join(line) + r'\\' + extraeol[i]
#        curline = ' & '.join(line) + r'\\\hline' + extraeol[i]
        res.append(curline)
    res.append(' & '.join(table[-1]) + extraeol[-1])

    res.append(r'\end{tabular}')
    res = '\n'.join(res)
    return res


def tableXLaTeX(table, spec, extraeol=()):
    """Generates a tabular from a sequence of sequence of strings.

    :param seq table: sequence of sequence of strings
    :param string spec: string for table specification, see
                        http://en.wikibooks.org/wiki/LaTeX/Tables#The_tabular_environment 
    :param seq extraeol: sequence of string the same length as the table
                         (same number of lines) which are added at the
                         end of each line.
    :returns: sequence of strings of a LaTeX tabular.

    """

    if not extraeol:
        extraeol = len(table) * ['']

    # TODO: check that spec and extraeol have the right format? 
    if 1 < 3:
        res = [r'\begin{tabularx}{1.0\textwidth}{%s}' % spec]
        for i, line in enumerate(table[:-1]):
            curline = ' & '.join(line) + r'\\' + extraeol[i]
            res.append(curline)
    else: # format with hline, when is it needed, for non-paper tables?
        res = [r'\begin{tabularx}{1.3\textwidth}{%s}' % spec]
        for i, line in enumerate(table[:-1]):
            curline = ' & '.join(line) + r'\\\hline' + extraeol[i]
            res.append(curline)
    
    res.append(' & '.join(table[-1]) + extraeol[-1])

    res.append(r'\end{tabularx}')
    res = '\n'.join(res)
    return res

def tableLaTeXStar(table, width, spec, extraeol=()):
    """Generates a tabular\* from a sequence of sequence of strings

    :param seq table: sequence of sequence of strings
    :param string width: string for the width of the table
    :param strin spec: string for table specification, see
                       http://en.wikibooks.org/wiki/LaTeX/Tables#The_tabular_environment 
    :param seq extraeol: sequence of string the same length as the table
                         (same number of lines) which are added at the
                         end of each line.

    """
    if not extraeol:
        extraeol = len(table) * ['']


    # TODO: check that spec and extraeol have the right format?

    res = [r'\begin{tabular*}{%s}{%s}' % (width, spec)]
    for i, line in enumerate(table[:-1]):
        curline = ' & '.join(line) + r'\\' + extraeol[i]
        res.append(curline)
    res.append(' & '.join(table[-1]) + extraeol[-1])

    res.append(r'\end{tabular*}')
    res = '\n'.join(res)
    return res

class DataTable(list):
    pass
