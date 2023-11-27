#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Various tools. 

"""
from __future__ import absolute_import, print_function

import os, time, warnings
import tempfile, shutil
from collections import OrderedDict as _OrderedDict
import re as _re
import numpy as np
from matplotlib import pyplot as plt
from subprocess import CalledProcessError, STDOUT
import pkg_resources

from . import genericsettings, testbedsettings

class Infolder(object):
    """Contextmanager to do some work in a folder of choice and change dir
    back in the end.

    Usage:

    >>> import os
    >>> import cocopp.toolsdivers
    >>> dir_ = os.getcwd()  # for the record
    >>> with cocopp.toolsdivers.Infolder('..'):
    ...     # do some work in a folder here, e.g. open a file
    ...     len(dir_) > len(os.getcwd()) and os.getcwd() in dir_
    True
    >>> # magically we are back in the original folder
    >>> assert dir_ == os.getcwd()

    """
    def __init__(self, foldername):
        self.target_dir = foldername
    def __enter__(self):
        self.root_dir = os.getcwd()
        os.chdir(self.target_dir)
    def __exit__(self, *args):
        os.chdir(self.root_dir)


class StringList(list):
    """A microtool to join a list of strings using property `as_string`.

    `StringList` can also be initialized with a string.

    >>> from cocopp.toolsdivers import StringList
    >>> StringList('ab bc') == ['ab', 'bc']
    True
    >>> word_list = StringList(['this', 'has', 'a', 'leading', 'and',
    ...                         'trailing', 'space'])
    >>> word_list.as_string
    ' this has a leading and trailing space '

    `as_string` is less typing than

    >>> ' ' + ' '.join(word_list) + ' ' == word_list.as_string
    True

    and provides tab completion.

    """
    def __init__(self, list_or_str):
        try:
            inlist = list_or_str.split()
        except AttributeError:
            inlist = list_or_str
        if inlist:  # prevent error on None
            list.__init__(self, inlist)

    @property
    def as_string(self):
        """return concatenation with spaces between"""
        return ' ' + ' '.join(self) + ' '


class InfolderGoneWithTheWind:
    """``with InfolderGoneWithTheWind(): ...`` executes the block in a

    temporary folder under the current folder. The temporary folder is
    deleted on exiting the block.

    >>> import os
    >>> dir_ = os.getcwd()  # for the record
    >>> len_ = len(os.listdir('.'))
    >>> with InfolderGoneWithTheWind():  # doctest: +SKIP
    ...     # do some work in a folder here, e.g. write files
    ...     len(dir_) > len(os.getcwd()) and os.getcwd() in dir_
    True
    >>> # magically we are back in the original folder
    >>> assert dir_ == os.getcwd()
    >>> assert len(os.listdir('.')) == len_

    """
    def __init__(self, prefix='_'):
        """no folder needs to be given"""
        self.prefix = prefix
    def __enter__(self):
        self.root_dir = os.getcwd()
        # self.target_dir = tempfile.mkdtemp(prefix=self.prefix, dir='.')
        self.target_dir = tempfile.mkdtemp(prefix=self.prefix)
        self._target_dir = self.target_dir
        os.chdir(self.target_dir)
    def __exit__(self, *args):
        os.chdir(self.root_dir)
        if self.target_dir == self._target_dir:
            shutil.rmtree(self.target_dir)
        else:
            raise ValueError("inconsistent temporary folder name %s vs %s"
                             % (self._target_dir, self.target_dir))


class StrList(list):
    """A list of `str` with search/find functionality.

    """
    def __init__(self, list_or_str):
        try:
            inlist = list_or_str.split()
        except AttributeError:
            inlist = list_or_str
        if inlist:  # prevent failing on None
            list.__init__(self, inlist)
        self._names_found = []

    @property
    def as_string(self):
        """return space separated string concatenation surrounded by spaces.
        
        To get only the recently found items use ``found.as_string``
        instead of ``as_string``.
        """
        return ' ' + ' '.join(self) + ' '

    @property
    def found(self):
        """`StrList` of elements found during the last call to `find`.
        """
        return StrList(self._names_found)

    def __call__(self, *substrs):
        """alias to `find`"""
        return self.find(*substrs)

    def find(self, *substrs):
        """return entries that match all `substrs`.

        This method serves for interactive exploration of available entries
        and may be aliased to the shortcut of calling the instance itself.

        When given several `substrs` arguments the results match each
        substring (AND search, an OR can be simply achieved by appending
        the result of two finds). Upper/lower case is ignored.

        When given a single `substrs` argument, it may be

        - a list of matching substrings, used as several substrings as above
        - an index of `type` `int`
        - a list of indices

        A single substring matches either if an entry contains the
        substring or if the substring matches as regular expression, where
        "." matches any single character and ".*" matches any number >= 0
        of characters.

        >>> from cocopp.toolsdivers import StrList
        >>> s = StrList(['abc', 'bcd', 'cde', ' cde'])
        >>> s('bc')  # all strings with a 'bc'
        ['abc', 'bcd']
        >>> s('a', 'b')  # all strings with an 'a' AND 'b'
        ['abc']
        >>> s(['a', 'b'])  # the same
        ['abc']
        >>> s('.c')  # regex 'c' as second char
        ['bcd', ' cde']
        >>> s('.*c')  # regex 'c' preceded with any sequence
        ['abc', 'bcd', 'cde', ' cde']

        Details: The list of matching names is stored in `found`.
        """
        # check whether the first arg is a list rather than a str
        if substrs and len(substrs) == 1 and substrs[0] != str(substrs[0]):
            substrs = substrs[0]  # we may now have a list of str as expected
            if isinstance(substrs, int):  # or maybe just an int
                self._names_found = [self[substrs]]
                return StrList(self._names_found)
            elif substrs and isinstance(substrs[0], int):  # or a list of indices
                self._names_found = [self[i] for i in substrs]
                return StrList(self._names_found)
        names = list(self)
        for s in substrs:
            rex = _re.compile(s, _re.IGNORECASE)
            try:
                names = [name for name in names if rex.match(name) or s.lower() in name.lower()]
            except AttributeError:
                warnings.warn("arguments to `find` must be strings or a "
                              "single integer or an integer list")
                raise
        self._names_found = names
        return StrList(names)

    def find_indices(self, *substrs):
        """same as `find` but returns indices instead of names"""
        return [self.index(name) for name in self.find(*substrs)]

    def print(self, *substrs):
        """print the result of ``find(*substrs)`` with indices.

        Details: does not change `found` and returns `None`.
        """
        current_names = list(self._names_found)
        for index in self.find_indices(*substrs):
            print("%4d: '%s'" % (index, self[index]))
        self._names_found = current_names


class AlgorithmList(list):
    """Not in use. Not necessary when the algorithm dict is an `OrderedDict` anyway.

    A `list` representing the algorithm name arguments in original order.

    The method `ordered_dict` allows to transform an algorithm `dict` into an
    `OrderedDict` using the order in self.

    >>> from cocopp.toolsdivers import AlgorithmList
    >>> l = ['b', 'a', 'c']
    >>> al = AlgorithmList(l)
    >>> d = dict(zip(l[-1::-1], [1, 2, 3]))
    >>> for i, name in enumerate(al.ordered_dict(d)):
    ...     assert name == l[i]

    """
    def ordered_dict(self, algorithms_dict):
        """return algorithms_dict as `OrderedDict` in order of self.

        Keys that are not in self are sorted using `sorted`.
        """
        if set(algorithms_dict) != set(self):
            warnings.warn("keys in algorithm dict: \n%s\n"
                          "do not agree with original algorithm list: \n%s\n"
                                 % (str(algorithms_dict.keys()), str(self)))
        res = _OrderedDict()
        for name in self:
            if name in algorithms_dict:
                res[name] = algorithms_dict[name]
        for name in sorted(algorithms_dict):
            if name not in res:
                res[name] = algorithms_dict[name]
        assert res == algorithms_dict  # compares keys and values
        return res


class DataWithFewSuccesses:
    """The `result` property is a `OrderedDict` with all ``(dimension, funcId)``-

    tuples that have less than ``self.minsuccesses`` successes. These
    tuples are the `dict` keys. The `dict` values are the respective
    numbers for ``[successes, trials]``. The minimal number of desired
    successes can be changed at any time by re-assigning the `minsuccesses`
    attribute in which case the return value of `result` may change.
    
    Usage concept example::

        >> run_more = DataWithFewSuccesses('folder_or_file_name_for_cocopp.load').result
        >> for p in cocoex.Suite('bbob', '', ''):
        ..     if (p.id_function, p.dimension) not in run_more:
        ..         continue
        ..     p.observe_with(...)
        ..     # run solver on problem p
        ..     [...]

    """
    @property
    def result(self):
        """depends on attributes `minsuccesses` and `successes`"""
        return _OrderedDict(sorted(
            ((ds.funcId, ds.dim), [s, len(ds.instancenumbers)])
                for s, ds in zip(self.successes, self.dsl)
                if s < self.minsuccesses))
    def __init__(self, folder_name, minsuccesses=9, success_threshold=1e-8):
        """`folder_name` can also be a filename or a `DataSetList`"""
        self.minsuccesses = minsuccesses
        self.success_threshold = success_threshold
        if isinstance(folder_name, list):
            self.dsl = folder_name
        else:
            import cocopp
            cocopp.genericsettings.balance_instances = False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.dsl = cocopp.load(folder_name)
        self.trials = [len(ds.instancenumbers) for ds in self.dsl]
        """number of trials in each data set, for the record only"""
        self.successes = self.compute_successes().successes  # declarative assignment
        """list of successful trials, depends on `success_threshold`.
           Can be recomputed by calling `compute_successes`.
           """
    def compute_successes(self, success_threshold=None):
        """Assign `successes` attribute as a `list` of number of successful trials

        in the data sets of `self.dsl` and return `self`.
        """
        self.successes = [ds.detSuccesses([success_threshold if success_threshold is not None 
                                           else self.success_threshold])[0] for ds in self.dsl]
        return self
    def print(self):
        """return a `str` with the number of data sets with too few successes"""
        return 'DataWithFewSuccesses: {}/{}'.format(len(self.result), len(self.dsl))
    def __len__(self):
        return len(self.result)


def print_done(message='  done'):
    """prints a message with time stamp"""
    print(message, '(' + time.asctime() + ').')

def equals_approximately(a, b, eps=1e-12):
    if a < 0:
        a, b = -1 * a, -1 * b
    return a - eps < b < a + eps or (1 - eps) * a < b < (1 + eps) * a

def less(a, b):
    """return a < b, while comparing nan results in False without warning"""
    current_err_setting = np.geterr()
    np.seterr(invalid='ignore')
    res = a < b
    np.seterr(**current_err_setting)
    return res

def diff_attr(m1, m2, exclude=('_', )):
    """return `list` of ``[name, val1, val2]`` triplets for
    attributes with different values.

    Attributes whose names start with any string from the
    `exclude` list are skipped. Furthermore, only attributes
    present in both `m1` and `m2` are compared.

    This function was introduced to compare the `genericsettings`
    module with its state directly after import. It should be
    applicable any other two class instances as well.

    Details: to "find" the attributes, `m1.__dict__` is iterated over. 
    """
    return [[key, getattr(m1, key), getattr(m2, key)]
            for key in m1.__dict__
                if hasattr(m2, key) and
                    not any(key.startswith(s) for s in exclude)
                    and np.all(getattr(m1, key) != getattr(m2, key))]

def prepend_to_file(filename, lines, maxlines=1000, warn_message=None):
    """"prepend lines the tex-command filename """
    try:
        with open(filename, 'r') as f:
            lines_to_append = list(f)
    except IOError:
        lines_to_append = []
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')
        for i, line in enumerate(lines_to_append):
            f.write(line)
            if i > maxlines:
                print(warn_message)
                break
        
def replace_in_file(filename, old_text, new_text):
    """"replace a string in the file with another string"""

    lines = []    
    try:
        with open(filename, 'r') as f:
            lines = list(f)
    except IOError:
        print('File %s does not exist.' % filename)
    
    if lines:    
        with open(filename, 'w') as f:
            for line in lines:
                f.write(line.replace(old_text, new_text))
        
def truncate_latex_command_file(filename, keeplines=200):
    """truncate file but keep in good latex shape"""
    open(filename, 'a').close()
    with open(filename, 'r') as f:
        lines = list(f)
    with open(filename, 'w') as f:
        for i, line in enumerate(lines):
            if i > keeplines and line.startswith(r'\providecommand'):
                break
            f.write(line)
    
def strip_pathname(name):
    """remove ../ and ./ and leading/trailing blanks and path separators
    from input string ``name`` and replace any remaining path separator
    with '/'"""
    return name.replace('..' + os.sep, '').replace('.' + os.sep, '').strip().strip(os.sep).replace(os.sep, '/')

def strip_pathname1(name):
    """remove ../ and ./ and leading/trailing blanks and path separators
    from input string ``name``, replace any remaining path separator
    with '/', and keep only the last part of the path"""
    return (name.replace('..' + os.sep, '').replace('.' + os.sep, '').strip().strip(os.sep).split(os.sep)[-1]).replace('data', '').replace('Data', '').replace('DATA', '').replace('.tar.gz', '').replace('.tgz', '').replace('.tar', '').replace(genericsettings.extraction_folder_prefix, '').strip(os.sep).replace(os.sep, '/')

def strip_pathname2(name):
    """as `strip_pathname1` but keep the last two parts of the path"""
    return os.sep.join(name.replace('..' + os.sep, '').replace('.' + os.sep, '').strip().strip(os.sep).split(os.sep)[-2:]).replace('data', '').replace('Data', '').replace('DATA', '').replace('.tar.gz', '').replace('.tgz', '').replace('.tar', '').strip(os.sep).replace(os.sep, '/')

def strip_pathname3(name):
    """as `strip_pathname1` and also remove `'noiseless'` from the name"""
    return strip_pathname1(name).replace('noiseless', '')

def str_to_latex(string):
    """do replacements in ``string`` such that it most likely compiles with latex """
    #return string.replace('\\', r'\textbackslash{}').replace('_', '\\_').replace(r'^', r'\^\,').replace(r'%', r'\%').replace(r'~', r'\ensuremath{\sim}').replace(r'#', r'\#')
    return string.replace('\\', r'\textbackslash{}').replace('_', ' ').replace(r'^', r'\^\,').replace(r'%', r'\%').replace(r'~', r'\ensuremath{\sim}').replace(r'#', r'\#')


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
     
    >>> from cocopp import toolsdivers as td
    >>> print([td.num2str(val) for val in [12345, 1234.5, 123.45, 12.345, 1.2345, .12345, .012345, .0012345]])
    ['12345', '1234', '123', '12', '1.2', '0.12', '0.012', '1.2e-3']
    
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
        # print(val, val_rounded, s, len(s), idx1, idx2)
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
    
    # print(s_float, s_exp)
    
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

def number_to_html(number_as_string):
    """usage as ``number_to_html(num2str(1.023e-12)) == "'-1.0 x 10<sup>-12</sup>'"``"""
    s = number_as_string
    if s.find('e') > 0:
        if s.startswith('1e') or s.startswith('-1e'):
            s = s.replace('1e', '10<sup>')
        else:
            s = s.replace('e', ' x 10<sup>')
        s += '</sup>'
    return s

def legend(*args, **kwargs):
   kwargs.setdefault('framealpha', 0.2)
   try:
      plt.legend(*args, **kwargs)
   except:
      warnings.warn("framealpha not effective")
      kwargs.pop('framealpha')
      plt.legend(*args, **kwargs)
      
try:
    from subprocess import check_output
except ImportError:
    import subprocess
    def check_output(*popenargs, **kwargs):
        r"""Run command with arguments and return its output as a byte string.
        Backported from Python 2.7 as it's implemented as pure python on stdlib.

        WARNING: This method is also defined in ../../code-experiments/tools/cocoutils.py.
        If you change something you have to change it in both files.
        """
        process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            error = subprocess.CalledProcessError(retcode, cmd)
            error.output = output
            raise error
        return output

def git(args):
    """Run a git command and return its output.

    All errors are deemed fatal and the system will quit.

    WARNING: This method is also defined in ../../code-experiments/tools/cocoutils.py.
    If you change something you have to change it in both files.
    """
    full_command = ['git']
    full_command.extend(args)
    try:
        output = check_output(full_command, env=os.environ,
                              stderr=STDOUT, universal_newlines=True)
        output = output.rstrip()
    except CalledProcessError as e:
        # print('Failed to execute "%s"' % str(full_command))
        raise
    return output

def get_version_label(algorithmID=None):
    """ Returns a string with the COCO version of the installed postprocessing,
        potentially adding the hash of the hypervolume reference values from
        the actual experiments (in the `bbob-biobj` setting).
        If algorithmID==None, the set of different hypervolume reference values
        from all algorithms, read in by the postprocessing, are returned in
        the string. If more than one reference value is present in the data,
        the string displays also a warning.
    """
    coco_version = pkg_resources.require('cocopp')[0].version
    reference_values = testbedsettings.get_reference_values(algorithmID)
    
    if reference_values and type(reference_values) is set:        
        label = "v%s, hv-hashes inconsistent:" % (coco_version)
        for r in reference_values:
            label = label + " %s and" % (r)
        label = label[:-3] + "found!"
    else:
        label = "v%s" % (coco_version) if reference_values is None else "v%s, hv-hash=%s" % (coco_version, reference_values)      
    return label


def path_in_package(sub_path=""):
    """return the absolute path prepended to `subpath` in this module.
    """
    egg_info = pkg_resources.require('cocopp')[0]
    return os.path.join(egg_info.location, egg_info.project_name, sub_path)
