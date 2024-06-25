#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generic routines for figure generation."""
from __future__ import absolute_import
# from __future__ import unicode_literals  # enum construction fails

import os
from collections import OrderedDict
from operator import itemgetter
from itertools import groupby
import warnings
import numpy as np
from matplotlib import pyplot as plt
import shutil
from six import advance_iterator
# from pdb import set_trace

# absolute_import => . refers to where ppfig resides in the package:
from . import genericsettings, testbedsettings, toolsstats, htmldesc, toolsdivers


# CLASS DEFINITIONS
class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


# FUNCTION DEFINITIONS
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


HtmlPage = enum('NON_SPECIFIED', 'ONE', 'MANY', 'PPRLDMANY_BY_GROUP', 'PPRLDMANY_BY_GROUP_MANY',
                'PPTABLE', 'PPTABLE2', 'PPTABLES', 'PPRLDISTR', 'PPRLDISTR2', 'PPLOGLOSS',
                'PPSCATTER', 'PPFIGS', 'PPFIGCONS', 'PPFIGCONS1')

_figsize_warnings = 1  # couldn't convince filterwarnings('once') to work as desired
'''remaining number of warnings to be issued'''

def save_figure(filename,
                algorithm=None,
                format=None,
                layout_rect=(0, 0, 0.99, 1),
                bbox_inches=None,
                subplots_adjust=None):
    """Save figure into an image file.

    `format` is a `str` denoting a file type known to `pylab.savefig`, like 
    "svg", or `None` in which case the defaults from `genericsettings` are
    applied.
    
    If `layout_rect`, the `pylab.tight_layout` method is invoked with
    matplotlib version < 3.

    `subplots_adjust` contains keyword arguments to call the matplotlib
    function with the same name with matplotlib version >= 3. The function
    grants relative additional space of size bottom, left, 1 - top, and
    1 - right by shrinking the printed axes. It is used to prevent outside
    text being cut away.

    'tight' `bbox_inches` lead possibly to (slightly) different figure
    sizes in each case, which is undesirable.
    """
    if not format:
        fig_formats = genericsettings.figure_file_formats
    else:
        fig_formats = (format, )

    label = toolsdivers.get_version_label(algorithm)
    plt.text(0.5, 0.01, label,
             horizontalalignment="center",
             verticalalignment="bottom",
             fontsize=10,
             color='0.5',
             transform=plt.gca().transAxes)
    for format in fig_formats:
        if plt.matplotlib.__version__[0] >= '3' and subplots_adjust:
            # subplots_adjust is used in pprldmany.main with bottom=0.135, right=0.735
            plt.subplots_adjust(**subplots_adjust)
        elif layout_rect:
            try:
                # possible alternative:
                # bbox = gcf().get_tightbbox(gcf().canvas.get_renderer())
                # bbox._bbox.set_points([[plt.xlim()[0], None], [None, None]])
                #
                # layout_rect[2]=0.88 extends the figure to the
                # right, i.e., 0.88 is where the "tight" right figure
                # border is placed whereas everything is plotted
                # further up to plotted figure border at 1
                plt.tight_layout(pad=0.15, rect=layout_rect)
            except Exception as e:
                warnings.warn(
                    'Figure tightening failed (matplotlib version %s)'
                    ' with Exception: "%s"' %
                    (plt.matplotlib.__version__, str(e)))
        try:
            if plt.rcParams['figure.figsize'] != genericsettings.figsize:
                # prevent saved figure to be different under Jupyter notebooks
                plt.gcf().set_size_inches(genericsettings.figsize)
                global _figsize_warnings
                if _figsize_warnings > 0:
                    m = 'Plotting with genericsettings.figsize=={} instead of default {}'.format(
                            genericsettings.figsize, plt.rcParams['figure.figsize'])
                    warnings.warn(m)
                    _figsize_warnings -= 1
            plt.savefig(filename + '.' + format,
                        dpi=60 if genericsettings.in_a_hurry else 300,
                        format=format,
                        bbox_inches=bbox_inches,
                        # pad_inches=0,  # default is 0.1?, 0 leads to cut label text
                        )
            if genericsettings.verbose:
                print('Wrote figure in %s.' % (filename + '.' + format))
        except IOError:
            warnings.warn('%s is not writeable.' % (filename + '.' + format))

pprldmany_per_func_dim_header = 'Runtime distributions (ECDFs) per function'
pprldmany_per_group_dim_header = 'Runtime distributions (ECDFs) summary and function groups'
convergence_plots_header = 'Convergence plots'

links_placeholder = '<!--links-->'

html_header = """<HTML>
<HEAD>
   <META NAME="description" CONTENT="COCO/BBOB figures by function">
   <META NAME="keywords" CONTENT="COCO, BBOB">
   <META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=iso-8859-1">
   <TITLE> %s </TITLE>
   <SCRIPT SRC="sorttable.js"></SCRIPT>
</HEAD>
<BODY>
<H1> %s
</H1>
%s
"""


def add_image(image_name, add_link_to_image, height=160):
    if add_link_to_image:
        return '<a href="%s"><IMG SRC="%s" height="%dem"></a>' % (image_name, image_name, height)
    else:
        return '<IMG SRC="%s" height="%dem">' % (image_name, height)

def format_link_list_entry_in_html(s):
    """format an html text line by appending the <br> tag (was: by the <H3> tag)"""
    return s + '<br>\n'  # the previous format was '<H3>{}</H3>\n'.format(s)

def add_link(current_dir, folder, file_name, label,
             indent='',
             ignore_file_exists=False,
             dimension=None):
    if folder:
        path = os.path.join(os.path.realpath(current_dir), folder, file_name)
        href = '%s/%s' % (folder, file_name)
    else:
        path = os.path.join(os.path.realpath(current_dir), file_name)
        href = file_name

    if ignore_file_exists or os.path.isfile(path):
        return format_link_list_entry_in_html('{}<a href="{}{}">{}</a>'.format(
            indent, href, "#" + str(dimension) if dimension else "", label))

    return ''


def save_index_html_file(filename):
    with open(filename + '.html', 'w') as f:
        text = ''
        f.write(html_header % tuple(2 * ['COCO Post-Processing Results'] + [text]))

        current_dir = os.path.dirname(os.path.realpath(filename))
        indent = '&nbsp;&nbsp;'

        comparison_links = ''
        many_algorithm_file = '%s.html' % genericsettings.many_algorithm_file_name
        for root, _dirs, files in os.walk(current_dir):
            for elem in _dirs:
                comparison_links += add_link(current_dir, elem, many_algorithm_file, elem, indent)

        if comparison_links:
            f.write('<H2>Comparison data</H2>\n')
            f.write(comparison_links)

        f.write('<H2>Single algorithm data</H2>\n')
        single_algorithm_file = '%s.html' % genericsettings.single_algorithm_file_name
        for root, _dirs, files in os.walk(current_dir):
            for elem in _dirs:
                f.write(add_link(current_dir, elem, single_algorithm_file, elem, indent))

        f.write("\n</BODY>\n</HTML>")


def save_folder_index_file(filename, image_file_extension):

    if not filename:
        return

    current_dir = os.path.dirname(os.path.realpath(filename))

    links = get_home_link()
    links += get_convergence_link(current_dir)
    links += get_rld_link(current_dir)
    links += add_link(current_dir, None, genericsettings.ppfigdim_file_name + '.html',
                      'Scaling with dimension for selected targets')
    links += add_link(current_dir, None, genericsettings.ppfigcons1_file_name + '.html',
                      'Scaling with constraints for selected targets')

    links += add_link(current_dir, None, 'pptable.html', 'Tables for selected targets')
    links += add_link(current_dir, None, 'pprldistr.html',
                      'Runtime distribution for selected targets and f-distributions')
    links += add_link(current_dir, None, 'pplogloss.html', 'Runtime loss ratios')

    image_file_name = 'pprldmany-single-functions/pprldmany.%s' % image_file_extension
    if os.path.isfile(os.path.join(current_dir, image_file_name)):
        links += "<H2> %s </H2>\n" % ' Runtime distributions (ECDFs) over all targets'
        links += add_image(image_file_name, True, 380)

    links += add_link(current_dir, None, genericsettings.ppfigs_file_name + '.html', 'Scaling with dimension')
    links += add_link(current_dir, None, genericsettings.ppfigcons_file_name + '.html', 'Scaling with constraints')
    links += add_link(current_dir, None, genericsettings.pptables_file_name + '.html',
                      'Tables for selected targets')
    links += add_link(current_dir, None, genericsettings.ppscatter_file_name + '.html', 'Scatter plots')
    links += add_link(current_dir, None, genericsettings.pprldistr2_file_name + '.html',
                      'Runtime distribution for selected targets and f-distributions')

    # add the ECDFs aggregated over all functions in all dimensions at the end:
    if os.path.isfile(os.path.join(current_dir,
                                   testbedsettings.current_testbed.plots_on_main_html_page[0])):
        links += "<H2> %s </H2>\n" % ' Runtime distributions (ECDFs) over all targets'
        i = 1 # counter to only put three plots per line
        for plotname in testbedsettings.current_testbed.plots_on_main_html_page:
            if os.path.isfile(os.path.join(current_dir, plotname)):
                if i % 3 == 0:
                    links += add_image(plotname, True, 220) + ' <br />'
                else:
                    links += add_image(plotname, True, 220)
                i = i + 1

    lines = []
    with open(filename) as infile:
        for line in infile:
            lines.append(line)
            if links_placeholder in line:
                lines.append("%s\n</BODY>\n</HTML>" % links)
                break

    with open(filename, 'w') as outfile:
        for line in lines:
            outfile.write(line)

    save_index_html_file(os.path.join(current_dir, '..', genericsettings.index_html_file_name))


def get_home_link():
    home_link = format_link_list_entry_in_html('<a href="%s%s.html">Home</a>')
    return home_link % ('../', genericsettings.index_html_file_name)


def get_convergence_link(current_dir):
    return add_link(current_dir, None, genericsettings.ppconv_file_name + '.html',
                    convergence_plots_header)


def get_rld_link(current_dir):
    links = ''
    folder = 'pprldmany-single-functions'

    file_name = '%s.html' % genericsettings.pprldmany_file_name
    links += add_link(current_dir, folder, file_name,
                      pprldmany_per_func_dim_header, dimension=testbedsettings.current_testbed.goto_dimension)

    file_name = '%s.html' % genericsettings.pprldmany_group_file_name
    links += add_link(current_dir, folder, file_name,
                      pprldmany_per_group_dim_header, dimension=testbedsettings.current_testbed.goto_dimension)

    file_name = '%s.html' % genericsettings.pprldmany_file_name
    links += add_link(current_dir, '', file_name,
                      pprldmany_per_group_dim_header, dimension=testbedsettings.current_testbed.goto_dimension)

    return links


def get_parent_link(html_page, parent_file_name):
    if parent_file_name and html_page not in (HtmlPage.ONE, HtmlPage.MANY):
        return format_link_list_entry_in_html('<a href="{}.html">Overview page</a>'
                                              .format(parent_file_name))
    return ''


def save_single_functions_html(filename,
                               algname='',
                               extension='svg',
                               add_to_names='',
                               dimensions=None,
                               htmlPage=HtmlPage.NON_SPECIFIED,
                               function_groups=None,
                               parentFileName=None,  # used only with HtmlPage.NON_SPECIFIED
                               header=None,  # used only with HtmlPage.NON_SPECIFIED
                               caption=None):  # used only with HtmlPage.NON_SPECIFIED and PPFIGCONS1 (dynamic caption setting)

    name = filename.split(os.sep)[-1]
    current_dir = os.path.dirname(os.path.realpath(filename))
    with open(filename + add_to_names + '.html', 'w') as f:
        header_title = algname + ', ' + name + add_to_names
        links = get_parent_link(htmlPage, parentFileName)

        f.write(html_header % (header_title.lstrip(',').strip(), algname, links))

        if function_groups is None:
            function_groups = OrderedDict([])

        if not testbedsettings.current_testbed.name.startswith("bbob-constrained"):
            function_group = "nzall" if genericsettings.isNoisy else "noiselessall"
            if htmlPage not in (HtmlPage.PPRLDMANY_BY_GROUP, HtmlPage.PPLOGLOSS):
                function_groups.update(
                    OrderedDict([(function_group, 'All functions')]))  # append the noiselesall group at the end

        first_function_number = testbedsettings.current_testbed.first_function_number
        last_function_number = testbedsettings.current_testbed.last_function_number
        caption_string_format = '<p/>\n%s\n<p/><p/>'

        if htmlPage in (HtmlPage.ONE, HtmlPage.MANY):
            f.write(links_placeholder)
        elif htmlPage is HtmlPage.PPSCATTER:
            current_header = 'Scatter plots per function'
            f.write("\n<H2> %s </H2>\n" % current_header)
            for function_number in range(first_function_number, last_function_number + 1):
                f.write(add_image('ppscatter_f%03d%s.%s' % (function_number, add_to_names, extension), True))
            f.write(caption_string_format % '##bbobppscatterlegend##')

        elif htmlPage is HtmlPage.PPFIGS:
            current_header = 'Scaling of run "time" with dimension'
            f.write("\n<H2> %s </H2>\n" % current_header)
            for function_number in range(first_function_number, last_function_number + 1):
                f.write(add_image('ppfigs_f%03d%s.%s' % (function_number, add_to_names, extension), True))
            f.write(caption_string_format % '##bbobppfigslegend##')

        elif htmlPage is HtmlPage.PPFIGCONS:
            current_header = 'Scaling of run "time" with number of constraints'
            f.write("\n<H2> %s </H2>\n" % current_header)
            for function_name in testbedsettings.current_testbed.func_cons_groups.keys():
                for dimension in dimensions if dimensions is not None else [2, 3, 5, 10, 20, 40]:
                    f.write(add_image('ppfigcons_%s_d%s%s.%s' % (function_name, dimension, add_to_names, extension), True))
            f.write(caption_string_format % '##bbobppfigconslegend##')

        elif htmlPage is HtmlPage.PPFIGCONS1:
            current_header = 'Scaling of run "time" with number of constraints for selected targets'
            f.write("\n<H2> %s </H2>\n" % current_header)
            for function_name in testbedsettings.current_testbed.func_cons_groups.keys():
                for dimension in dimensions if dimensions is not None else [2, 3, 5, 10, 20, 40]:
                    f.write(add_image('ppfigcons1_%s_d%s%s.%s' % (function_name, dimension, add_to_names, extension), True))
            # caption given by caption in this case

        elif htmlPage is HtmlPage.NON_SPECIFIED:
            current_header = header
            f.write("\n<H2> %s </H2>\n" % current_header)
            if dimensions is not None:
                for index, dimension in enumerate(dimensions):
                    f.write(write_dimension_links(dimension, dimensions, index))
                    for function_number in range(first_function_number, last_function_number + 1):
                        f.write(add_image('%s_f%03d_%02dD.%s' % (name, function_number, dimension, extension), True))
            else:
                for function_number in range(first_function_number, last_function_number + 1):
                    f.write(add_image('%s_f%03d%s.%s' % (name, function_number, add_to_names, extension), True))
        elif htmlPage is HtmlPage.PPRLDMANY_BY_GROUP:
            current_header = pprldmany_per_group_dim_header
            f.write("\n<H2> %s </H2>\n" % current_header)
            for index, dimension in enumerate(dimensions):
                f.write(write_dimension_links(dimension, dimensions, index))
                for fg in function_groups:
                    f.write(add_image('%s_%s_%02dD.%s' % (name, fg, dimension, extension), True, 200))

        elif htmlPage is HtmlPage.PPRLDMANY_BY_GROUP_MANY:
            current_header = pprldmany_per_group_dim_header
            f.write("\n<H2> %s </H2>\n" % current_header)
            for index, dimension in enumerate(dimensions):
                f.write(write_dimension_links(dimension, dimensions, index))
                for typeKey, typeValue in function_groups.items():
                    f.write(add_image('%s_%02dD_%s.%s' % (name, dimension, typeKey, extension), True))

            f.write(caption_string_format % '\n##bbobECDFslegend##')

        elif htmlPage is HtmlPage.PPTABLE:
            current_header = 'ERT in number of function evaluations'
            f.write("<H2> %s </H2>\n" % current_header)
            for index, dimension in enumerate(dimensions):
                f.write(write_dimension_links(dimension, dimensions, index))
                f.write("\n<!--pptableHtml_%d-->\n" % dimension)
            key = 'bbobpptablecaption' + testbedsettings.current_testbed.scenario
            f.write(caption_string_format % htmldesc.getValue('##' + key + '##'))

        elif htmlPage is HtmlPage.PPTABLES:
            write_tables(f, caption_string_format,
                         testbedsettings.current_testbed.reference_algorithm_filename,
                         'pptablesHtml', 'bbobpptablesmanylegend', dimensions)

        elif htmlPage is HtmlPage.PPRLDISTR:
            names = ['pprldistr', 'ppfvdistr']
            dimensions = testbedsettings.current_testbed.rldDimsOfInterest

            header_ecdf = ' Empirical cumulative distribution functions (ECDF)'
            f.write("<H2> %s </H2>\n" % header_ecdf)
            for dimension in dimensions:
                for typeKey, typeValue in function_groups.items():
                    f.write('<p><b>%s in %d-D</b></p>' % (typeValue, dimension))
                    f.write('<div>')
                    for name in names:
                        f.write(add_image('%s_%02dD_%s.%s' % (name, dimension,
                                                              typeKey, extension), True))
                    f.write('</div>')

            key = 'bbobpprldistrlegend' + testbedsettings.current_testbed.scenario
            f.write(caption_string_format % htmldesc.getValue('##' + key + '##'))

        elif htmlPage is HtmlPage.PPRLDISTR2:
            names = ['pprldistr', 'pplogabs']
            dimensions = testbedsettings.current_testbed.rldDimsOfInterest

            header_ecdf = 'Empirical cumulative distribution functions ' \
                         '(ECDFs) per function group'
            f.write("\n<H2> %s </H2>\n" % header_ecdf)
            for dimension in dimensions:
                for typeKey, typeValue in function_groups.items():
                    f.write('<p><b>%s in %d-D</b></p>' % (typeValue, dimension))
                    f.write('<div>')
                    for name in names:
                        f.write(add_image('%s_%02dD_%s.%s'
                                          % (name, dimension, typeKey, extension),
                                          True))
                    f.write('</div>')

            key = 'bbobpprldistrlegendtwo' + testbedsettings.current_testbed.scenario
            f.write(caption_string_format % htmldesc.getValue('##' + key + '##'))

        elif htmlPage is HtmlPage.PPLOGLOSS:
            dimensions = testbedsettings.current_testbed.rldDimsOfInterest
            if testbedsettings.current_testbed.reference_algorithm_filename:
                current_header = 'ERT loss ratios'
                f.write("<H2> %s </H2>\n" % current_header)

                dimension_list = '-D, '.join(str(x) for x in dimensions) + '-D'
                index = dimension_list.rfind(",")
                dimension_list = dimension_list[:index] + ' and' + dimension_list[index + 1:]

                f.write('<p><b>%s in %s</b></p>' % ('All functions', dimension_list))
                f.write('<div>')
                for dimension in dimensions:
                    f.write(add_image('pplogloss_%02dD_%s.%s' % (dimension, function_group, extension), True))
                f.write('</div>')

                f.write("\n<!--tables-->\n")
                scenario = testbedsettings.current_testbed.scenario
                f.write(caption_string_format % htmldesc.getValue('##bbobloglosstablecaption' + scenario + '##'))

                for typeKey, typeValue in function_groups.items():
                    f.write('<p><b>%s in %s</b></p>' % (typeValue, dimension_list))
                    f.write('<div>')
                    for dimension in dimensions:
                        f.write(add_image('pplogloss_%02dD_%s.%s' % (dimension, typeKey, extension), True))
                    f.write('</div>')

                f.write(caption_string_format % htmldesc.getValue('##bbobloglossfigurecaption' + scenario + '##'))

        if caption:
            f.write(caption_string_format % caption)

        f.write("\n<BR/><BR/><BR/><BR/><BR/>\n</BODY>\n</HTML>")
        
    toolsdivers.replace_in_file(filename + add_to_names + '.html', '??COCOVERSION??', '<br />Data produced with COCO %s' % (toolsdivers.get_version_label(None)))

    if parentFileName:
        save_folder_index_file(os.path.join(current_dir, parentFileName + '.html'), extension)


def write_dimension_links(dimension, dimensions, index):
    links = '<p><A NAME="%d"></A>' % dimension
    if index == 0:
        links += '<A HREF="#%d">Last dimension</A> | ' % dimensions[-1]
    else:
        links += '<A HREF="#%d">Previous dimension</A> | ' % dimensions[index - 1]
    links += '<A HREF="#%d"><b>Dimension = %d</b></A>' % (dimension, dimension)
    if index == len(dimensions) - 1:
        links += ' | <A HREF="#%d">First dimension</A>' % dimensions[0]
    else:
        links += ' | <A HREF="#%d">Next dimension</A>' % dimensions[index + 1]
    links += '</p>\n'

    return links


def write_tables(f, caption_string_format, best_alg_exists, html_key, legend_key, dimensions):
    current_header = 'Table showing the ERT in number of evaluations'
    if best_alg_exists:
        current_header += ' divided by the best ERT of the reference algorithm'

    f.write("\n<H2> %s </H2>\n" % current_header)
    for index, dimension in enumerate(dimensions):
        f.write(write_dimension_links(dimension, dimensions, index))
        f.write("\n<!--%s_%d-->\n" % (html_key, dimension))
    key = legend_key + testbedsettings.current_testbed.scenario
    f.write(caption_string_format % htmldesc.getValue('##' + key + '##'))


def copy_js_files(output_dir):
    """Copies js files to output directory."""

    js_folder = os.path.join(toolsdivers.path_in_package(), 'js')
    for file_in_folder in os.listdir(js_folder):
        if file_in_folder.endswith(".js"):
            shutil.copy(os.path.join(js_folder, file_in_folder), output_dir)


def discretize_limits(limits, smaller_steps_limit=3.1):
    """return new limits with discrete values in k * 10**i with k in [1, 3].

    `limits` has len 2 and the new lower limit is always ``10**-0.2``.

    if `limits[1] / limits[0] < 10**smaller_steps_limits`, k == 3 is an
    additional choice.
    """
    ymin, ymax = limits
    ymin = np.max((ymin, 10 ** -0.2))
    ymax = int(ymax + 1)

    ymax_new = 10 ** np.ceil(np.log10(ymax)) * (1 + 1e-6)
    if 3. * ymax_new / 10 > ymax and np.log10(ymax / ymin) < smaller_steps_limit:
        ymax_new *= 3. / 10
    ymin_new = 10 ** np.floor(np.log10(ymin)) / (1 + 1e-6)
    if 11 < 3 and 3 * ymin_new < ymin and np.log10(ymax / ymin) < 1.1:
        ymin_new *= 3

    if ymin_new < 1.1:
        ymin_new = 10 ** -0.2
    ymin_new = 10 ** -0.2
    return ymin_new, ymax_new


def marker_positions(xdata, ydata, nbperdecade, maxnb,
                     ax_limits=None, y_transformation=None, xmin=1.1):
    """return randomized marker positions

    replacement for downsample, could be improved by becoming independent
    of axis limits?
    """
    if ax_limits is None:  # use current axis limits
        ax_limits = plt.axis()
    tfy = y_transformation
    if tfy is None:
        tfy = lambda x: x  # identity

    xdatarange = np.log10(max([max(xdata), ax_limits[0], ax_limits[1]]) + 0.501) - \
                 np.log10(max([0,  # addresses ax_limits[0] < 0, assumes above max >= 0
                     min([min(xdata), ax_limits[0], ax_limits[1]])]) + 0.5)  # np.log10(xdata[-1]) - np.log10(xdata[0])
    ydatarange = tfy(max([max(ydata), ax_limits[2], ax_limits[3]]) + 0.5) - \
                 tfy(min([min(ydata), ax_limits[2], ax_limits[3]]) + 0.5)  # tfy(ydata[-1]) - tfy(ydata[0])
    nbmarkers = np.min([maxnb, nbperdecade +
                        np.ceil(nbperdecade * (1e-99 + np.abs(np.log10(max(xdata)) - np.log10(min(xdata)))))])
    probs = np.abs(np.diff(np.log10(xdata))) / xdatarange + \
            np.abs(np.diff(tfy(ydata))) / ydatarange
    xpos = []
    ypos = []
    if sum(probs) > 0:
        xoff = np.random.rand() / nbmarkers
        probs /= sum(probs)
        cum = np.cumsum(probs)
        for xact in np.arange(0, 1, 1. / nbmarkers):
            pos = xoff + xact + (1. / nbmarkers) * (0.3 + 0.4 * np.random.rand())
            idx = np.abs(cum - pos).argmin()  # index of closest value
            if xdata[idx] > xmin:
                xpos.append(xdata[idx])
                ypos.append(ydata[idx])
    xpos.append(xdata[-1])
    ypos.append(ydata[-1])
    return xpos, ypos


def plotUnifLogXMarkers(x, y, nbperdecade, logscale=False, **kwargs):
    """Proxy plot function: markers are evenly spaced on the log x-scale

    Remark/TODO: should be called plot_with_unif_markers!? Here is where
    the ECDF plot "done in pprldmany" actually happens.

    This method generates plots with markers regularly spaced on the
    x-scale whereas the matplotlib.pyplot.plot function will put markers
    on data points.

    This method outputs a list of three lines.Line2D objects: the first
    with the line style, the second for the markers and the last for the
    label.

    This function only works with monotonous graph.
    """
    line_args = kwargs.copy()
    line_args['marker'] = ''
    line_args['label'] = ''
    res = plt.plot(x, y, **line_args)  # shouldn't this be done in the calling code?

    if 'marker' in kwargs and len(x) > 0:
        # x2, y2 = downsample(x, y)
        x2, y2 = marker_positions(x, y, nbperdecade, 19, plt.axis(),
                                  np.log10 if logscale else None)
        marker_args = kwargs.copy()
        marker_args['drawstyle'] = 'default'
        marker_args['linestyle' if 'ls' not in marker_args else
                    'ls'] = ''  # 'linestyle' cannot be mixed with 'ls' as used elsewhere
        marker_args['label'] = ''
        res2 = plt.plot(x2, y2, **marker_args)
        res.extend(res2)

    if 'label' in kwargs:
        res3 = plt.plot([-1.], [-1.], **kwargs)
        res.extend(res3)

    return res


def consecutiveNumbers(data, prefix=''):
    """Groups a sequence of integers into ranges of consecutive numbers.
    If the prefix is set then the it's placed before each number.

    Example::
      >>> import os
      >>> import cocopp
      >>> returnpath = os.getcwd()  # needed for no effect on other doctests
      >>> os.chdir(cocopp.toolsdivers.path_in_package())
      >>> cocopp.ppfig.consecutiveNumbers([0, 1, 2, 4, 5, 7, 8, 9])
      '0-2, 4, 5, 7-9'
      >>> cocopp.ppfig.consecutiveNumbers([0, 1, 2, 4, 5, 7, 8, 9], 'f')
      'f0-f2, f4, f5, f7-f9'
      >>> os.chdir(returnpath)  # no effect on path from this doctest

    Range of consecutive numbers is at least 3 (therefore [4, 5] is
    represented as "4, 5").
    """
    res = []
    tmp = groupByRange(data)
    for i in tmp:
        tmpstring = list(prefix + str(j) for j in i)
        if len(i) <= 2:  # This means length of ranges are at least 3
            res.append(', '.join(tmpstring))
        else:
            res.append('-'.join((tmpstring[0], tmpstring[-1])))

    return ', '.join(res)


def groupByRange(data):
    """Groups a sequence of integers into ranges of consecutive numbers.

    Helper function of consecutiveNumbers(data), returns a list of lists.
    The key to the solution is differencing with a range so that
    consecutive numbers all appear in same group.
    Useful for determining ranges of functions.
    Ref: http://docs.python.org/release/3.0.1/library/itertools.html
    """
    res = []
    for _k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
        res.append(list(i for i in map(itemgetter(1), g)))

    return res


def logxticks(limits=[-np.inf, np.inf]):
    """Modify log-scale figure xticks from 10^i to i for values with the
    ``limits`` and (re-)sets the current xlim() thereby turning autoscale
    off (if it was on).

    This is to have xticks that are more visible.
    Modifying the x-limits of the figure after calling this method will
    not update the ticks.
    Please make sure the xlabel is changed accordingly.
    """
    _xticks = plt.xticks()
    xlims = plt.xlim()
    newxticks = []
    for j in _xticks[0]:
        if j > limits[0] and j < limits[1]:  # tick annotations only within the limits
            newxticks.append('%d' % round(np.log10(j)))
        else:
            newxticks.append('')
    plt.xticks(_xticks[0], newxticks)  # this changes the limits (only in newer versions of mpl?)
    plt.xlim(xlims[0], xlims[1])
    # TODO: check the xlabel is changed accordingly?

def beautify():
    """ deprecated method - not used anywhere
        Customize a figure by adding a legend, axis label, etc."""
    # TODO: what is this function for?
    # Input checking

    # Get axis handle and set scale for each axis
    axisHandle = plt.gca()
    axisHandle.set_yscale("log")

    # Grid options
    axisHandle.grid(True)

    _ymin, ymax = plt.ylim()
    plt.ylim(10 ** -0.2, ymax)  # Set back the default maximum.

    tmp = axisHandle.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(np.log10(i)))
    axisHandle.set_yticklabels(tmp2)
    axisHandle.set_ylabel('log10 of ERT')


def generateData(dataSet, targetFuncValue):
    """Returns an array of results to be plotted.

    1st column is ert, 2nd is  the number of success, 3rd the success
    rate, 4th the sum of the number of function evaluations, and
    finally the median on successful runs.
    """
    it = iter(reversed(dataSet.evals))
    i = advance_iterator(it)
    prev = np.array([np.nan] * len(i))

    while i[0] <= targetFuncValue:
        prev = i
        try:
            i = advance_iterator(it)
        except StopIteration:
            break

    data = prev[1:].copy()  # keep only the number of function evaluations.
    # was up to rev4997: succ = (np.isnan(data) == False)  # better: ~np.isnan(data)
    succ = np.isfinite(data)
    if succ.any():
        med = toolsstats.prctile(data[succ], 50)[0]
        # Line above was modified at rev 3050 to make sure that we consider only
        # successful trials in the median
    else:
        med = np.nan

    # prepare to compute runlengths / ERT with restarts (AKA SP1)
    data[np.isnan(data)] = dataSet.maxevals[np.isnan(data)]

    res = []
    res.extend(toolsstats.sp(data, issuccessful=succ, allowinf=False))
    res.append(np.mean(data))  # mean(FE)
    res.append(med)

    return np.array(res)


def plot(dsList, _valuesOfInterest=(10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8),
         isbyinstance=True, kwargs={}):
    """From a DataSetList, plot a graph. Not in use and superseeded by ppfigdim.main!?"""

    # set_trace()
    res = []

    valuesOfInterest = list(_valuesOfInterest)
    valuesOfInterest.sort(reverse=True)

    def transform(dsList):
        """Create dictionary of instances."""

        class StrippedUpDS():
            """Data Set stripped up of everything."""

            pass

        res = {}
        for i in dsList:
            dictinstance = i.createDictInstance()
            for j, idx in sorted(list(dictinstance.items())):
                tmp = StrippedUpDS()
                idxs = list(k + 1 for k in idx)
                idxs.insert(0, 0)
                tmp._evals = i._evals[:, np.r_[idxs]].copy()
                tmp.maxevals = i.maxevals[np.ix_(idx)].copy()
                res.setdefault(j, [])
                res.get(j).append(tmp)
        return res

    for i in range(len(valuesOfInterest)):

        succ = []
        unsucc = []
        displaynumber = []

        dictX = transform(dsList)
        for x in sorted(dictX.keys()):
            dsListByX = dictX[x]
            for j in dsListByX:
                tmp = generateData(j, valuesOfInterest[i])
                if tmp[2] > 0:  # Number of success is larger than 0
                    succ.append(np.append(x, tmp))
                    if tmp[2] < j.nbRuns():
                        displaynumber.append((x, tmp[0], tmp[2]))
                else:
                    unsucc.append(np.append(x, tmp))

        if succ:
            tmp = np.vstack(succ)
            # ERT
            res.extend(plt.plot(tmp[:, 0], tmp[:, 1], **kwargs))
            # median
            tmp2 = plt.plot(tmp[:, 0], tmp[:, -1], **kwargs)
            plt.setp(tmp2, linestyle='', marker='+', markersize=30, markeredgewidth=5)
            # , color=colors[i], linestyle='', marker='+', markersize=30, markeredgewidth=5))
            res.extend(tmp2)

        # To have the legend displayed whatever happens with the data.
        tmp = plt.plot([], [], **kwargs)
        plt.setp(tmp, label=' %+d' % (np.log10(valuesOfInterest[i])))
        res.extend(tmp)

        # Only for the last target function value
        if unsucc:
            tmp = np.vstack(unsucc)  # tmp[:, 0] needs to be sorted!
            res.extend(plt.plot(tmp[:, 0], tmp[:, 1], **kwargs))

    if displaynumber:  # displayed only for the smallest valuesOfInterest
        for j in displaynumber:
            t = plt.text(j[0], j[1] * 1.85, "%.0f" % j[2],
                         horizontalalignment="center",
                         verticalalignment="bottom")
            res.append(t)

    return res


def get_first_html_file(current_dir, prefix):
    filename_list = get_sorted_html_files(current_dir, prefix)
    if filename_list:
        return filename_list[0][0]

    return None


def get_sorted_html_files(current_dir, prefix):

    suffix = 'D.html'
    prefix += '_'

    filename_dict = {}
    for (dir_path, dir_names, file_names) in os.walk(current_dir):
        for filename in file_names:
            if filename.startswith(prefix) and filename.endswith(suffix):
                stripped_filename = filename.replace(prefix, '').replace(suffix, '')
                if stripped_filename.isdigit():
                    key = int(stripped_filename)
                    filename_dict[key] = filename
        break

    pair_list = []
    firstFile = None
    previousFile = None
    for key, filename in sorted(filename_dict.items()):
        if not firstFile:
            firstFile = filename

        if previousFile:
            pair_list.append([previousFile, filename])
        previousFile = filename

    if firstFile and previousFile:
        pair_list.append([previousFile, firstFile])

    return pair_list


class PlottingStyle(object):

    def __init__(self, pprldmany_styles, ppfigs_styles, algorithm_list, in_background):
        self.pprldmany_styles = pprldmany_styles
        self.ppfigs_styles = ppfigs_styles
        self.algorithm_list = algorithm_list
        self.in_background = in_background


def get_plotting_styles(algorithms, only_foreground=False):

    plotting_styles = []

    if not only_foreground:
        for format, pathnames in genericsettings.background.items():
            assert isinstance(pathnames, (list, tuple, set))
            if format is None:
                format = genericsettings.background_default_style
            background_algorithms = [algorithm for algorithm in algorithms
                                     if algorithm in pathnames]
            background_algorithms.sort()
            if len(background_algorithms) > 0:
                ppfigs_styles = {'marker': '',
                                 'color': format[0],
                                 'linestyle': format[1],
                                 'zorder': -1.5,
                                 }
                pprldmany_styles = {'marker': '',
                                    'label': '',
                                    'color': format[0],
                                    'linestyle': format[1],
                                    'zorder': -1.5,
                                    }
                plotting_styles.append(PlottingStyle(pprldmany_styles, ppfigs_styles, background_algorithms, True))

    foreground_algorithms = [key for key in algorithms
                             if key in genericsettings.foreground_algorithm_list]
    # foreground_algorithms.sort()  # sorting is not desired, we want to be able to control the order!
    plotting_styles.append(PlottingStyle({},
                                         {},
                                         foreground_algorithms if len(foreground_algorithms) > 0 else algorithms,
                                         False))

    return plotting_styles

def getFontSize(nameList):
    maxFuncLength = max(len(i) for i in nameList)
    fontSize = 24 - max(0, 2 * ((maxFuncLength - 35) / 5))
    return fontSize
