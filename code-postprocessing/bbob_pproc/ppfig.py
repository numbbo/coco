#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generic routines for figure generation."""
from __future__ import absolute_import
import os
from collections import OrderedDict
from operator import itemgetter
from itertools import groupby
import warnings
import numpy as np
from matplotlib import pyplot as plt
import shutil
# from pdb import set_trace
from . import genericsettings, toolsstats, htmldesc, toolsdivers  # absolute_import => . refers to where ppfig resides in the package


bbox_inches_choices = {  # do we also need pad_inches = 0?
    'svg': 'tight',
}


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)
    
HtmlPage = enum('NON_SPECIFIED', 'ONE', 'TWO', 'MANY', 'PPRLDMANY_BY_GROUP', 'PPTABLE', 'PPRLDISTR', 'PPLOGLOSS')

def saveFigure(filename, figFormat=(), verbose=True):
    """Save figure into an image file.

    `figFormat` can be a string or a list of strings, like
    ``('pdf', 'svg')``

    """
    plt.text(0.35, 0.01, toolsdivers.getGitVersion(True), 
             horizontalalignment = "left",
             verticalalignment = "bottom", 
             fontsize = 10,
             color = '0.5',
             transform=plt.gca().transAxes)

    if not figFormat:    
        figFormat=genericsettings.getFigFormats()    
        
    if isinstance(figFormat, basestring):
        figFormat = (figFormat, )
    for format in figFormat:
        try:
            plt.savefig(filename + '.' + format,
                        dpi = 60 if genericsettings.in_a_hurry else 300,
                        format=format,
                        bbox_inches=bbox_inches_choices.get(format, None)
            )
            if verbose:
                print 'Wrote figure in %s.' %(filename + '.' + format)
        except IOError:
            warnings.warn('%s is not writeable.' % (filename + '.' + format))

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

def next_dimension_str(s):
    try:
        dim = int(s.strip().strip('_').rstrip('D'))
        return s.replace('%02d' % dim, '%02d' % next_dimension(dim))
    except:
        warnings.warn('next_dimension_str failed on "%s"' % s)
        print(s)
        raise


def next_dimension(dim):
    """next dimension when clicking single function html pages"""
    if dim == 2:
        return 3
    if dim == 3:
        return 5
    if dim == 40:
        return 2
    return 2 * dim

def addImage(imageName, addLink):
    if (addLink):
        return '<a href="file:%s"><IMG SRC="%s"></a>' % (2 * (imageName,))
    else:
        return '<IMG SRC="%s">' % imageName

def addLink(currentDir, folder, fileName, label, indent = '', ignoreFileExists = False):

    if folder:    
        path = os.path.join(os.path.realpath(currentDir), folder, fileName)
        href = '%s/%s' % (folder, fileName)
    else:
        path = os.path.join(os.path.realpath(currentDir), fileName)
        href = fileName

    if ignoreFileExists or os.path.isfile(path):
        return '<H3>%s<a href="%s">%s</a></H3>\n' % (indent, href, label)

    return ''

def save_index_html_file(filename):

    with open(filename + '.html', 'w') as f:
        f.write(html_header % ('Post processing results', 'Post processing results', ''))
            
        f.write('<H2>Single algorithm data</H2>\n')

        currentDir = os.path.dirname(os.path.realpath(filename))
        indent = '&nbsp;&nbsp;'
        singleAlgFile = 'templateBBOBarticle.html'
        for root, _dirs, files in os.walk(currentDir):
            for elem in _dirs:
                f.write(addLink(currentDir, elem, singleAlgFile, elem, indent))
        
        comparisonLinks = ''    
        comparisonLinks += addLink(currentDir, None, 'templateBBOBcmp.html', 'Two algorithm comparison', indent)
        comparisonLinks += addLink(currentDir, None, 'templateBBOBmany.html', 'Many algorithm comparison', indent)
        if comparisonLinks:
            f.write('<H2>Comparison data</H2>\n')
            f.write(comparisonLinks)

        f.write("\n</BODY>\n</HTML>")

def getHomeLink(htmlPage):
    homeLink = '<H3><a href="%s%s.html">Home</a></H3>'    
    if htmlPage is HtmlPage.ONE:
        return homeLink % ('../', genericsettings.index_html_file_name)
    elif htmlPage is HtmlPage.TWO or htmlPage is HtmlPage.MANY:
        return homeLink % ('', genericsettings.index_html_file_name)
    
    return ''

def getConvLink(htmlPage, currentDir):
    if htmlPage in (HtmlPage.ONE, HtmlPage.TWO, HtmlPage.MANY):
        return addLink(currentDir, None, genericsettings.ppconv_file_name + '.html',
                       'Convergence plots', ignoreFileExists=genericsettings.isConv)
    
    return ''
    
def getRldLink(htmlPage, currentDir):
    
    links = ''        
    folder = 'pprldmany-single-functions'

    ignoreFileExists = genericsettings.isRldOnSingleFcts
    if htmlPage in (HtmlPage.ONE, HtmlPage.TWO, HtmlPage.MANY):
        fileName = '%s.html' % genericsettings.pprldmany_file_name
        links += addLink(currentDir, folder, fileName, 'Runtime distribution plots',
                         ignoreFileExists=ignoreFileExists)
        fileName = '%s_02D.html' % genericsettings.pprldmany_file_name
        links += addLink(currentDir, folder, fileName, 'Runtime distribution plots (per dimension)',
                         ignoreFileExists=ignoreFileExists)
        fileName = '%s_02D.html' % genericsettings.pprldmany_group_file_name
        links += addLink(currentDir, folder, fileName, 'Runtime distribution plots by group (per dimension)',
                         ignoreFileExists=ignoreFileExists)
    
    return links

def getParentLink(htmlPage, parentFileName):
    if parentFileName and htmlPage not in (HtmlPage.ONE, HtmlPage.TWO, HtmlPage.MANY):
        return '<H3><a href="%s.html">Overview page</a></H3>' % parentFileName
    
    return ''

def save_single_functions_html(filename, 
                               algname='', 
                               extension='svg',
                               add_to_names = '', 
                               htmlPage = HtmlPage.NON_SPECIFIED,
                               values_of_interest = [],
                               isBiobjective = False, 
                               functionGroups = None,
                               parentFileName = None, # used only with HtmlPage.NON_SPECIFIED
                               header = None, # used only with HtmlPage.NON_SPECIFIED
                               caption = None): # used only with HtmlPage.NON_SPECIFIED
    
    name = filename.split(os.sep)[-1]
    currentDir = os.path.dirname(os.path.realpath(filename))
    with open(filename + add_to_names + '.html', 'w') as f:
        header_title = algname + ' ' + name + add_to_names
        links = getHomeLink(htmlPage)
        links += getConvLink(htmlPage, currentDir)
        links += getRldLink(htmlPage, currentDir)
        links += getParentLink(htmlPage, parentFileName)

        f.write(html_header % (header_title.strip().replace(' ', ', '), algname, links))
            
        if functionGroups is None:
            functionGroups = OrderedDict([])
        
        if not htmlPage == HtmlPage.PPRLDMANY_BY_GROUP:
            functionGroups.update({'noiselessall':'All functions'})

        maxFunctionIndex = genericsettings.current_testbed.number_of_functions
        captionStringFormat = '<p/>\n%s\n<p/><p/>'
        addLinkForNextDim = add_to_names.endswith('D')
        bestAlgExists = not isBiobjective
        
        if htmlPage is HtmlPage.ONE:
            f.write('<H3><a href="ppfigdim.html">Average runtime versus dimension for selected targets</a></H3>\n')
            f.write('<H3><a href="pptable.html">Average runtime for selected targets</a></H3>\n')
            f.write('<H3><a href="pprldistr.html">Runtime for selected targets and f-distributions</a></H3>\n')
            if not isBiobjective:            
                f.write('<H3><a href="pplogloss.html">Runtime loss ratios</a></H3>\n')

            headerECDF = ' Runtime distributions (ECDF) over all targets'
            f.write("<H2> %s </H2>\n" % headerECDF)
            f.write(addImage('pprldmany.%s' % (extension), True))            

        elif htmlPage is HtmlPage.TWO:
            currentHeader = 'Scaling of aRT with dimension'
            f.write("\n<H2> %s </H2>\n" % currentHeader)
            for ifun in range(1, maxFunctionIndex + 1):
                f.write(addImage('ppfigs_f%03d%s.%s' % (ifun, add_to_names, extension), True))
            f.write(captionStringFormat % '##bbobppfigslegend##')
        
            currentHeader = 'Scatter plots per function'
            f.write("\n<H2> %s </H2>\n" % currentHeader)
            if addLinkForNextDim:
                name_for_click = next_dimension_str(add_to_names)
                f.write('<A HREF="%s">\n' % (filename.split(os.sep)[-1] + name_for_click  + '.html'))
            for ifun in range(1, maxFunctionIndex + 1):
                f.write(addImage('ppscatter_f%03d%s.%s' % (ifun, add_to_names, extension), not addLinkForNextDim))
            if addLinkForNextDim:
                f.write('"\n</A>\n')
    
            f.write(captionStringFormat % '##bbobppscatterlegend##')

            names = ['pprldistr', 'pplogabs']
            dimensions = [5, 20]

            headerECDF = 'Empirical cumulative distribution functions (ECDFs) per function group'
            f.write("\n<H2> %s </H2>\n" % headerECDF)
            for dimension in dimensions:
                for typeKey, typeValue in functionGroups.iteritems():
                    f.write('<p><b>%s in %d-D</b></p>' % (typeValue, dimension))
                    f.write('<div>')
                    for name in names:
                        f.write(addImage('%s_%02dD_%s.%s' % (name, dimension, typeKey, extension), True))
                    f.write('</div>')

            key = 'bbobpprldistrlegendtwo' + genericsettings.current_testbed.scenario
            f.write(captionStringFormat % htmldesc.getValue('##' + key + '##'))

            currentHeader = 'Table showing the aRT in number of function evaluations'
            if bestAlgExists:
                currentHeader += ' divided by the best aRT measured during BBOB-2009'
                
            f.write("\n<H2> %s </H2>\n" % currentHeader)
            f.write("\n<!--pptable2Html-->\n")
            f.write(captionStringFormat % '##bbobpptablestwolegend##')
            
        elif htmlPage is HtmlPage.MANY:
            currentHeader = 'Scaling of aRT with dimension'
            f.write("\n<H2> %s </H2>\n" % currentHeader)
            if addLinkForNextDim:
                name_for_click = next_dimension_str(add_to_names)
                f.write('<A HREF="%s">\n' % (filename.split(os.sep)[-1] + name_for_click  + '.html'))
            for ifun in range(1, maxFunctionIndex + 1):
                f.write(addImage('ppfigs_f%03d%s.%s' % (ifun, add_to_names, extension), not addLinkForNextDim))
            if addLinkForNextDim:
                f.write('"\n</A>\n')
            
            f.write(captionStringFormat % '##bbobppfigslegend##')

            write_ECDF(f, 5, extension, captionStringFormat, functionGroups)
            write_ECDF(f, 20, extension, captionStringFormat, functionGroups)
                
            write_pptables(f, 5, captionStringFormat, maxFunctionIndex, bestAlgExists)
            write_pptables(f, 20, captionStringFormat, maxFunctionIndex, bestAlgExists)

        elif htmlPage is HtmlPage.NON_SPECIFIED:
            currentHeader = header
            f.write("\n<H2> %s </H2>\n" % currentHeader)
            if addLinkForNextDim:
                name_for_click = next_dimension_str(add_to_names)
                f.write('<A HREF="%s">\n' % (name + name_for_click  + '.html'))
            for ifun in range(1, maxFunctionIndex + 1):
                f.write(addImage('%s_f%03d%s.%s' % (name, ifun, add_to_names, extension), not addLinkForNextDim))
            if addLinkForNextDim:
                f.write('"\n</A>\n')
        elif htmlPage is HtmlPage.PPRLDMANY_BY_GROUP:
            currentHeader = 'Runtime distributions (ECDF), function groups over all targets'
            f.write("\n<H2> %s </H2>\n" % currentHeader)
            if addLinkForNextDim:
                name_for_click = next_dimension_str(add_to_names)
                f.write('<A HREF="%s">\n' % (name + name_for_click  + '.html'))
            
            for fg in functionGroups:
                f.write(addImage('%s_%s%s.%s' % (name, fg, add_to_names, extension), not addLinkForNextDim))
            if addLinkForNextDim:
                f.write('"\n</A>\n')
        elif htmlPage is HtmlPage.PPTABLE:
            currentHeader = 'aRT in number of function evaluations'
            f.write("<H2> %s </H2>\n" % currentHeader)
            f.write("\n<!--pptableHtml-->\n")
            key = (('bbobpptablecaptionrlbased' if genericsettings.runlength_based_targets else 'bbobpptablecaptionfixed') if genericsettings.current_testbed.name != 'bbob-biobj' else 'bbobpptablecaptionbiobjfixed')            
            f.write(captionStringFormat % htmldesc.getValue('##' + key + '##'))
    
        elif htmlPage is HtmlPage.PPRLDISTR:
            names = ['pprldistr', 'ppfvdistr']
            dimensions = [5, 20]
            
            headerECDF = ' Empirical cumulative distribution functions (ECDF)'
            f.write("<H2> %s </H2>\n" % headerECDF)
            for dimension in dimensions:
                for typeKey, typeValue in functionGroups.iteritems():
                    f.write('<p><b>%s in %d-D</b></p>' % (typeValue, dimension))
                    f.write('<div>')
                    for name in names:
                        f.write(addImage('%s_%02dD_%s.%s' % (name, dimension, typeKey, extension), True))
                    f.write('</div>')

            key = 'bbobpprldistrlegend' + genericsettings.current_testbed.scenario
            f.write(captionStringFormat % htmldesc.getValue('##' + key + '##'))

        elif htmlPage is HtmlPage.PPLOGLOSS:
            dimensions = [5, 20]
            if not isBiobjective:            
                currentHeader = 'aRT loss ratios'
                f.write("<H2> %s </H2>\n" % currentHeader)
                for dimension in dimensions:
                    f.write(addImage('pplogloss_%02dD_noiselessall.%s' % (dimension, extension), True))
                f.write("\n<!--tables-->\n")
                scenario = genericsettings.current_testbed.scenario
                f.write(captionStringFormat % htmldesc.getValue('##bbobloglosstablecaption' + scenario + '##'))
            
                for typeKey, typeValue in functionGroups.iteritems():
                    f.write('<p><b>%s in %s</b></p>' % (typeValue, '-D and '.join(str(x) for x in dimensions) + '-D'))
                    f.write('<div>')
                    for dimension in dimensions:
                        f.write(addImage('pplogloss_%02dD_%s.%s' % (dimension, typeKey, extension), True))
                    f.write('</div>')
                        
                f.write(captionStringFormat % htmldesc.getValue('##bbobloglossfigurecaption' + scenario + '##'))
        
        if caption:
            f.write(captionStringFormat % caption)

        f.write("\n</BODY>\n</HTML>")
    
def write_ECDF(f, dimension, extension, captionStringFormat, functionGroups):
    """Writes line for ECDF images."""

    names = ['pprldmany']
    
    headerECDF = 'Empirical Cumulative Distribution Functions (ECDFs) per function group for dimension %d' % dimension
    f.write("\n<H2> %s </H2>\n" % headerECDF)
    for typeKey, typeValue in functionGroups.iteritems():
        f.write('<p><b>%s</b></p>' % typeValue)
        for name in names:
            f.write(addImage('%s_%02dD_%s.%s' % (name, dimension, typeKey, extension), True))
    
    f.write(captionStringFormat % ('\n##bbobECDFslegend%d##' % dimension))

def write_pptables(f, dimension, captionStringFormat, maxFunctionIndex, bestAlgExists):
    """Writes line for pptables images."""

    additionalText = 'divided by the best aRT measured during BBOB-2009' if bestAlgExists else ''
    currentHeader = 'Table showing the aRT in number of function evaluations %s ' \
                'for dimension %d' % (additionalText, dimension)
    
    f.write("\n<H2> %s </H2>\n" % currentHeader)
    for ifun in range(1, maxFunctionIndex + 1):
        f.write("\n<!--pptablesf%03d%02dDHtml-->\n" % (ifun, dimension))
    
    if genericsettings.isTab:
        key = 'bbobpptablesmanylegend' + genericsettings.current_testbed.scenario
        f.write(captionStringFormat % htmldesc.getValue('##' + key + str(dimension) + '##'))

def copy_js_files(outputdir):
    """Copies js files to output directory."""
    
    js_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'js')
    for file in os.listdir(js_folder):
        if file.endswith(".js"):
            shutil.copy(os.path.join(js_folder, file), outputdir)  


def discretize_limits(limits, smaller_steps_limit=3.1):
    """return new limits with discrete values in k * 10**i with k in [1, 3].

    `limits` has len 2 and the new lower limit is always ``10**-0.2``. 

    if `limits[1] / limits[0] < 10**smaller_steps_limits`, k == 3 is an 
    additional choice.
    """
    ymin, ymax = limits
    ymin=np.max((ymin, 10**-0.2))
    ymax=int(ymax + 1)

    ymax_new = 10**np.ceil(np.log10(ymax)) * (1 + 1e-6)
    if 3. * ymax_new / 10 > ymax and np.log10(ymax / ymin) < smaller_steps_limit:
        ymax_new *= 3. / 10
    ymin_new = 10**np.floor(np.log10(ymin)) / (1 + 1e-6)
    if 11 < 3 and 3 * ymin_new < ymin and np.log10(ymax / ymin) < 1.1:
        ymin_new *= 3

    if ymin_new < 1.1:
        ymin_new = 10**-0.2
    ymin_new = 10**-0.2
    return ymin_new, ymax_new


def marker_positions(xdata, ydata, nbperdecade, maxnb,
                     ax_limits=None, y_transformation=None):
    """return randomized marker positions

    replacement for downsample, could be improved by becoming independent
    of axis limits?
    """
    if ax_limits is None:  # use current axis limits
        ax_limits = plt.axis()
    tfy = y_transformation
    if tfy is None:
        tfy = lambda x: x  # identity

    xdatarange = np.log10(max([max(xdata), ax_limits[0], ax_limits[1]]) + 0.5) - \
                 np.log10(min([min(xdata), ax_limits[0], ax_limits[1]]) + 0.5)  #np.log10(xdata[-1]) - np.log10(xdata[0])
    ydatarange = tfy(max([max(ydata), ax_limits[2], ax_limits[3]]) + 0.5) - \
                 tfy(min([min(ydata), ax_limits[2], ax_limits[3]]) + 0.5)  # tfy(ydata[-1]) - tfy(ydata[0])
    nbmarkers = np.min([maxnb, nbperdecade +
                               np.ceil(nbperdecade * (1e-99 + np.abs(np.log10(max(xdata)) - np.log10(min(xdata)))))])
    probs = np.abs(np.diff(np.log10(xdata))) / xdatarange + \
            np.abs(np.diff(tfy(ydata))) / ydatarange
    xpos = []
    ypos= []
    if sum(probs) > 0:
        xoff = np.random.rand() / nbmarkers
        probs /= sum(probs)
        cum = np.cumsum(probs)
        for xact in np.arange(0, 1, 1./nbmarkers):
            pos = xoff + xact + (1./nbmarkers) * (0.3 + 0.4 * np.random.rand())
            idx = np.abs(cum - pos).argmin()  # index of closest value
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
    res = plt.plot(x, y, **kwargs)  # shouldn't this be done in the calling code?

    if 'marker' in kwargs and len(x) > 0:
        # x2, y2 = downsample(x, y)
        x2, y2 = marker_positions(x, y, nbperdecade, 19, plt.axis(),
                                  np.log10 if logscale else None)
        res2 = plt.plot(x2, y2)
        for i in res2:
            i.update_from(res[0]) # copy all attributes of res
        plt.setp(res2, linestyle='', label='')
        res.extend(res2)

    if 'label' in kwargs:
        res3 = plt.plot([], [], **kwargs)
        for i in res3:
            i.update_from(res[0]) # copy all attributes of res
        res.extend(res3)

    plt.setp(res[0], marker='', label='')
    return res

def consecutiveNumbers(data, prefix = ''):
    """Groups a sequence of integers into ranges of consecutive numbers.
    If the prefix is set then the it's placed before each number.

    Example::
      >>> import os
      >>> os.chdir(os.path.abspath(os.path.dirname(os.path.dirname('__file__'))))
      >>> import bbob_pproc as bb
      >>> bb.ppfig.consecutiveNumbers([0, 1, 2, 4, 5, 7, 8, 9])
      '0-2, 4, 5, 7-9'
      >>> bb.ppfig.consecutiveNumbers([0, 1, 2, 4, 5, 7, 8, 9], 'f')
      'f0-f2, f4, f5, f7-f9'

    Range of consecutive numbers is at least 3 (therefore [4, 5] is
    represented as "4, 5").
    """
    res = []
    tmp = groupByRange(data)
    for i in tmp:
        tmpstring = list(prefix + str(j) for j in i)
        if len(i) <= 2 : # This means length of ranges are at least 3
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
    for _k, g in groupby(enumerate(data), lambda (i,x):i-x):
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
        if j > limits[0] and j < limits[1]: # tick annotations only within the limits
            newxticks.append('%d' % round(np.log10(j)))
        else:
            newxticks.append('')
    plt.xticks(_xticks[0], newxticks)  # this changes the limits (only in newer versions of mpl?)
    plt.xlim(xlims[0], xlims[1])
    # TODO: check the xlabel is changed accordingly?

def beautify():
    """ Customize a figure by adding a legend, axis label, etc."""
    # TODO: what is this function for?
    # Input checking

    # Get axis handle and set scale for each axis
    axisHandle = plt.gca()
    axisHandle.set_yscale("log")

    # Grid options
    axisHandle.grid(True)

    _ymin, ymax = plt.ylim()
    plt.ylim(ymin=10**-0.2, ymax=ymax) # Set back the default maximum.

    tmp = axisHandle.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(np.log10(i)))
    axisHandle.set_yticklabels(tmp2)
    axisHandle.set_ylabel('log10 of aRT')

def generateData(dataSet, targetFuncValue):
    """Returns an array of results to be plotted.

    1st column is ert, 2nd is  the number of success, 3rd the success
    rate, 4th the sum of the number of function evaluations, and
    finally the median on successful runs.
    """
    it = iter(reversed(dataSet.evals))
    i = it.next()
    prev = np.array([np.nan] * len(i))

    while i[0] <= targetFuncValue:
        prev = i
        try:
            i = it.next()
        except StopIteration:
            break

    data = prev[1:].copy() # keep only the number of function evaluations.
    # was up to rev4997: succ = (np.isnan(data) == False)  # better: ~np.isnan(data)
    succ = np.isfinite(data)
    if succ.any():
        med = toolsstats.prctile(data[succ], 50)[0]
        #Line above was modified at rev 3050 to make sure that we consider only
        #successful trials in the median
    else:
        med = np.nan

    # prepare to compute runlengths / aRT with restarts (AKA SP1)
    data[np.isnan(data)] = dataSet.maxevals[np.isnan(data)]

    res = []
    res.extend(toolsstats.sp(data, issuccessful=succ, allowinf=False))
    res.append(np.mean(data)) #mean(FE)
    res.append(med)

    return np.array(res)

def plot(dsList, _valuesOfInterest=(10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8),
         isbyinstance=True, kwargs={}):
    """From a DataSetList, plot a graph. Not in use and superseeded by ppfigdim.main!?"""

    #set_trace()
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
            for j, idx in dictinstance.iteritems():
                tmp = StrippedUpDS()
                idxs = list(k + 1 for k in idx)
                idxs.insert(0, 0)
                tmp.evals = i.evals[:, np.r_[idxs]].copy()
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
                if tmp[2] > 0: #Number of success is larger than 0
                    succ.append(np.append(x, tmp))
                    if tmp[2] < j.nbRuns():
                        displaynumber.append((x, tmp[0], tmp[2]))
                else:
                    unsucc.append(np.append(x, tmp))

        if succ:
            tmp = np.vstack(succ)
            #aRT
            res.extend(plt.plot(tmp[:, 0], tmp[:, 1], **kwargs))
            #median
            tmp2 = plt.plot(tmp[:, 0], tmp[:, -1], **kwargs)
            plt.setp(tmp2, linestyle='', marker='+', markersize=30, markeredgewidth=5)
            #, color=colors[i], linestyle='', marker='+', markersize=30, markeredgewidth=5))
            res.extend(tmp2)

        # To have the legend displayed whatever happens with the data.
        tmp = plt.plot([], [], **kwargs)
        plt.setp(tmp, label=' %+d' % (np.log10(valuesOfInterest[i])))
        res.extend(tmp)

        #Only for the last target function value
        if unsucc:
            tmp = np.vstack(unsucc) # tmp[:, 0] needs to be sorted!
            res.extend(plt.plot(tmp[:, 0], tmp[:, 1], **kwargs))

    if displaynumber: # displayed only for the smallest valuesOfInterest
        for j in displaynumber:
            t = plt.text(j[0], j[1]*1.85, "%.0f" % j[2],
                         horizontalalignment="center",
                         verticalalignment="bottom")
            res.append(t)

    return res

