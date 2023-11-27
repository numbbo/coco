#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Provides basic functionality for creating figure and table captions.

    In particular, a replace functionality is provided that takes a raw
    caption text and replaces certain !!KEYWORDS!!  with the corresponding
    values, depending on the current settings (genericsettings, testbedsettings,
    ...).
"""
import warnings
import numpy as np
from . import genericsettings
from . import testbedsettings
from . import pproc, toolsdivers

# certain settings, only needed for the captions for now are grouped here:
ynormalize_by_dimension = True


def replace(text, html=False):
    """Replaces all !!KEYWORDS!! in the text as specified in replace_dict.
    
       If html==True, some translation is done before and after the
       actual replacement in order to deal with HTML-specific codings.
    """

    global tohtml  # used for NBUP and NBLOW replacement
    tohtml = html

    if html:
        text = text.replace(r'&#8722;', '-')

    for key in replace_dict:
        if key in text:
            text = text.replace(key, replace_dict[key]())
    if html:
        for key in replace_dict_html:
            if key in text:
                text = text.replace(key, replace_dict_html[key]())

    if '!!' in text:
        warnings.warn("Still, '!!' occurs in caption after replacement: " + text)

    if html:
        text = text.replace('-D', 'different dimensions')  # for caption in pprldmany.html
        text = text.replace('-', r'&#8722;')
        text = text.replace('\\triangledown',
                            '<span style="color:#008D00">&#9661;</span>')  # to color and display correctly in ppscatter.py
        text = text.replace('\\Diamond', '&#9671;')

    return text


def get_reference_algorithm_text(best_algorithm_mandatory=True):
    text = ''
    testbed = testbedsettings.current_testbed
    if testbed.reference_algorithm_filename:
        if (testbed.name in [testbedsettings.suite_name_single,
                             testbedsettings.default_testbed_single_noisy,
                             testbedsettings.suite_name_bi,
                             'bbob-largescale']):
            if testbed.reference_algorithm_displayname:
                if "best 2009" in testbed.reference_algorithm_displayname:
                    text = "the best algorithm from BBOB 2009"
                elif "best 2010" in testbed.reference_algorithm_displayname:
                    text = "the best algorithm from BBOB 2010"
                elif "best 2012" in testbed.reference_algorithm_displayname:
                    text = "the best algorithm from BBOB 2012"
                elif "best 2013" in testbed.reference_algorithm_displayname:
                    text = "the best algorithm from BBOB 2013"
                elif "best 2016" in testbed.reference_algorithm_displayname:
                    text = "the best algorithm from BBOB 2016"
                elif "best 2009-16" in testbed.reference_algorithm_displayname:
                    text = "the best algorithm from BBOB 2009--16"
                elif "best 2019" in testbed.reference_algorithm_displayname:
                    text = "the best algorithm from BBOB 2019"
                else:
                    text = "the reference algorithm"
        else:
            raise NotImplementedError('reference algorithm not supported for this testbed')
    elif best_algorithm_mandatory:
        raise NotImplementedError('no reference algorithm indicated in testbedsettings.py')

    return text


def get_best_ert_text():
    text = ''
    testbed = testbedsettings.current_testbed
    if testbed.reference_algorithm_filename:
        if (testbed.name == testbedsettings.suite_name_single or
                testbed.name == testbedsettings.default_testbed_single_noisy
                or testbed.name == testbedsettings.suite_name_bi):
            if testbed.reference_algorithm_displayname:
                if "best 2009" in testbed.reference_algorithm_displayname:
                    text = r"best \ERT\ measured during BBOB-2009"
                elif "best 2010" in testbed.reference_algorithm_displayname:
                    text = r"best \ERT\ measured during BBOB-2010"
                elif "best 2012" in testbed.reference_algorithm_displayname:
                    text = r"best \ERT\ measured during BBOB-2012"
                elif "best 2013" in testbed.reference_algorithm_displayname:
                    text = r"best \ERT\ measured during BBOB-2013"
                elif "best 2016" in testbed.reference_algorithm_displayname:
                    text = r"best \ERT\ measured during BBOB-2016"
                elif "best 2009-16" in testbed.reference_algorithm_displayname:
                    text = r"best \ERT\ measured during BBOB-2009-16"
                else:
                    text = r"the \ERT\ of the reference algorithm"
        else:
            raise NotImplementedError('reference algorithm not supported for this testbed')
    else:
        warnings.warn('no reference algorithm indicated in testbedsettings.py')

    return text


def get_light_brown_line_text(testbedname):
    if (testbedname == testbedsettings.suite_name_bi):
        return r"""Shown are aggregations over functions where the single
            objectives are in the same BBOB function class, as indicated on the
            left side and the aggregation over all 55 functions in the last row."""
    elif (testbedname in [testbedsettings.suite_name_bi_ext,
                          testbedsettings.suite_name_bi_mixint]):
        return r"""Shown are aggregations over functions where the single
            objectives are in the same BBOB function class, as indicated on the
            left side and the aggregation over all 92 functions in the last row."""
    elif (testbedname == testbedsettings.suite_name_cons):
        return r"""Shown are aggregations over problems where the objective
            functions are in the same BBOB function class and the aggregation
            over all 48 functions in the last row."""  # TODO: check whether this makes sense
    elif (testbedname in [testbedsettings.suite_name_single,
                          testbedsettings.suite_name_single_noisy]):
        return r"""Light brown lines in the background show ECDFs for the most difficult target of all
            algorithms benchmarked during BBOB-2009."""  # hard-coded here as long as this 
                                                         # is hard-coded also in the code
    elif (testbedname in [testbedsettings.suite_name_ls,
                          testbedsettings.suite_name_mixint,
                          'sbox-cost',
                          'bbob-JOINED-sbox-cost']):
        return ""
    else:
        warnings.warn("Current testbed not supported for this caption text.")
        return ""


# please try to avoid underscores in the labels to not break the HTML code:
replace_dict = {
    '!!BOOTSTRAPPED-BEGINNING!!': lambda: r"""E""" if (
                testbedsettings.current_testbed.name in [testbedsettings.suite_name_bi,
                                                         testbedsettings.suite_name_bi_ext,
                                                         testbedsettings.suite_name_bi_mixint]) else "Bootstrapped e",
    '!!SIMULATED-BOOTSTRAP!!': lambda: r"""""" if (
                testbedsettings.current_testbed.name in [testbedsettings.suite_name_bi,
                                                         testbedsettings.suite_name_bi_ext,
                                                         testbedsettings.suite_name_bi_mixint]) else "simulated (bootstrapped)",
    '!!BOOTSTRAPPED!!': lambda: r"""""" if (
            testbedsettings.current_testbed.name in [testbedsettings.suite_name_bi,
                                                     testbedsettings.suite_name_bi_ext,
                                                     testbedsettings.suite_name_bi_mixint]) else "(bootstrapped)",
    '!!NOTCHED-BOXES!!': lambda: r"""Notched boxes: interquartile range with median of simulated runs; """
    if genericsettings.scaling_figures_with_boxes else "",
    '!!DF!!': lambda: r"""\Df""" if not (testbedsettings.current_testbed.name in [testbedsettings.suite_name_bi,
                                                                                  testbedsettings.suite_name_bi_ext,
                                                                                  testbedsettings.suite_name_bi_mixint]) else r"""\DI""",
    '!!FOPT!!': lambda: r"""\fopt""" if not (testbedsettings.current_testbed.name in [testbedsettings.suite_name_bi,
                                                                                      testbedsettings.suite_name_bi_ext,
                                                                                      testbedsettings.suite_name_bi_mixint]) else r"""\hvref""",
    '!!DIVIDED-BY-DIMENSION!!': lambda: r"""divided by dimension and """ if ynormalize_by_dimension else "",
    '!!LIGHT-THICK-LINE!!': lambda: r"""The light thick line with diamonds indicates """ + get_reference_algorithm_text(
        False) + r""" for the most difficult target. """ if testbedsettings.current_testbed.reference_algorithm_filename else "",
    '!!F!!': lambda: r"""I_{\mathrm HV}^{\mathrm COCO}""" if (testbedsettings.current_testbed.name
                                                              in [testbedsettings.suite_name_bi,
                                                                  testbedsettings.suite_name_bi_ext,
                                                                  testbedsettings.suite_name_bi_mixint]) else "f",
    '!!THE-REF-ALG!!': lambda: get_reference_algorithm_text(False),
    '!!HARDEST-TARGET-LATEX!!': lambda: testbedsettings.current_testbed.hardesttargetlatex,
    '!!DIM!!': lambda: r"""\DIM""",
    '!!SINGLE-RUNLENGTH-FACTORS!!': lambda: '$' + 'D, '.join(
        [str(i) for i in genericsettings.single_runlength_factors[:6]]) + r'D,\dots$',
    '!!LIGHT-BROWN-LINES!!': lambda: get_light_brown_line_text(testbedsettings.current_testbed.name),
    '!!PPFIGS-FTARGET!!': lambda: get_ppfigs_ftarget(),
    '!!NUM-OF-TARGETS-IN-ECDF!!': lambda: str(len(testbedsettings.current_testbed.pprldmany_target_values)),
    '!!TARGET-RANGES-IN-ECDF!!': lambda: str(testbedsettings.current_testbed.pprldmany_target_range_latex),
    '!!TOTAL-NUM-OF-FUNCTIONS!!': lambda: str(
        testbedsettings.current_testbed.last_function_number - testbedsettings.current_testbed.first_function_number + 1),
    '!!BEST-ERT!!': lambda: get_best_ert_text(),
    '!!NBTARGETS-SCATTER!!': lambda: str(len(testbedsettings.current_testbed.ppscatter_target_values)),
    '!!NBLOW!!': lambda: get_nblow(),
    '!!NBUP!!': lambda: get_nbup()
}

replace_dict_html = {
    '\\Df': lambda: str(r"""&Delta;f"""),
    '\\DI': lambda: str(r"""&Delta;I""")
}

tohtml = False


def get_nblow():
    global tohtml
    targets = testbedsettings.current_testbed.ppscatter_target_values
    if genericsettings.runlength_based_targets:
        if tohtml:
            text = (toolsdivers.number_to_html(targets.label(0)) +
                    r'&times; DIM' if targets.times_dimension else '')
        else:
            text = (toolsdivers.number_to_latex(targets.label(0)) +
                    r'\times DIM' if targets.times_dimension else '')
        return text
    else:
        if tohtml:
            return toolsdivers.number_to_html(targets.label(0))
        else:
            return toolsdivers.number_to_latex(targets.label(0))


def get_nbup():
    global tohtml
    targets = testbedsettings.current_testbed.ppscatter_target_values
    if genericsettings.runlength_based_targets:
        if tohtml:
            text = (toolsdivers.number_to_html(targets.label(-1)) +
                    r'&times; DIM' if targets.times_dimension else '')
        else:
            text = (toolsdivers.number_to_latex(targets.label(-1)) +
                    r'\times DIM' if targets.times_dimension else '')
        return text
    else:
        if tohtml:
            return toolsdivers.number_to_html(targets.label(-1))
        else:
            return toolsdivers.number_to_latex(targets.label(-1))


def get_ppfigs_ftarget():
    target = testbedsettings.current_testbed.ppfigs_ftarget
    target = pproc.TargetValues.cast([target] if np.isscalar(target) else target)
    assert len(target) == 1

    return toolsdivers.number_to_latex(target.label(0))
