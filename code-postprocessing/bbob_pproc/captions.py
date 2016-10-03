#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Provides basic functionality for creating figure and table captions.

    In particular, a replace functionality is provided that takes a raw
    caption text and replaces certain !!KEYWORDS!!  with the corresponding
    values, depending on the current settings (genericsettings, testbedsettings,
    ...).
"""
from . import genericsettings
from . import testbedsettings


# certain settings, only needed for the captions for now are grouped here:
ynormalize_by_dimension = True               



def replace(text):
    """Replaces all !!KEYWORDS!! in the text as specified in replace_dict."""
    
    for key in replace_dict:
        if key in text:
            text = text.replace(key, replace_dict[key]())
    
    # TODO: give a warning if in the resulting text, still some `!!` occur
    
    return text
    
    
    
def get_reference_algorithm_text():
    testbed = testbedsettings.current_testbed
    if testbed.reference_algorithm_filename:
        if (testbed.name == testbedsettings.testbed_name_single or
                testbed.name == testbedsettings.default_testbed_single_noisy
                or testbed.name == testbedsettings.testbed_name_bi):
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
                else:
                    text = "the reference algorithm"
        else:
            raise NotImplementedError('reference algorithm not supported for this testbed')
    else:
        raise NotImplementedError('no reference algorithm indicated in testbedsettings.py')

    return text
    
    
replace_dict = {
        '!!NOTCHED_BOXES!!': lambda: r"""Notched boxes: interquartile range with median of simulated runs; """ 
            if genericsettings.scaling_figures_with_boxes else "",
        '!!DF!!': lambda: r"""\\Df""" if not (testbedsettings.current_testbed.name == testbedsettings.testbed_name_bi) else r"""\\DI""",
        '!!FOPT!!': lambda: r"""\\fopt""" if not (testbedsettings.current_testbed.name == testbedsettings.testbed_name_bi) else r"""\\hvref""",
        '!!DIVIDED_BY_DIMENSION!!': lambda: r"""divided by dimension and """ if ynormalize_by_dimension else "",
        '!!LIGHT_THICK_LINE!!': lambda: r"""The light thick line with diamonds indicates """ + get_reference_algorithm_text() + r""" for the most difficult target. """ if testbedsettings.current_testbed.reference_algorithm_filename else "",
        '!!F!!': lambda: r"""I_{\mathrm HV}^{\mathrm COCO}""" if testbedsettings.current_testbed.name == testbedsettings.testbed_name_bi else "f",
        '!!THE_REF_ALG!!': get_reference_algorithm_text
         }
