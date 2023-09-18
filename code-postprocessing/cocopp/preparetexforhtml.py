#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepares the descriptions of images and tables which will be converted to html.

This module creates a tex file with all the descriptions of the images and tables.

"""

import os
import sys
import warnings

from . import genericsettings, pplogloss, ppfigdim, ppfigcons1, pptable, pprldistr, config
from . import testbedsettings
from .compall import pptables, ppfigs, ppfigcons
from .comp2 import ppscatter

# Initialization

header = """

\\documentclass{article}

% Packages
\\usepackage{graphicx}
\\usepackage[usenames,dvipsnames]{xcolor}
\\usepackage{MnSymbol}

% pre-defined commands
\\newcommand{\\DIM}{\ensuremath{\mathrm{DIM}}}
\\newcommand{\\ERT}{\ensuremath{\mathrm{ERT}}}
\\newcommand{\\FEvals}{\ensuremath{\mathrm{FEvals}}}
\\newcommand{\\nruns}{\ensuremath{\mathrm{Nruns}}}
\\newcommand{\\Dfb}{\ensuremath{\Delta f_{\mathrm{best}}}}
\\newcommand{\\Df}{\ensuremath{\Delta f}}
\\newcommand{\\DI}{\ensuremath{\Delta I}}
\\newcommand{\\nbFEs}{\ensuremath{\mathrm{\#FEs}}}
\\newcommand{\\fopt}{\ensuremath{f_\mathrm{opt}}}
\\newcommand{\\hvref}{I^{\mathrm{ref}}}
\\newcommand{\\ftarget}{\ensuremath{f_\mathrm{t}}}
\\newcommand{\\CrE}{\ensuremath{\mathrm{CrE}}}
\\newcommand{\\change}[1]{{\color{red} #1}}
\\newcommand{\\cocoversion}{??COCOVERSION??}

\\begin{document}

"""


def main(latex_commands_for_html):
    """Reads all the descriptions and saves them into a tex file. 

    """

    f = open(latex_commands_for_html, 'w')

    f.write(header)

    single_objective_testbed = testbedsettings.default_suite_single_noisy if genericsettings.isNoisy \
        else testbedsettings.default_suite_single

    for scenario in testbedsettings.all_scenarios:
        # set up scenario, especially wrt genericsettings
        if scenario == testbedsettings.scenario_rlbased:
            genericsettings.runlength_based_targets = True
            config.config(single_objective_testbed)
        elif scenario == testbedsettings.scenario_fixed:
            genericsettings.runlength_based_targets = False
            config.config(single_objective_testbed)
        elif scenario == testbedsettings.scenario_biobjfixed:
            genericsettings.runlength_based_targets = False
            config.config(testbedsettings.default_suite_bi)
        elif scenario == testbedsettings.scenario_biobjrlbased:
            genericsettings.runlength_based_targets = True
            config.config(testbedsettings.default_suite_bi)
        elif scenario == testbedsettings.scenario_biobjextfixed:
            genericsettings.runlength_based_targets = False
            config.config(testbedsettings.suite_name_bi_ext)
        elif scenario == testbedsettings.scenario_constrainedfixed:
            genericsettings.runlength_based_targets = False
            config.config(testbedsettings.suite_name_cons)
        elif scenario == testbedsettings.scenario_largescalefixed:
            genericsettings.runlength_based_targets = False
            config.config(testbedsettings.suite_name_ls)
        elif scenario == testbedsettings.scenario_mixintfixed:
            genericsettings.runlength_based_targets = False
            config.config(testbedsettings.suite_name_mixint)
        elif scenario == testbedsettings.scenario_sboxcostfixed:
            genericsettings.runlength_based_targets = False
            config.config(testbedsettings.SBOXCOSTTestbed)
        else:
            warnings.warn("Scenario '%s' not supported yet in HTML" % scenario)

        # prepare LaTeX captions first
        # 1. ppfigs
        f.writelines(prepare_providecommand('bbobECDFslegend', scenario,
                                            ppfigs.prepare_ecdfs_figure_caption()))
        f.writelines(prepare_providecommand('bbobppfigslegend', scenario,
                                            ppfigs.prepare_scaling_figure_caption()))

        # 2. pprldistr
        f.writelines(prepare_providecommand('bbobpprldistrlegend', scenario,
                                            pprldistr.caption_single()))
        pprldistrtwo = (pprldistr.caption_two()).replace('\\algorithmA', 'algorithmA')
        pprldistrtwo = pprldistrtwo.replace('\\algorithmB', 'algorithmB')
        f.writelines(prepare_providecommand('bbobpprldistrlegendtwo', scenario, pprldistrtwo))

        # 3. ppfigdim
        f.writelines(prepare_providecommand('bbobppfigdimlegend', scenario,
                                            ppfigdim.scaling_figure_caption()
                                            .replace('values_of_interest', 'valuesofinterest')))

        # 4. pptable
        f.writelines(prepare_providecommand('bbobpptablecaption', scenario, pptable.get_table_caption()))

        # 5. pptables
        f.writelines(prepare_providecommand('bbobpptablesmanylegend', scenario, pptables.get_table_caption()))

        # 6. ppscatter
        ppscatterLegend = ppscatter.prepare_figure_caption()
        ppscatterLegend = ppscatterLegend.replace('\\algorithmA', 'algorithmA')
        ppscatterLegend = ppscatterLegend.replace('\\algorithmB', 'algorithmB')
        f.writelines(prepare_providecommand('bbobppscatterlegend', scenario, ppscatterLegend))

        # 7. pplogloss
        f.writelines(prepare_providecommand('bbobloglosstablecaption', scenario,
                                            pplogloss.table_caption().replace('Figure~\\ref{fig:ERTlogloss}',
                                                                              'the following figure')))
        f.writelines(prepare_providecommand('bbobloglossfigurecaption', scenario,
                                            pplogloss.figure_caption().replace('Figure~\\ref{tab:ERTloss}',
                                                                               'the previous figure')))

        if scenario == testbedsettings.scenario_constrainedfixed:
            # 8. ppfigcons and ppfigcons1
            f.writelines(prepare_providecommand('bbobppfigconslegend', scenario, 
                                                ppfigcons.prepare_scaling_figure_caption()))
            f.writelines(prepare_providecommand('bbobppfigconsonelegend', scenario, 
                                                ppfigcons1.scaling_figure_caption()
                                                .replace('values_of_interest', 'valuesofinterest')))

        # prepare tags for later HTML preparation
        testbed = testbedsettings.current_testbed
        # 1. ppfigs
        f.write(prepare_item('bbobECDFslegend' + scenario))
        param = '$f_{%d}$ and $f_{%d}$' % (min(testbed.functions_with_legend), max(testbed.functions_with_legend))
        f.write(prepare_item('bbobppfigslegend' + scenario, param=param))
        # 2. pprldistr
        f.write(prepare_item('bbobpprldistrlegend' + scenario))
        f.write(prepare_item('bbobpprldistrlegendtwo' + scenario))
        # 3. ppfigdim
        f.write(prepare_item('bbobppfigdimlegend' + scenario))
        # 4. pptable
        f.write(prepare_item('bbobpptablecaption' + scenario, param='different dimensions'))
        # 5. pptables
        f.write(prepare_item('bbobpptablesmanylegend' + scenario, param='different dimensions'))

        # 6. ppscatter
        param = '$f_{%d}$ - $f_{%d}$' % (testbed.first_function_number, testbed.last_function_number)
        f.write(prepare_item('bbobppscatterlegend' + scenario, param=param))

        # 7. pplogloss
        f.write(prepare_item('bbobloglosstablecaption' + scenario))
        f.write(prepare_item('bbobloglossfigurecaption' + scenario))

        if scenario == testbedsettings.scenario_constrainedfixed:
            # 8. ppfigcons and ppfigcons1
            f.write(prepare_item('bbobppfigconslegend' + scenario, param=param))
            f.write(prepare_item('bbobppfigconsonelegend' + scenario))

    f.write('\n\#\#\#\n\\end{document}\n')
    f.close()


def prepare_providecommand(command, scenario, captiontext):
    return ['\\providecommand{\\', command, scenario, '}[1]{\n', captiontext, '\n}\n']


def prepare_providecommand_two(command, scenario, captiontext):
    return ['\\providecommand{\\', command, scenario, '}[2]{\n', captiontext, '\n}\n']


def prepare_item(name, command_name='', param=''):
    if not command_name:
        command_name = name

    return '\#\#%s\#\#\n\\%s{%s}\n' % (name, command_name, param)

def prepare_item_two(name, command_name='', paramOne='', paramTwo=''):
    if not command_name:
        command_name = name

    return '\#\#%s\#\#\n\\%s{%s}{%s}\n' % (name, command_name, paramOne, paramTwo)
