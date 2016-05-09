#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepares the descriptions of images and tables which will be converted to html.

This module creates a tex file with all the descriptions of the images and tables.

"""

import os
import sys
import warnings

from . import genericsettings, pplogloss, ppfigdim, pptable, pprldistr, config
from .compall import pptables, ppfigs
from .comp2 import ppscatter, pptable2

# Add the path to bbob_pproc
if __name__ == "__main__":
    (filepath, filename) = os.path.split(sys.argv[0])
    sys.path.append(os.path.join(filepath, os.path.pardir))
    import matplotlib

    matplotlib.use('Agg')  # To avoid window popup and use without X forwarding

# Initialization

header = """

\\documentclass{article}

% Packages
\\usepackage{graphicx}
\usepackage[usenames,dvipsnames]{xcolor}
\\usepackage{MnSymbol}

% pre-defined commands
\\newcommand{\\DIM}{\ensuremath{\mathrm{DIM}}}
\\newcommand{\\aRT}{\ensuremath{\mathrm{aRT}}}
\\newcommand{\\FEvals}{\ensuremath{\mathrm{FEvals}}}
\\newcommand{\\nruns}{\ensuremath{\mathrm{Nruns}}}
\\newcommand{\\Dfb}{\ensuremath{\Delta f_{\mathrm{best}}}}
\\newcommand{\\Df}{\ensuremath{\Delta f}}
\\newcommand{\\nbFEs}{\ensuremath{\mathrm{\#FEs}}}
\\newcommand{\\fopt}{\ensuremath{f_\mathrm{opt}}}
\\newcommand{\\hvref}{\ensuremath{HV_\mathrm{ref}}}
\\newcommand{\\ftarget}{\ensuremath{f_\mathrm{t}}}
\\newcommand{\\CrE}{\ensuremath{\mathrm{CrE}}}
\\newcommand{\\change}[1]{{\color{red} #1}}

\\begin{document}

"""


def main(latex_commands_for_html):
    """Reads all the descriptions and saves them into a tex file. 

    """

    f = open(latex_commands_for_html, 'w')

    f.write(header)

    for scenario in genericsettings.all_scenarios:
        # set up scenario, especially wrt genericsettings
        if scenario == genericsettings.scenario_rlbased:
            genericsettings.runlength_based_targets = True
            config.config(isBiobjective=False)
        elif scenario == genericsettings.scenario_fixed:
            genericsettings.runlength_based_targets = False
            config.config(isBiobjective=False)
        elif scenario == genericsettings.scenario_biobjfixed:
            genericsettings.runlength_based_targets = False
            config.config(isBiobjective=True)
        else:
            warnings.warn("Scenario not supported yet in HTML")

        # prepare LaTeX captions first
        # 1. ppfigs
        f.writelines(prepare_providecommand('bbobECDFslegend', scenario,
                                            ppfigs.prepare_ecdfs_figure_caption()
                                            .replace('REFERENCE_ALGORITHM', 'REFERENCEALGORITHM')))
        f.writelines(prepare_providecommand('bbobppfigslegend', scenario,
                                            ppfigs.prepare_scaling_figure_caption()
                                            .replace('REFERENCE_ALGORITHM', 'REFERENCEALGORITHM')))

        # 2. pprldistr
        f.writelines(prepare_providecommand('bbobpprldistrlegend', scenario,
                                            pprldistr.caption_single().replace('TO_BE_REPLACED', 'TOBEREPLACED')))
        pprldistrtwo = (pprldistr.caption_two()).replace('\\algorithmA', 'algorithmA')
        pprldistrtwo = pprldistrtwo.replace('\\algorithmB', 'algorithmB')
        f.writelines(prepare_providecommand('bbobpprldistrlegendtwo', scenario, pprldistrtwo))

        # 3. ppfigdim
        f.writelines(prepare_providecommand('bbobppfigdimlegend', scenario,
                                            ppfigdim.scaling_figure_caption()
                                            .replace('values_of_interest', 'valuesofinterest')))

        # 4. pptable
        f.writelines(prepare_providecommand('bbobpptablecaption', scenario, pptable.get_table_caption()))

        # 5. pptable2
        pptable2Legend = (pptable2.get_table_caption()).replace('\\algorithmA', 'algorithmA')
        pptable2Legend = pptable2Legend.replace('\\algorithmB', 'algorithmB')
        pptable2Legend = pptable2Legend.replace('\\algorithmAshort', 'algorithmAshort')
        pptable2Legend = pptable2Legend.replace('\\algorithmBshort', 'algorithmBshort')
        f.writelines(prepare_providecommand('bbobpptablestwolegend', scenario, pptable2Legend))

        # 6. pptables
        f.writelines(prepare_providecommand('bbobpptablesmanylegend', scenario, pptables.get_table_caption()))

        # 7. ppscatter
        ppscatterLegend = ppscatter.prepare_figure_caption().replace('REFERENCE_ALGORITHM', 'REFERENCEALGORITHM')
        ppscatterLegend = ppscatterLegend.replace('\\algorithmA', 'algorithmA')
        ppscatterLegend = ppscatterLegend.replace('\\algorithmB', 'algorithmB')
        f.writelines(prepare_providecommand('bbobppscatterlegend', scenario, ppscatterLegend))

        # 8. pplogloss
        f.writelines(prepare_providecommand('bbobloglosstablecaption', scenario,
                                            pplogloss.table_caption().replace('Figure~\\ref{fig:ERTlogloss}',
                                                                              'the following figure')))
        f.writelines(prepare_providecommand('bbobloglossfigurecaption', scenario,
                                            pplogloss.figure_caption().replace('Figure~\\ref{tab:ERTloss}',
                                                                               'the previous figure')))

        # prepare tags for later HTML preparation
        # 1. ppfigs
        for dim in ['5', '20']:
            f.write(prepare_item('bbobECDFslegend' + scenario + dim, 'bbobECDFslegend' + scenario, str(dim)))
        param = '$f_1$ and $f_{%d}$' % genericsettings.current_testbed.number_of_functions
        f.write(prepare_item('bbobppfigslegend' + scenario, param=param))

        # 2. pprldistr
        f.write(prepare_item('bbobpprldistrlegend' + scenario))
        f.write(prepare_item('bbobpprldistrlegendtwo' + scenario))
        # 3. ppfigdim
        f.write(prepare_item('bbobppfigdimlegend' + scenario))
        # 4. pptable
        f.write(prepare_item('bbobpptablecaption' + scenario))
        # 5. pptable2
        f.write(prepare_item('bbobpptablestwolegend' + scenario, param='48'))

        # 6. pptables
        command_name = 'bbobpptablesmanylegend' + scenario
        for dim in ['5', '20']:
            f.write(prepare_item(command_name + dim, command_name, 'dimension ' + dim))

        # 7. ppscatter
        param = '$f_1$ - $f_{%d}$' % genericsettings.current_testbed.number_of_functions
        f.write(prepare_item('bbobppscatterlegend' + scenario, param=param))

        # 8. pplogloss
        f.write(prepare_item('bbobloglosstablecaption' + scenario))
        f.write(prepare_item('bbobloglossfigurecaption' + scenario))

    f.write('\n\#\#\#\n\\end{document}\n')
    f.close()


def prepare_providecommand(command, scenario, captiontext):
    return ['\\providecommand{\\', command, scenario, '}[1]{\n', captiontext, '\n}\n']


def prepare_item(name, command_name='', param=''):
    if not command_name:
        command_name = name

    return '\#\#%s\#\#\n\\%s{%s}\n' % (name, command_name, param)
