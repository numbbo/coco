#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepares the descriptions of images and tables which will be converted to html.

This module creates a tex file with all the descriptions of the images and tables.

"""

import os
import sys
import numpy
import warnings

# Add the path to bbob_pproc
if __name__ == "__main__":
    (filepath, filename) = os.path.split(sys.argv[0])
    sys.path.append(os.path.join(filepath, os.path.pardir))
    import matplotlib
    matplotlib.use('Agg') # To avoid window popup and use without X forwarding

from bbob_pproc import genericsettings, pplogloss, ppfigdim, pptable, pprldistr, config, pproc
from bbob_pproc.compall import pptables, ppfigs
from bbob_pproc.comp2 import ppscatter, pptable2

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

def main(verbose=True):
    """Reads all the descriptions and saves them into a tex file. 

    """

    latex_commands_for_html = os.path.join(os.path.dirname(os.path.realpath(__file__)), genericsettings.latex_commands_for_html + '.tex')
    
    f = open(latex_commands_for_html, 'w')
    
    for scenario in genericsettings.all_scenarios:
        # set up scenario, especially wrt genericsettings
        if (scenario == genericsettings.scenario_rlbased):
            genericsettings.runlength_based_targets = True
            config.config(isBiobjective=False)
        elif (scenario == genericsettings.scenario_fixed):
            genericsettings.runlength_based_targets = False
            config.config(isBiobjective=False)
        elif (scenario == genericsettings.scenario_biobjfixed):
            genericsettings.runlength_based_targets = False
            config.config(isBiobjective=True)
        else:
            warnings.warn("Scenario not supported yet in HTML")
    
        # update captions accordingly    
        # 1. ppfigs
        ppfigsftarget = genericsettings.current_testbed.ppfigs_ftarget
        ppfigsftarget = pproc.TargetValues.cast([ppfigsftarget] if numpy.isscalar(ppfigsftarget) else ppfigsftarget)
        f.writelines(['\\providecommand{\\bbobECDFslegend', scenario, '}[1]{\n', 
                  (ppfigs.ecdfs_figure_caption(ppfigsftarget)).replace('REFERENCE_ALGORITHM', 'REFERENCEALGORITHM'), '\n}\n'])
        f.writelines(['\\providecommand{\\bbobppfigslegend', scenario, '}[1]{\n',
                      ppfigs.prepare_scaling_figure_caption().replace('REFERENCE_ALGORITHM', 'REFERENCEALGORITHM'), '\n}\n'])

        # 2. pprldistr
        f.writelines(['\\providecommand{\\bbobpprldistrlegend', scenario , '}[1]{\n', 
                  (pprldistr.caption_single()).replace('TO_BE_REPLACED', 'TOBEREPLACED'), '\n}\n'])
        pprldistrtwo = (pprldistr.caption_two()).replace('\\algorithmA', 'algorithmA')
        pprldistrtwo = pprldistrtwo.replace('\\algorithmB', 'algorithmB')    
        f.writelines(['\\providecommand{\\bbobpprldistrlegendtwo', scenario, '}[1]{\n', pprldistrtwo, '\n}\n'])
              
        # 3. ppfigdim
        f.writelines(['\\providecommand{\\bbobppfigdimlegend', scenario, '}[1]{\n', 
                  (ppfigdim.scaling_figure_caption()).replace('values_of_interest', 'valuesofinterest'), '\n}\n'])

        # 4. pptable
        f.writelines(['\\providecommand{\\bbobpptablecaption', scenario, '}[1]{\n', 
                  pptable.get_table_caption(), '\n}\n'])

        # 5. pptable2
        pptable2Legend = (pptable2.get_table_caption()).replace('\\algorithmA', 'algorithmA')
        pptable2Legend = pptable2Legend.replace('\\algorithmB', 'algorithmB')    
        pptable2Legend = pptable2Legend.replace('\\algorithmAshort', 'algorithmAshort')    
        pptable2Legend = pptable2Legend.replace('\\algorithmBshort', 'algorithmBshort')    
        f.writelines(['\\providecommand{\\bbobpptablestwolegend', scenario, '}[1]{\n', 
                  pptable2Legend, '\n}\n'])

        # 6. pptables
        f.writelines(['\\providecommand{\\bbobpptablesmanylegend', scenario, '}[1]{\n',
                      pptables.get_table_caption(), '\n}\n'])

        # 7. ppscatter
        ppscatterLegend = ppscatter.prepare_figure_caption().replace('REFERENCE_ALGORITHM', 'REFERENCEALGORITHM')
        ppscatterLegend = ppscatterLegend.replace('\\algorithmA', 'algorithmA')
        ppscatterLegend = ppscatterLegend.replace('\\algorithmB', 'algorithmB')
        f.writelines(['\\providecommand{\\bbobppscatterlegend', scenario, '}[1]{\n', ppscatterLegend, '\n}\n'])

        # 8. pplogloss
        f.writelines(['\\providecommand{\\bbobloglosstablecaption', scenario, '}[1]{\n',
                      pplogloss.table_caption().replace('Figure~\\ref{fig:ERTlogloss}',
                                                        'the following figure'), '\n}\n'])
        f.writelines(['\\providecommand{\\bbobloglossfigurecaption', scenario, '}[1]{\n',
                      pplogloss.figure_caption().replace('Figure~\\ref{tab:ERTloss}',
                                                       'the previous figure'), '\n}\n'])

    f.write(header)

    for scenario in genericsettings.all_scenarios:
        # set up scenario, especially wrt genericsettings
        if (scenario == genericsettings.scenario_rlbased):
            genericsettings.runlength_based_targets = True
            config.config(isBiobjective=False)
        elif (scenario == genericsettings.scenario_fixed):
            genericsettings.runlength_based_targets = False
            config.config(isBiobjective=False)
        elif (scenario == genericsettings.scenario_biobjfixed):
            genericsettings.runlength_based_targets = False
            config.config(isBiobjective=True)
        else:
            warnings.warn("Scenario not supported yet in HTML")

        # update captions accordingly    
        # 1. ppfigs
        for dim in ['5', '20']:
            f.write(prepare_item('bbobECDFslegend' + scenario + dim, 'bbobECDFslegend' + scenario, str(dim)))                        
        param = '$f_1$ and $f_{%d}$' % (genericsettings.current_testbed.number_of_functions)
        f.write(prepare_item('bbobppfigslegend' + scenario, param = param))

        # 2. pprldistr
        f.write(prepare_item('bbobpprldistrlegend' + scenario))
        f.write(prepare_item('bbobpprldistrlegendtwo' + scenario))              
        # 3. ppfigdim
        f.write(prepare_item('bbobppfigdimlegend' + scenario))
        # 4. pptable
        f.write(prepare_item('bbobpptablecaption'+ scenario))
        # 5. pptable2
        f.write(prepare_item('bbobpptablestwolegend' + scenario, param = '48'))

        # 6. pptables
        command_name = 'bbobpptablesmanylegend' + scenario
        for dim in ['5', '20']:
            f.write(prepare_item(command_name + dim, command_name, 'dimension ' + dim))

        # 7. ppscatter
        param = '$f_1$ - $f_{%d}$' % (genericsettings.current_testbed.number_of_functions)
        f.write(prepare_item('bbobppscatterlegend' + scenario, param = param))

        # 8. pplogloss
        f.write(prepare_item('bbobloglosstablecaption' + scenario))
        f.write(prepare_item('bbobloglossfigurecaption' + scenario))
    
    f.write('\n\#\#\#\n\\end{document}\n')

def prepare_item(name, command_name = '', param = ''):
    
    if not command_name:
        command_name = name
        
    return '\#\#%s\#\#\n\\%s{%s}\n' % (name, command_name, param)


if __name__ == '__main__':
    main()

