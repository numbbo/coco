#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepares the descriptions of images and tables which will be converted to html.

This module creates a tex file with all the descriptions of the images and tables.

"""

import os
import sys

# Add the path to bbob_pproc
if __name__ == "__main__":
    (filepath, filename) = os.path.split(sys.argv[0])
    sys.path.append(os.path.join(filepath, os.path.pardir))
    import matplotlib
    matplotlib.use('Agg') # To avoid window popup and use without X forwarding

from bbob_pproc import genericsettings, pplogloss, ppfigdim, pptable, pprldistr
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
\\newcommand{\\ART}{\ensuremath{\mathrm{ART}}}
\\newcommand{\\FEvals}{\ensuremath{\mathrm{FEvals}}}
\\newcommand{\\nruns}{\ensuremath{\mathrm{Nruns}}}
\\newcommand{\\Dfb}{\ensuremath{\Delta f_{\mathrm{best}}}}
\\newcommand{\\Df}{\ensuremath{\Delta f}}
\\newcommand{\\nbFEs}{\ensuremath{\mathrm{\#FEs}}}
\\newcommand{\\fopt}{\ensuremath{f_\mathrm{opt}}}
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
    
    f.writelines(['\\providecommand{\\bbobloglosstablecaption}[1]{\n', 
                  pplogloss.table_caption.replace('Figure~\\ref{fig:ERTlogloss}', 'the following figure'), '\n}\n'])
    f.writelines(['\\providecommand{\\bbobloglossfigurecaption}[1]{\n', 
                  pplogloss.figure_caption.replace('Figure~\\ref{tab:ERTloss}', 'the previous figure'), '\n}\n'])
    f.writelines(['\\providecommand{\\bbobpprldistrlegendrlbased}[1]{\n', 
                  pprldistr.caption_single_rlbased.replace('TO_BE_REPLACED', 'TOBEREPLACED'), '\n}\n'])
    f.writelines(['\\providecommand{\\bbobpprldistrlegendfixed}[1]{\n', 
                  pprldistr.caption_single_fixed.replace('TO_BE_REPLACED', 'TOBEREPLACED'), '\n}\n'])
    f.writelines(['\\providecommand{\\bbobppfigdimlegendrlbased}[1]{\n', 
                  ppfigdim.scaling_figure_caption_rlbased.replace('values_of_interest', 'valuesofinterest'), '\n}\n'])
    f.writelines(['\\providecommand{\\bbobppfigdimlegendfixed}[1]{\n', 
                  ppfigdim.scaling_figure_caption_fixed.replace('values_of_interest', 'valuesofinterest'), '\n}\n'])
    f.writelines(['\\providecommand{\\bbobpptablecaption}[1]{\n', 
                  pptable.table_caption, '\n}\n'])

    f.writelines(['\\providecommand{\\bbobppfigslegendrlbased}[1]{\n', 
                  ppfigs.scaling_figure_caption_start_rlbased.replace('REFERENCE_ALGORITHM', 'REFERENCEALGORITHM'), '\n}\n'])
    f.writelines(['\\providecommand{\\bbobppfigslegendfixed}[1]{\n', 
                  ppfigs.scaling_figure_caption_start_fixed.replace('REFERENCE_ALGORITHM', 'REFERENCEALGORITHM'), '\n}\n'])
    f.writelines(['\\providecommand{\\bbobppfigslegendend}[1]{\n', 
                  ppfigs.scaling_figure_caption_end, '\n}\n'])
    f.writelines(['\\providecommand{\\bbobECDFslegendrlbased}[1]{\n', 
                  ppfigs.ecdfs_figure_caption_rlbased.replace('REFERENCE_ALGORITHM', 'REFERENCEALGORITHM'), '\n}\n'])
    f.writelines(['\\providecommand{\\bbobECDFslegendstandard}[1]{\n', 
                  ppfigs.ecdfs_figure_caption_standard.replace('REFERENCE_ALGORITHM', 'REFERENCEALGORITHM'), '\n}\n'])
    f.writelines(['\\providecommand{\\bbobpptablesmanylegend}[1]{\n', 
                  pptables.tables_many_legend, '\n}\n'])
    f.writelines(['\\providecommand{\\bbobpptablesmanylegendexpensive}[1]{\n', 
                  pptables.tables_many_expensive_legend, '\n}\n'])

    ppscatterLegend = ppscatter.caption_start_rlbased.replace('REFERENCE_ALGORITHM', 'REFERENCEALGORITHM')
    ppscatterLegend = ppscatterLegend.replace('\\algorithmA', 'algorithmA')
    ppscatterLegend = ppscatterLegend.replace('\\algorithmB', 'algorithmB')    
    f.writelines(['\\providecommand{\\bbobppscatterlegendrlbased}[1]{\n', ppscatterLegend, '\n}\n'])

    ppscatterLegend = ppscatter.caption_start_fixed.replace('REFERENCE_ALGORITHM', 'REFERENCEALGORITHM')
    ppscatterLegend = ppscatterLegend.replace('\\algorithmA', 'algorithmA')
    ppscatterLegend = ppscatterLegend.replace('\\algorithmB', 'algorithmB')    
    f.writelines(['\\providecommand{\\bbobppscatterlegendfixed}[1]{\n', ppscatterLegend, '\n}\n'])
    f.writelines(['\\providecommand{\\bbobppscatterlegendend}[1]{\n', 
                  ppscatter.caption_finish, '\n}\n'])

    pprldistrtwo = pprldistr.caption_two_rlbased.replace('\\algorithmA', 'algorithmA')
    pprldistrtwo = pprldistrtwo.replace('\\algorithmB', 'algorithmB')    
    f.writelines(['\\providecommand{\\bbobpprldistrlegendtworlbased}[1]{\n', pprldistrtwo, '\n}\n'])

    pprldistrtwo = pprldistr.caption_two_fixed.replace('\\algorithmA', 'algorithmA')
    pprldistrtwo = pprldistrtwo.replace('\\algorithmB', 'algorithmB')    
    f.writelines(['\\providecommand{\\bbobpprldistrlegendtwofixed}[1]{\n', pprldistrtwo, '\n}\n'])

    pptable2Legend = pptable2.table_caption_expensive.replace('\\algorithmA', 'algorithmA')
    pptable2Legend = pptable2Legend.replace('\\algorithmB', 'algorithmB')    
    pptable2Legend = pptable2Legend.replace('\\algorithmAshort', 'algorithmAshort')    
    pptable2Legend = pptable2Legend.replace('\\algorithmBshort', 'algorithmBshort')    
    f.writelines(['\\providecommand{\\bbobpptablestwolegendexpensive}[1]{\n', pptable2Legend, '\n}\n'])
    pptable2Legend = pptable2.table_caption.replace('\\algorithmA', 'algorithmA')
    pptable2Legend = pptable2Legend.replace('\\algorithmB', 'algorithmB')    
    pptable2Legend = pptable2Legend.replace('\\algorithmAshort', 'algorithmAshort')    
    pptable2Legend = pptable2Legend.replace('\\algorithmBshort', 'algorithmBshort')    
    f.writelines(['\\providecommand{\\bbobpptablestwolegend}[1]{\n', pptable2Legend, '\n}\n'])

    f.write(header)    

    f.write(prepare_item('bbobloglosstablecaption'))
    f.write(prepare_item('bbobloglossfigurecaption'))
    f.write(prepare_item('bbobpprldistrlegendrlbased'))
    f.write(prepare_item('bbobpprldistrlegendfixed'))
    f.write(prepare_item('bbobppfigdimlegendrlbased'))
    f.write(prepare_item('bbobppfigdimlegendfixed'))
    f.write(prepare_item('bbobpptablecaption'))

    f.write(prepare_item('bbobppfigslegendrlbased'))
    f.write(prepare_item('bbobppfigslegendfixed'))
    f.write(prepare_item('bbobppfigslegendend', param = '$f_1$ and $f_{24}$'))
    
    f.write(prepare_item('bbobppscatterlegendrlbased'))
    f.write(prepare_item('bbobppscatterlegendfixed', param = '$f_1$ - $f_{24}$'))
    f.write(prepare_item('bbobppscatterlegendend'))
    
    f.write(prepare_item('bbobpprldistrlegendtworlbased'))
    f.write(prepare_item('bbobpprldistrlegendtwofixed'))
    
    f.write(prepare_item('bbobpptablestwolegendexpensive', param = '48'))
    f.write(prepare_item('bbobpptablestwolegend', param = '48'))

    for dim in ['5', '20']:
        for command_name in ['bbobECDFslegendrlbased', 'bbobECDFslegendstandard']:
            f.write(prepare_item(command_name + dim, command_name, str(dim)))
        for command_name in ['bbobpptablesmanylegend', 'bbobpptablesmanylegendexpensive']:
            f.write(prepare_item(command_name + dim, command_name, 'dimension ' + dim))

    f.write('\n\#\#\#\n\\end{document}\n')

def prepare_item(name, command_name = '', param = ''):
    
    if not command_name:
        command_name = name
        
    return '\#\#%s\#\#\n\\%s{%s}\n' % (name, command_name, param)


if __name__ == '__main__':
    main()

