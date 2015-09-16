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

# Initialization

header = """

\\documentclass{article}

% Packages
\\usepackage{graphicx}
\\usepackage{xcolor}

% pre-defined commands
\\newcommand{\\DIM}{\ensuremath{\mathrm{DIM}}}
\\newcommand{\\ERT}{\ensuremath{\mathrm{ERT}}}
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
    
    f.writelines(['\\providecommand{\\bbobloglosstablecaption}[1]{\n', pplogloss.table_caption, '\n}\n'])
    f.writelines(['\\providecommand{\\bbobloglossfigurecaption}[1]{\n', pplogloss.figure_caption, '\n}\n'])
    f.writelines(['\\providecommand{\\bbobpprldistrlegendrlbased}[1]{\n', pprldistr.caption_single_rlbased.replace('TO_BE_REPLACED', 'TOBEREPLACED'),  '\n}\n'])
    f.writelines(['\\providecommand{\\bbobpprldistrlegendfixed}[1]{\n', pprldistr.caption_single_fixed.replace('TO_BE_REPLACED', 'TOBEREPLACED'),  '\n}\n'])
    f.writelines(['\\providecommand{\\bbobppfigdimlegendrlbased}[1]{\n', ppfigdim.scaling_figure_caption_rlbased.replace('values_of_interest', 'valuesofinterest'), '\n}\n'])
    f.writelines(['\\providecommand{\\bbobppfigdimlegendfixed}[1]{\n', ppfigdim.scaling_figure_caption_fixed.replace('values_of_interest', 'valuesofinterest'), '\n}\n'])
    f.writelines(['\\providecommand{\\bbobpptablecaption}[1]{\n', pptable.table_caption, '\n}\n'])
    
    f.write(header)    

    f.write(prepare_item('bbobloglosstablecaption'))
    f.write(prepare_item('bbobloglossfigurecaption'))
    f.write(prepare_item('bbobpprldistrlegendrlbased'))
    f.write(prepare_item('bbobpprldistrlegendfixed'))
    f.write(prepare_item('bbobppfigdimlegendrlbased'))
    f.write(prepare_item('bbobppfigdimlegendfixed'))
    f.write(prepare_item('bbobpptablecaption'))

    f.write('\n\#\#\#\n\\end{document}\n')

def prepare_item(name):
    
    return '\#\#%s\#\#\n\\%s{}\n' % (name, name)


if __name__ == '__main__':
    main()

