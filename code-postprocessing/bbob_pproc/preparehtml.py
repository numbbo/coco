# -*- coding: utf-8 -*-
"""
Prepares the figure and table descriptions in html.

"""
import os
import sys
import subprocess

# Add the path to bbob_pproc
if __name__ == "__main__":
    (filepath, filename) = os.path.split(sys.argv[0])
    sys.path.append(os.path.join(filepath, os.path.pardir))
    import matplotlib
    matplotlib.use('Agg') # To avoid window popup and use without X forwarding

from bbob_pproc import preparetexforhtml, genericsettings

def main():

    preparetexforhtml.main()

    texFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), genericsettings.latex_commands_for_html + '.tex')

    FNULL = open(os.devnull, 'w')
    args = "pdflatex %s" % texFile
    subprocess.call(args.split(), stdout=FNULL, stderr=FNULL, shell=False)    
    
    tthFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tth\\tth.exe')
    args = "%s %s" % (tthFile, texFile)
    subprocess.call(args.split(), stdout=FNULL, stderr=FNULL, shell=False)    

if __name__ == '__main__':
    main()

