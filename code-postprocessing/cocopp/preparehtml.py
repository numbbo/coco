# -*- coding: utf-8 -*-
"""
Prepares the figure and table descriptions in html.

"""
import os
import sys
import subprocess
import warnings

try:
    from . import preparetexforhtml, genericsettings
except:
    from cocopp import preparetexforhtml, genericsettings


def main(args):

    validation = len(args) > 0 and '-v' in args
    if validation:
        validate_html()
    else:
        texFile = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               genericsettings.latex_commands_for_html + '.tex')
        prepare_html(texFile)


def prepare_html(texFile):


    preparetexforhtml.main(texFile)

    FNULL = open(os.devnull, 'w')
    args = "pdflatex %s" % texFile

    # subprocess.call(args.split(), stdout=FNULL, stderr=FNULL, shell=False)
    subprocess.call(args.split())        
    
    print('pdflatex done')
    
    if ('win32' in sys.platform):
        tthFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tth\\tth.exe')
    elif 'darwin' in sys.platform:
        tthFile = 'tth'
        try:
            import shutil
            if shutil.which(tthFile) is None:
                raise FileNotFoundError
        except Exception:
            warnings.warn('tth on found, run "brew install tth" and try again')
    else:
        tthFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tth_C/tth')

    args = "%s %s" % (tthFile, texFile)
    subprocess.call(args.split(), stdout=FNULL, stderr=FNULL, shell=False)    

    print('tth.exe call done')


def validate_html():

    original_file_name = genericsettings.latex_commands_for_html
    comparing_file_name = original_file_name + '_compare'

    texFile = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           comparing_file_name + '.tex')

    prepare_html(texFile)

    original_html = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 original_file_name + '.html')
    generated_html = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  comparing_file_name + '.html')

    footer_start = '<small>'
    original_lines = list(open(original_html, 'r'))
    generated_lines = list(open(generated_html, 'r'))
    for i in range(0, len(original_lines)):
        original_line = original_lines[i]
        generated_line = generated_lines[i]
        if footer_start in original_line and footer_start in generated_line:
            break

        if not original_line.strip() == generated_line.strip():
            print('Validation failed! File %s.html should be regenerated!' % original_file_name)
            print('Original line: %s' % original_line)
            print('Generated line: %s' % generated_line)
            break

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # To avoid window popup and use without X forwarding

    args = sys.argv[1:] if len(sys.argv) else []
    main(args)
