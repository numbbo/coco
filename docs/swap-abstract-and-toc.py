#! /usr/bin/env python 
"""In effect, this script moves ``\\tableofcontents`` behind the abstract and
calls ``pdflatex`` 4 times. 

The file to work upon is the first input argument. 

Details: First, all ``\\tableofcontents`` are incommented, second, the command 
is written directly behind ``\end{abstract}``. 

"""

from __future__ import absolute_import, print_function
import os, sys
from subprocess import call, check_output, CalledProcessError

folder = os.path.join('build', 'latex')
filename = 'coco-doc.tex'


def condition1(old, line):
    return old in line
def change1(line, old, new):
    """replace old with new if old in line. """
    return line.replace(old, new)

def condition2(old, line):
    return line.startswith(old)
def change2(line, old, new):
    """replace line with new if line.startswith(old)"""
    if line.startswith(old):
        return new + "\n"
    return line

condition = condition1
change = change1

def git_revision():
    nb_commits = check_output('git rev-list --count --first-parent HEAD'.split())
    last_commit_id = check_output('git describe --always'.split())
    return '-'.join(['rev', nb_commits.strip(), last_commit_id.strip()])
    
def main(old, new, *files):
    """replace str `old` with str `new` in each of `files`."""
    global condition
    global change
    if old.startswith("line.startswith."):
        condition = condition2  # effects only console output
        change = change2
        old = old.split('.')[2]
        print('replace lines starting with "%s" with "%s"' % (old, new))
    else:
        print('replacing ' + old + ' with ' + new)

    counter = 0
    found = 0
    p = os.path
    for filename in files:
        # print(filename)
        counter += 1
        tfilename = p.join(p.dirname(filename), '__tmp__' + 
                           p.split(filename)[-1] + '__tmp__');
        os.rename(filename, tfilename)
        with open(filename, 'a') as fp: # a is just in case
            for line in open(tfilename):
                if condition(old, line):
                    found += 1
                fp.write(change(line, old, new))
        sys.stdout.flush()  # for print
    print(counter, 'files visited,', found, 'times replaced')


if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print(__doc__)
    else:
        file = sys.argv[1]
        folder = os.path.dirname(file)
        filename = os.path.split(file)[-1]
        
        try:
            rev = "%git:" + git_revision()
        except:
            rev = ""
        
        main(r'\tableofcontents', r'%\tableofcontents ' + rev, file)
        # main(r'%%\tableofcontents', r'%\tableofcontents', file)
        main(r'\end{abstract}', r'\end{abstract}\tableofcontents ' + rev, file)
        # main(r'\tableofcontents%\tableofcontents', r'\tableofcontents', file)
        main(r'\author{', r'\date{\vspace{-1ex}}\author{', file)
    
        oldwd = os.getcwd()
        try:
            os.chdir(folder)
            for i in range(4):
                call(['pdflatex', filename]), 
                # output = check_output(['pdflatex', file]), 
                                        #stderr=sys.stdout, 
                                        #env=os.environ, 
                                        #universal_newlines=True)
                # print(output)
        except CalledProcessError as e:
            print("ERROR: return value=%i" % e.returncode)
            print(e.output)
            raise
        finally:
            os.chdir(oldwd)
    
