#! /usr/bin/env python 
"""copy latex and related files from build folder "build_origin" and zip them.

By default, ``build_origin==../build/latex``. 

Details: this script calls `os.system` with *nix-like commands. It does not
change any files which are part of the repository. 

Related: the file ``*sub?.zip`` is a submission "tag". 

"""
import os

this_dir = 'arxiv'
build_origin = '../build/latex'  # this script lies in folder this_dir

c = os.system
wd = os.getcwd

if __name__ == '__main__':
    entry_folder = wd()
    if not entry_folder.endswith(this_dir):
        os.chdir(this_dir)
    working_folder = wd()
    # we could make a try finally block here
    os.chdir(build_origin)
    files = os.listdir(wd())
    name = None
    for file in files:
        if file.endswith('.tex'):
            if name is not None:
                raise ValueError('Cannot decide to take name %s or %s' % (name, file))
            name = file.split('.')[0]
    c('cp *.tex *.sty *.pdf %s' % working_folder)
    os.chdir(working_folder)
    c('zip -r %s.zip *.tex *.sty *.pdf' % name)
    os.chdir(entry_folder)

