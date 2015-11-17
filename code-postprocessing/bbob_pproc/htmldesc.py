#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepares the descriptions of images and tables which will be converted to html.

This module creates a tex file with all the descriptions of the images and tables.

"""

import os

from bbob_pproc import genericsettings

# Initialization

descriptions = dict()

def getValue(key):
    """Gets the value for the given key. 

    """

    if (not descriptions):

        htmlFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), genericsettings.latex_commands_for_html + '.html')
        
        if not os.path.isfile(htmlFile):
            return ''
            
        with open(htmlFile) as f:
            content = f.readlines()
        
        currentKey = ''
        currentValue = []
        for line in content:
            if line.startswith('###'):
                break
            
            if line.startswith('##'):
                if currentKey:
                    descriptions[currentKey] = ' '.join(currentValue)
                currentKey = line.strip()
                currentValue = []
            elif not currentKey:
                continue
            else:
                currentValue.append(line.strip())                
    
        if currentKey:
            descriptions[currentKey] = ' '.join(currentValue)

    return descriptions[key] if key in descriptions else ''
    

