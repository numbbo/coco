#!/usr/bin/env python2.7
## -*- mode: python -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import re
from os import path

class Amalgator:
    def __init__(self, destination_file, release):
        self.release = release
        self.included_files = []
        self.destination_fd = open(destination_file, 'w')
        self.destination_fd.write("""
/************************************************************************
 * WARNING
 *
 * This file is an auto-generated amalgamation. Any changes made to this
 * file will be lost when it is regenerated!
 ************************************************************************/

""")

    def finish(self):
        self.destination_fd.close()

    def process_file(self, filename):
        if filename in self.included_files:
            return
        self.included_files.append(filename)
        fd = open(filename)
        line_number = 1
        if not self.release:
            self.destination_fd.write("#line %i \"%s\"\n" % (line_number, filename))
        for line in fd.readlines():
            ## Is this an include statement?
            matches = re.match("#include \"(.*)\"", line)
            if matches:
                include_file = path.join(path.dirname(filename), matches.group(1))
                ## Has this file not been included previously?
                if not include_file in self.included_files:
                    self.process_file(include_file)
                if not self.release:
                    self.destination_fd.write("#line %i \"%s\"\n" % 
                                              (line_number + 1, filename))
            else:
                self.destination_fd.write(line)
            line_number = line_number + 1
        fd.close()

def amalgamate(source_files, destination_file, release=False):
    print("AML\t%s -> %s" % (str(source_files), destination_file))
    amalgator = Amalgator(destination_file, release)
    for filename in source_files:
        amalgator.process_file(filename)
    amalgator.finish()
