            # -*- coding: utf-8 -*-

import sys
import os
import re
import argparse
import numpy as np
import json
import commands as cmd

from utils import *


def clear(languages, out_path, _all=0):
    if _all:
        cmd.getstatusoutput('rm -rf ' + out_path + '/*')

    for l in languages:
        cmd.getstatusoutput('rm -rf ' + os.path.join(out_path, l) + '*')
        cmd.getstatusoutput('mkdir ' + os.path.join(out_path, l))


def import_languages(config, languages, in_path, out_path):
    current_id = 0
    subfolders = cmd.getoutput('ls -1 ' + in_path).split('\n')
    subfolders = [os.path.join(in_path, sf) for sf in subfolders]

    lang_folders = {}
    for l in config['languages']:
        lang_folders[l] = []

    for sf in subfolders:
        aux = get_configuration(os.path.join(sf, 'contents.json'))
        print os.path.join(sf, 'contents.json')
        aux['language'] = config['languages_inv'][aux['language'].lower()]
        aux['path'] = sf
        lang_folders[aux['language']].append(aux)

    i = '0'
    for l in languages:
        lf_name = ' ' + out_path + '/' + l + '/'
        truth = []
        for lf in lang_folders[l]:
            lf['index'] = i
            cp_path = 'cp -R ' + lf['path'] + '/'
            copies = ';'.join([cp_path + p + lf_name + p + '_' + i \
                                for p in lf['problems']])
            cmd.getoutput(copies)
            truth.append(cmd.getoutput('awk \'$1=$1\"_' + i + '\"\' '\
                            + lf['path'] + '/truth.txt')[3:].replace('\r', ''))
            i = str(int(i) + 1)
        f = open('' + out_path + '/' + l + '_truth.txt', 'w')
        f.write('\n'.join(truth))
        f.close()

    #clean
    cmd.getoutput('rm -rf *.DS_Store')

    f = open('' + out_path + '/contents.json', 'w')
    f.write(json.dumps(lang_folders))
    f.close()
