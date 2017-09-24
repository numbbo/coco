#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Recursively find :file:`info` and :file:`pickle` files within a directory

This module can be called from the shell, it will recursively look for
:file:`info` and :file:`pickle` files in the current working directory::

  $ python -c "from cocopp.findfiles import main; print(main())"

displays found (extracted) files.

TODO: we do not use pickle files anymore.
"""
from __future__ import absolute_import, division, print_function
import os
import sys
import warnings
import tarfile
import zipfile
import hashlib
if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
from .toolsdivers import StringList  # def StringList(list_): return list_
from . import genericsettings

# Initialization


def is_recognized_repository_filetype(filename):
    return (os.path.isdir(filename.strip())
            or filename.find('.tar') > 0
            or filename.find('.tgz') > 0
            or filename.find('.zip') > 0)


def main(directory='.'):
    """Lists "data" files recursively in a given directory, tar files
    are extracted.

    The "data" files have :file:`info` and :file:`pickle` extensions.

    TODO: not only recognize .tar and .tar.gz and .tgz but .zip...

    """

    file_list = list()
    root = ''
    directory = get_directory(directory, True)

    # Search through the directory directory and all its subfolders.
    for root, _dirs, files in os.walk(directory):
        if genericsettings.verbose:
            print('Searching in %s ...' % root)

        for elem in files:
            if elem.endswith('.info') or elem.endswith('.pickle') or elem.endswith('.pickle.gz'):
                file_list.append(os.path.join(root, elem))

    if genericsettings.verbose:
        print('Found %d file(s).' % (len(file_list)))
    if not file_list:
        warnings.warn('Could not find any file of interest in %s!' % root)
    return file_list


def get_directory(directory, extract_files):

    directory = directory.strip()

    # if directory.endswith('.zip'):
    #   archive = zipfile.ZipFile(directory)
    #   for elem in archive.namelist():
    #     if elem.endswith('.info'):
    #       (root,elem) = os.path.split(elem)
    #       filelist = IndexFile(root,elem,archive)
    if not os.path.isdir(directory) and is_recognized_repository_filetype(directory):
        if '.zip' in directory:
            head, tail = os.path.split(directory[:directory.find('.z')])
            dir_name = head + os.sep + genericsettings.extraction_folder_prefix + tail
            # extract only if extracted folder does not exist yet or if it was
            # extracted earlier than last change of archive:
            if extract_files:
                if (not os.path.exists(dir_name)) or (os.path.getmtime(dir_name) < os.path.getmtime(directory)):
                    with zipfile.ZipFile(directory, "r") as zip_ref:
                        # check first on Windows systems if paths are not too long
                        if ('win32' in sys.platform):
                            longest_file_length = max(len(i) for i in zipfile.ZipFile.namelist(zip_ref))
                            if len(dir_name) + longest_file_length > 259:
                                raise IOError(2, 'Some of the files cannot be extracted ' +
                                              'from "%s". The path is too long.' % directory)
                        zip_ref.extractall(dir_name)

                    print('    archive extracted to folder', dir_name, '...')
            directory = dir_name
        else: # i.e. either directory or .tar or zipped .tar
            head, tail = os.path.split(directory[:directory.find('.t')])
            dir_name = head + os.sep + genericsettings.extraction_folder_prefix + tail
            # extract only if extracted folder does not exist yet or if it was
            # extracted earlier than last change of archive:
            if extract_files:
                if (not os.path.exists(dir_name)) or (os.path.getmtime(dir_name) < os.path.getmtime(directory)):
                    tar_file = tarfile.TarFile.open(directory)
                    longest_file_length = max(len(i) for i in tar_file.getnames())
                    if ('win32' in sys.platform) and len(dir_name) + longest_file_length > 259:
                        raise IOError(2, 'Some of the files cannot be extracted ' +
                                      'from "%s". The path is too long.' % directory)

                    tar_file.extractall(dir_name)
                    # TarFile.open handles tar.gz/tgz
                    print('    archive extracted to folder', dir_name, '...')
            directory = dir_name
            # archive = tarfile.TarFile(directory)
            # for elem in archivefile.namelist():
            #    ~ if elem.endswith('.info'):
            #        ~ (root,elem) = os.path.split(elem)
            #        ~ filelist = IndexFile(root,elem,archive)

    return directory


def get_output_directory_sub_folder(args):

    directory = ''
    if isinstance(args, str):
        directory = args.strip().rstrip(os.path.sep)

        if not os.path.isdir(directory) and is_recognized_repository_filetype(directory):
            directory = directory[:directory.find('.t')]

        directory = (directory.split(os.sep)[-1]).replace(genericsettings.extraction_folder_prefix, '')
    else:
        for index, argument in enumerate(args):
            if not os.path.isdir(argument) and is_recognized_repository_filetype(argument):
                argument = argument[:argument.find('.t')]
            argument = argument.split(os.sep)[-1]
            directory += (argument if len(argument) <= 5 else argument[:5]) + '_'
            if index >= 6:
                directory += 'et_al'
                break
        directory = directory.rstrip('_')

    if len(directory) == 0:
        raise ValueError(args)

    return directory


class COCODataArchive(list):
    """[versatile/beta] An interface to retrieve archived COCO data.

    This class "is" a list of names which are relative path/file names
    separated with slashes "/". Each name represents the data from a
    full experiment, benchmarking one algorithm on an entire benchmark
    suite.

    Method `find` helps to find entries matching one or several
    substrings, e.g. a year or a method. `find_indices` returns the
    respective indices instead of the names.

    Method `get` downloads "matching" data if necessary and returns the
    list of absolute data paths which can be passed to `cocopp.main`.

    `get_one` gets only the first match, which will never change when
    the archive grows (currently the archive is still versatile and no
    guaranties are given). It also downloads at most one data set.

    Method `index` is inherited from `list` and finds the index of the
    respective name entry in the archive (exact match only):

    >>> from cocopp import data_archive as cda
    >>> len(cda) > 100
    True

    Get a `list` of already downloaded data full pathnames:

    >>> cda.get('', remote=False)  # doctest:+ELLIPSIS
    [...

    Find something more specific:

    >>> cda.find('auger')[0]
    'bbob/2009/CMA-ESPLUSSEL_auger_noiseless.tgz'
    >>> cda.index('bbob/2009/CMA-ESPLUSSEL_auger_noiseless.tgz')
    5

    >>> data_paths = (cda.get(['auger', '2013'], remote=False)  # understood as AND search
    ...               + cda.get(['hansen', '2010'], remote=False))
    >>> assert len(data_paths) >= 0  # could be any number, because remote was False and archive could be growing
    >>> data_path = cda.get_one(['au', '2009'], remote=False)
    >>> assert data_path is None or str(data_path) == data_path

    where `data_paths` could be empty. Now we can call
    ``cocopp.main(' '.join(data_paths))``.

    Details: most of the matching work is done in `find`. Calls to
    `get`, `get_one`, `find_indices` rely on calling `find` to
    resolve the input matches. The resulting match can always be found
    in the property attribute `names_found`.
    """
    # _all can be fully generated on a synced archive (e.g. rsync'ed) with
    # method _generate_names_list
    _all = [  # name, sha256 hash, size [kB]
            ('bbob/2009/ALPS_hornby_noiseless.tgz', '98810d28d879eb25d87949f3d7666b36f524a591e3c7d16ef89eb2caec02613b') ,
            ('bbob/2009/AMALGAM_bosman_noiseless.tgz', 'e92951f11f8d56e0d1bdea7026bb1087142c3ea054d9e7be44fea2b063c6f394') ,
            ('bbob/2009/BAYEDA_gallagher_noiseless.tgz', 'ed674ff71237cc020c9526b332e2d817d5cd82579920c7ff2d25ff064a57ed33') ,
            ('bbob/2009/BFGS_ros_noiseless.tgz', 'ca9dbeab9f7ecd7d3bb39596a6835a01f832178f57add98d95592143f0095c7a') ,
            ('bbob/2009/BIPOP-CMA-ES_hansen_noiseless.tgz', '6b1276dc15988dc71db0d48525ae8c41781aba8a171079159cdc845cc0f1932d') ,
            ('bbob/2009/CMA-ESPLUSSEL_auger_noiseless.tgz', 'b43aa7576babce52b9380abd1c5b4f1d9fd12386b5c5a548b3230c3d183856fc') ,
            ('bbob/2009/Cauchy-EDA_posik_noiseless.tgz', 'd256677b215fe9a2bfc6f5a2b509b1adc0d76223e142bfe0775e70de9b5609e9') ,
            ('bbob/2009/DASA_korosec_noiseless.tgz', '2b98fbf25a6c92b597eb16b061aaf234a15c084753bae7fed9b6c6a86b5cea1d') ,
            ('bbob/2009/DE-PSO_garcia-nieto_noiseless.tgz', '796e7cf175cc68bd9475927a207dab05abb5c98054bbd852f272418225ddbdae') ,
            ('bbob/2009/DIRECT_posik_noiseless.tgz', '5cfe3e57d847a43d2b3e770fa81ffd462fdedfa38d284855f45930c28068f66f') ,
            ('bbob/2009/EDA-PSO_el-abd_noiseless.tgz', '0c97b91b9fd9656ca7ffba77449d9de3888e0be3fbe5bf515bb3dc00de47d8bd') ,
            ('bbob/2009/FULLNEWUOA_ros_noiseless.tgz', '5edafe995cd2bd9c02233c638bf61bb185993aee15c92b390231eb9d036ab236') ,
            ('bbob/2009/G3PCX_posik_noiseless.tgz', 'c9a943f839dccb9ef418adf0ad5e506eed2580866bd1bdb20b55f97e53608fcc') ,
            ('bbob/2009/GA_nicolau_noiseless.tgz', 'a49bec35b95916afdfa07c60c903cb40f316efba85aeb98fa92589dea084027b') ,
            ('bbob/2009/GLOBAL_pal_noiseless.tgz', 'eeeae4a60ab7e86bc27cc1772465ce4ac0a13961698b8e9a2071b9a06c282a80') ,
            ('bbob/2009/IPOP-SEP-CMA-ES_ros_noiseless.tgz', '8fc1af860fd3d46f4dcad70923f3ac610bf2542f306c9157a03494f3bd7630ef') ,
            ('bbob/2009/LSfminbnd_posik_noiseless.tgz', '2fbf1087a390a921e94da119644ebf87784c54267c745a1f0bb541ea97df78f6') ,
            ('bbob/2009/LSstep_posik_noiseless.tgz', 'a29629ea5c9b8d57d74ea6654d78c85539aa4e210ac1fb33f9d7cdae0095615b') ,
            ('bbob/2009/MA-LS-CHAIN_molina_noiseless.tgz', 'c9b354c892c167d377f45a68df0c09fb74c9b4c3c665a4498a3f62fde3e977ca') ,
            ('bbob/2009/MCS_huyer_noiseless.tgz', '14060fdefda9641f90c09f0f20c97ff7301f254811515d613515a26132b7502c') ,
            ('bbob/2009/NELDERDOERR_doerr_noiseless.tgz', '01db00bfaee07c26b5ea1b579b7760f32e245e5802e99b68f48296236af94e3d') ,
            ('bbob/2009/NELDER_hansen_noiseless.tgz', '3c05507a05a4ed5c98a451b003907dfb60d03bd270f78475901815e0b1f19240') ,
            ('bbob/2009/NEWUOA_ros_noiseless.tgz', '72e2b65d2ab6dbe3dfe9d8355d844bc6ea69e0c9cac07fa28f5f35e4939c1502') ,
            ('bbob/2009/ONEFIFTH_auger_noiseless.tgz', '0c260fa02a6d709ae69411ba887a122b48b9e3e94c032345ecd4cf6c2e0f5886') ,
            ('bbob/2009/POEMS_kubalik_noiseless.tgz', '2eb3ad99a56f1f5e122c861b477265d90e34cfdce0b5a6436e03c468a3da1daa') ,
            ('bbob/2009/PSO_Bounds_el-abd_noiseless.tgz', '34b11a175657dfcb199883207923dbba5e7225a0e842d195250abf67d1c87a78') ,
            ('bbob/2009/PSO_el-abd_noiseless.tgz', '206f1a618e35a7723ec0747aad40fd10001b5fbb2785910fa155b60f9c47f81a') ,
            ('bbob/2009/RANDOMSEARCH_auger_noiseless.tgz', '2c76caf6a069a2b5b2c32441c0fb5b267aa4d5cde8b8c40de5baaa06c16a33cf') ,
            ('bbob/2009/Rosenbrock_posik_noiseless.tgz', 'b74cf1e9909a5322c4f111fe5c9c5e5b147d6a25f428391e2de6a27efbd0a8f8') ,
            ('bbob/2009/VNS_garcia-martinez_noiseless.tgz', '3a9ba4c6305ef0216537684da1033ab1342cdd16d523a5a8c74dc9d20743729d') ,
            ('bbob/2009/iAMALGAM_bosman_noiseless.tgz', '02cfa688d20710d90243be6b57f874d8b45386189da5fcb22ecaf01bb88f7564') ,
            ['bbob/2010/1komma2_brockhoff_noiseless.tar.gz'],
            ['bbob/2010/1komma2mir_brockhoff_noiseless.tar.gz'],
            ['bbob/2010/1komma2mirser_brockhoff_noiseless.tar.gz'],
            ['bbob/2010/1komma2ser_brockhoff_noiseless.tar.gz'],
            ['bbob/2010/1komma4_brockhoff_noiseless.tar.gz'],
            ['bbob/2010/1komma4mir_brockhoff_noiseless.tar.gz'],
            ['bbob/2010/1komma4mirser_brockhoff_noiseless.tar.gz'],
            ['bbob/2010/1komma4ser_brockhoff_noiseless.tar.gz'],
            ['bbob/2010/1plus1_brockhoff_noiseless.tar.gz'],
            ['bbob/2010/1plus2mirser_brockhoff_noiseless.tar.gz'],
            ['bbob/2010/ABC_elabd_noiseless.tar.gz'],
            ['bbob/2010/AVGNEWUOA_ros_noiseless.tar.gz'],
            ['bbob/2010/CMAEGS_finck_noiseless.tar.gz'],
            ['bbob/2010/DE-F-AUC_fialho_noiseless.tar.gz'],
            ['bbob/2010/DEuniform_fialho_noiseless.tar.gz'],
            ['bbob/2010/IPOP-ACTCMA-ES_ros_noiseless.tar.gz'],
            ['bbob/2010/IPOP-CMA-ES_ros_noiseless.tar.gz'],
            ['bbob/2010/MOS_torre_noiseless.tar.gz'],
            ['bbob/2010/NBC-CMA_preuss_noiseless.tar.gz'],
            ['bbob/2010/PM-AdapSS-DE_fialho_noiseless.tar.gz'],
            ['bbob/2010/RCGA_tran_noiseless.tar.gz'],
            ['bbob/2010/SPSA_finck_noiseless.tar.gz'],
            ['bbob/2010/oPOEMS_kubalic_noiseless.tar.gz'],
            ['bbob/2010/pPOEMS_kubalic_noiseless.tar.gz'],
            ['bbob/2012/ACOR_liao_noiseless.tgz'],
            ['bbob/2012/BIPOPaCMA_loshchilov_noiseless.tgz'],
            ['bbob/2012/BIPOPsaACM_loshchilov_noiseless.tgz'],
            ['bbob/2012/CMAES_posik_noiseless.tgz'],
            ['bbob/2012/CMA_brockhoff_noiseless.tgz'],
            ['bbob/2012/CMAa_brockhoff_noiseless.tgz'],
            ['bbob/2012/CMAm_brockhoff_noiseless.tgz'],
            ['bbob/2012/CMAma_brockhoff_noiseless.tgz'],
            ['bbob/2012/CMAmah_brockhoff_noiseless.tgz'],
            ['bbob/2012/CMAmh_brockhoff_noiseless.tgz'],
            ['bbob/2012/DBRCGA_chuang_noiseless.tgz'],
            ['bbob/2012/DE-AUTO_voglis_noiseless.tgz'],
            ['bbob/2012/DE-BFGS_voglis_noiseless.tgz'],
            ['bbob/2012/DE-ROLL_voglis_noiseless.tgz'],
            ['bbob/2012/DE-SIMPLEX_voglis_noiseless.tgz'],
            ['bbob/2012/DEAE_posik_noiseless.tgz'],
            ['bbob/2012/DE_posik_noiseless.tgz'],
            ['bbob/2012/DEb_posik_noiseless.tgz'],
            ['bbob/2012/DEctpb_posik_noiseless.tgz'],
            ['bbob/2012/IPOPsaACM_loshchilov_noiseless.tgz'],
            ['bbob/2012/JADE_posik_noiseless.tgz'],
            ['bbob/2012/JADEb_posik_noiseless.tgz'],
            ['bbob/2012/JADEctpb_posik_noiseless.tgz'],
            ['bbob/2012/MVDE_melo_noiseless.tgz'],
            ['bbob/2012/NBIPOPaCMA_loshchilov_noiseless.tgz'],
            ['bbob/2012/NIPOPaCMA_loshchilov_noiseless.tgz'],
            ['bbob/2012/PSO-BFGS_voglis_noiseless.tgz'],
            ['bbob/2012/SNES_schaul_noiseless.tgz'],
            ['bbob/2012/xNES_schaul_noiseless.tgz'],
            ['bbob/2012/xNESas_schaul_noiseless.tgz'],
            ['bbob/2013/BIPOP-aCMA-STEP_loshchilov_noiseless.tgz'],
            ['bbob/2013/BIPOP-saACM-k_loshchilov_noiseless.tgz'],
            ['bbob/2013/CGA-grid16_holtschulte_noiseless.tgz'],
            ['bbob/2013/CGA-grid100_holtschulte_noiseless.tgz'],
            ['bbob/2013/CGA-ring16_holtschulte_noiseless.tgz'],
            ['bbob/2013/CGA-ring100_holtschulte_noiseless.tgz'],
            ['bbob/2013/CMAES_Hutter_hutter_noiseless.tgz'],
            ['bbob/2013/DE_Pal_pal_noiseless.tgz'],
            ['bbob/2013/GA-100_holtschulte_noiseless.tgz'],
            ['bbob/2013/HCMA_loshchilov_noiseless.tgz'],
            ['bbob/2013/HILL_holtschulte_noiseless.tgz'],
            ['bbob/2013/HMLSL_pal_noiseless.tgz'],
            ['bbob/2013/IP-10DDr_liao_noiseless.tgz'],
            ['bbob/2013/IP-500_liao_noiseless.tgz'],
            ['bbob/2013/IPOP400D_auger_noiseless.tgz'],
            ['bbob/2013/IP_liao_noiseless.tgz'],
            ['bbob/2013/MEMPSODE_voglis_noiseless.tgz'],
            ['bbob/2013/MLSL_pal_noiseless.tgz'],
            ['bbob/2013/OQNLP_pal_noiseless.tgz'],
            ['bbob/2013/P-DCN_tran_noiseless.tgz'],
            ['bbob/2013/P-zero_tran_noiseless.tgz'],
            ['bbob/2013/PRCGA_sawyerr_noiseless.tgz'],
            ['bbob/2013/SMAC-BBOB_hutter_noiseless.tgz'],
            ['bbob/2013/U-DCN_tran_noiseless.tgz'],
            ['bbob/2013/U-zero_tran_noiseless.tgz'],
            ['bbob/2013/fmincon_pal_noiseless.tgz'],
            ['bbob/2013/fminunc_pal_noiseless.tgz'],
            ['bbob/2013/lmm-CMA-ES_auger_noiseless.tgz'] ,
            ['bbob/2013/simplex_pal_noiseless.tgz'],
            ['bbob/2013/tany_liao_noiseless.tgz'],
            ['bbob/2013/texp_liao_noiseless.tgz'],
            ['bbob/2015-CEC/MATSUMOTO-Brockhoff-noiseless.tgz'],
            ['bbob/2015-CEC/R-DE-10e2-Tanabe-noiseless.tgz'],
            ['bbob/2015-CEC/R-DE-10e5-Tanabe-noiseless.tgz'],
            ['bbob/2015-CEC/R-SHADE-10e2-Tanabe-noiseless.tgz'],
            ['bbob/2015-CEC/R-SHADE-10e5-Tanabe-noiseless.tgz'],
            ['bbob/2015-CEC/RL-SHADE-10e2-Tanabe-noiseless.tgz'],
            ['bbob/2015-CEC/RL-SHADE-10e5-Tanabe-noiseless.tgz'],
            ['bbob/2015-CEC/SOO-Derbel-noiseless.tgz'],
            ['bbob/2015-GECCO/BSif.tgz'],
            ['bbob/2015-GECCO/BSifeg.tgz'],
            ['bbob/2015-GECCO/BSqi.tgz'],
            ['bbob/2015-GECCO/BSrr.tgz'],
            ['bbob/2015-GECCO/CMA-CSA.tgz'],
            ['bbob/2015-GECCO/CMA-MSR.tgz'],
            ['bbob/2015-GECCO/CMA-TPA.tgz'],
            ['bbob/2015-GECCO/GP1-CMAES-2013instances.tgz'],
            ['bbob/2015-GECCO/GP1-CMAES.tgz'],
            ['bbob/2015-GECCO/GP5-CMAES-2013instances.tgz'],
            ['bbob/2015-GECCO/GP5-CMAES.tgz'],
            ['bbob/2015-GECCO/IPOPCMAv3p61-2013instances.tgz'],
            ['bbob/2015-GECCO/IPOPCMAv3p61.tgz'],
            ['bbob/2015-GECCO/LHD-2xDefault-MATSuMoTo.tgz'],
            ['bbob/2015-GECCO/LHD-10xDefault-MATSuMoTo.tgz'],
            ['bbob/2015-GECCO/RAND-2xDefault-MATSuMoTo.tgz'],
            ['bbob/2015-GECCO/RF1-CMAES-2013instances.tgz'],
            ['bbob/2015-GECCO/RF1-CMAES.tgz'],
            ['bbob/2015-GECCO/RF5-CMAES-2013instances.tgz'],
            ['bbob/2015-GECCO/RF5-CMAES.tgz'],
            ['bbob/2015-GECCO/Sif.tgz'],
            ['bbob/2015-GECCO/Sifeg.tgz'],
            ['bbob/2015-GECCO/Srr.tgz'],
            ['bbob/2016/PSAaLmC.tgz'],
            ['bbob/2016/PSAaLmD.tgz'],
            ['bbob/2016/PSAaSmC.tgz'],
            ['bbob/2016/PSAaSmD.tgz'],
            ['bbob/2017/CMAES-APOP.tgz'],
            ['bbob/2017/DTS-CMA-ES.tgz'],
            ['bbob/2017/EvoSpace-PSO-GA.tgz'],
            ['bbob/2017/KL-BIPOP.tgz'],
            ['bbob/2017/KL-IPOP.tgz'],
            ['bbob/2017/KL-Restart.tgz'],
            ['bbob/2017/Ord-Q-DTS.tgz'],
            ['bbob/2017/SSEABC.tgz'],
            ['bbob/2017-outsideGECCO/RS4-1e7D.tgz'],
            ['bbob/2017-outsideGECCO/RS4p5-1e7D.tgz'],
            ['bbob/2017-outsideGECCO/RS5-1e7D.tgz'],
            ['bbob-noisy/2009/BFGS_ros_noisy.tgz'],  # selectively added for tests
            ['bbob-noisy/2009/MCS_huyer_noisy.tgz'],
            ['bbob-biobj/2016/GA-MULTIOBJ-NSGA-II.tgz'],
            ['bbob-biobj/2016/RS-4.tgz'],  # selectively added for tests
            ['bbob-biobj/2016/RS-100.tgz'],
	        ['test/N-II.tgz'],
            ['test/RS-4.zip'],
    ]
    def __init__(self,
                 local_path='~/.cocopp/data-archive',
                 url='http://coco.gforge.inria.fr/data-archive'):
        """Arguments are a full local path and an URL.

        ``~`` refers to the user home folder. By default the archive is
        hosted at ``'~/.cocopp/data-archive'``.
        """
        # extract names (first column) from _all
        list.__init__(self, (entry[0] for entry in COCODataArchive._all))
        self.local_data_path = os.path.expanduser(os.path.join(*local_path.split('/')))
        self.remote_data_path = url
        self._names_found = []  # names recently found
        self.print_ = print  # like this we can make it quiet for testing
    @property
    def names_found(self):
        """names as (to be) used in `get` when called without argument.

        `names_found` is set in the `get` or `find` methods when called
        with a search or index argument.

        """
        return self._names_found

    def find(self, *substrs):
        """return names of archived data that match all `substrs`.

        This method serves for interactive exploration of available data.

        When given several `substrs` arguments the result matches each
        substring (AND search). Upper/lower case is ignored.

        When given a single `substrs` argument, it may be

        - a list of matching substrings, used as several substrings above
        - an index of `type` `int`
        - a list of indices

        Returned names correspond to the unique trailing subpath of data
        filenames. The next call to `get` without argument will
        retrieve the found data and return a `list` of full data paths
        which can be used with `cocopp.main`.

        The single input argument can be used as is also in `get`.

        Example::

            >>> import cocopp
            >>> cocopp.data_archive.find('Auger', '2013')[1]
            'bbob/2013/lmm-CMA-ES_auger_noiseless.tgz'

        Details: The list of matching names is stored in `current_names`.
        """
        # check whether the first arg is a list rather than a str
        if substrs and len(substrs) == 1 and substrs[0] != str(substrs[0]):
            substrs = substrs[0]  # we may now have a list of str as expected
            if isinstance(substrs, int):  # or maybe just an int
                self._names_found = [self[substrs]]
                return StringList(self._names_found)
            elif substrs and isinstance(substrs[0], int):  # or a list of indices
                self._names_found = [self[i] for i in substrs]
                return StringList(self._names_found)
        names = list(self)
        for s in substrs:
            names = [name for name in names if s.lower() in name.lower()]
        self._names_found = names
        return StringList(names)

    def find_indices(self, *substrs):
        """same as `find` but returns indices instead of names"""
        return StringList([self.index(name) for name in self.find(*substrs)])

    def get_one(self,  substrs=None, remote=True):
        """get the first match of `substrs` in the archived data.

        Return the absolute pathname if a name matches all substrings.

        Argument `substrs` can be a list of substrings or a single
        substring and is passed to `find`.

        If no match is found, `None` is returned.

        If ``substrs is None`` (default), the result from the last
        ``find*`` or ``get*`` is used like `self.current_names[0]``.

        When successful, `get_one` guaranties, like ``find(...)[0]``
        and in contrast to `get`, a stable result even when the data
        base grows.

        See also `get`.
        """
        if substrs is not None:
            self.find(substrs)
        if not self.names_found:
            return None
        # name is in archive, but with remote=False we may still end up with nothing
        res = self.get(self.names_found[0], remote=remote)
        assert len(res) == 1 or not remote
        return res[0] if res else None

    def get(self, substrs=None, remote=True):
        """get matching archived data to be used as argument to `cocopp.main`.

        Return a list of absolute pathnames.

        `substrs` may be a single substring or a list of matching
        substrings used as argument to `find`.

        `substrs` may also be an index (as `int`) or list of indices.

        If ``substrs is None`` (default), the last result of `find` is
        used.

        ``get('')`` matches everything and may download the entire archive
        before it returns all known data paths.

        If ``remote is True`` (default), the respective data are
        downloaded from the remote location if necessary.
        """
        if not isinstance(remote, bool):
            raise ValueError(
                "Second argument to `COCODataArchive.get` must be a "
                "`bool`,\n that is either `True` or `False`. "
                "Use a `list` of `str` to define several substrings.")
        if substrs is not None:
            self.find(substrs)  # set self._names_found as desired

        full_names = []
        for name in self.names_found:
            full_name = self.full_path(name)
            if os.path.exists(full_name):
                self.check_hash(name)
                full_names.append(full_name)
                continue
            if not remote:
                if 22 < 3:
                    warnings.warn('name locally not found by COCODataArchive (consider option "remote=True"):'
                          ' %s' % full_name)
                continue
            if not os.path.exists(os.path.split(full_name)[0]):
                os.makedirs(os.path.split(full_name)[0])
            url = '/'.join((self.remote_data_path, name))
            self.print_("downloading %s to %s" % (url, full_name))
            urlretrieve(url, full_name)
            self.check_hash(name)
            full_names.append(full_name)
        return StringList(full_names)

    def full_path(self, name):
        """return full local path of `name` or of a full path, idempotent
        """
        if name.startswith(self.local_data_path):
            return name
        return os.path.join(self.local_data_path,
                            os.path.join(*name.split('/')))

    def name(self, full_path):
        """return name of `full_path`, idempotent.

        If `full_path` is not from the data archive, the outcome is
        undefined.

        >>> from cocopp import data_archive as cda
        >>> for path in cda.get("", remote=False):  # path to all locally available data
        ...     name = cda.name(path)
        ...     assert cda.name(name) == name, "method `name` is not idempotent on '%s'" % name
        ...     assert cda.count(name) == 1, "%s counted %d times in data archive" % (name, cda.count(name))

        Check that all names are only once found in the data archive:

        >>> for name in cda:
        ...     assert cda.count(name) == 1, "%s counted %d times in data archive" % (name, cda.count(name))
        ...     assert len(cda.find(name)) == 1, "%s found %d times" % (name, cda.find(name))

        """
        if full_path.startswith(self.local_data_path):
            name = full_path[len(self.local_data_path) + 1:]
        else:
            name = full_path
        name = name.replace(os.path.sep, '/')  # this may not be a 100% fix
        if name not in self:
            warnings.warn('name "%s" not in COCODataArchive' % name)
        return name

    def check_hash(self, name):
        """warn if hash is unknown, raise ValueError if hashes disagree
        """
        known_hash = self._known_hash(name)
        if known_hash is None:
            warnings.warn(
                'COCODataArchive has no hash checksum for\n  %s\n'
                'The computed checksum was \n  %s\n'
                'To remove this warning, consider to manually insert this hash in `COCODataArchive._all`\n'
                'Or, if this happens for many different data, consider using `_generate_names_list` to\n'
                'compute all hashes of local data and then manually insert the hash in _all.\n'
                'Or consider filing a bug report (issue) at https://github.com/numbbo/coco/issues'
                '' % (name, self._hash(name)))
        elif self._hash(name) != known_hash:
            raise ValueError(
                'wrong checksum for "%s".'
                'Consider to (re)move file\n'
                '   %s\n'
                'as it may be a partial/unsuccessful download.\n'
                'A missing file will be downloaded again by `get`.'
                '' % (name, self.full_path(name)))

    def _hash(self, name, hash_function=hashlib.sha256):
        """compute hash of `name` or path"""
        with open(self.full_path(name) if name in self else
                  name, 'rb') as file_:
            return hash_function(file_.read()).hexdigest()

    def _known_hash(self, name):
        """return known hash or `None`
        """
        try:
            return self._all[self.index(name)][1]
        except IndexError:
            return None

    def _generate_names_list(self, archive_root=None):
        """print an _all list of an existing archive including hashes and
        filesizes.

        This may serve as new/improved/updated/final _all class
        attribute via manual copy-paste.

        May or may not need to be modified under Windows.
        """
        if archive_root is None:
            archive_root = self.local_data_path
        for dirpath, dirnames, filenames in os.walk(archive_root):
            for filename in filenames:
                if '.extracted' not in dirpath and not filename.endswith(('dat', 'info')):
                    name = '/'.join([dirpath.replace(os.path.sep, '/'), filename])[len(archive_root) + 1:]
                    path = os.path.join(dirpath, filename)
                    print("('%s', '%s', %d), " % (
                        name,
                        self._hash(path),
                         os.path.getsize(path) // 1000))  # or os.stat(path).st_size

