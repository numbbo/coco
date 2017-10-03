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
import ast
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

    Calling the class instance (alias to `find`) helps to find entries
    matching one or several substrings, e.g. a year or a method.
    `find_indices` returns the respective indices instead of the names.
    `print` displays both.

    Method `get` downloads the "matching" data set if necessary and
    returns the the absolute data path which can be used with
    `cocopp.main`.

    Method `index` is inherited from `list` and finds the index of the
    respective name entry in the archive (exact match only):

    >>> from cocopp import bbob
    >>> len(bbob) > 150
    True

    >>> bbob[:3]  # doctest:+ELLIPSIS
    ['2009/...
    >>> bbob('2009/bi')[0]  # doctest:+ELLIPSIS
    '...

    Get a `list` of already downloaded data full pathnames:

    >>> [bbob.get(i, remote=False) for i in range(len(bbob))] # doctest:+ELLIPSIS
    [...

    Find something more specific:

    >>> bbob('auger')[0]  # == bbob.find('auger')[0]
    '2009/CMA-ESPLUSSEL_auger_noiseless.tgz'

    corresponds to ``cocopp.main('auger!')``.

    >>> bbob.index('2009/CMA-ESPLUSSEL_auger_noiseless.tgz')  # just list.index
    6

    >>> data_path = bbob.get(bbob(['au', '2009'])[0], remote=False)
    >>> assert data_path is None or str(data_path) == data_path

    These commands may download data, to avoid this the option ``remote=False`` is given:

    >>> ' '.join(bbob.get(i, remote=False) for i in [2, 13, 33])  # can serve as argument to cocopp.main  # doctest:+ELLIPSIS
    '...
    >>> bbob.get_all([2, 13, 33], remote=False).as_string  # is the same  # doctest:+ELLIPSIS
    ' ...
    >>> ' '.join(bbob.get(name, remote=False) for name in [bbob[2], bbob[13], bbob[33]])  # is the same  # doctest:+ELLIPSIS
    '...
    >>> ' '.join(bbob.get(name, remote=False) for name in [
    ...         '2009/BAYEDA_gallagher_noiseless.tgz',
    ...         '2009/GA_nicolau_noiseless.tgz',
    ...         '2010/1komma2mirser_brockhoff_noiseless.tar.gz'])  # is the same  # doctest:+ELLIPSIS
    '...

    """
    # _all can be fully generated on a synced archive (e.g. rsync'ed) with
    # method _generate_names_list
    _all = [  # name, sha256 hash, size [kB]
        ('bbob/2009/ALPS_hornby_noiseless.tgz', '98810d28d879eb25d87949f3d7666b36f524a591e3c7d16ef89eb2caec02613b', 19150),
        ('bbob/2009/AMALGAM_bosman_noiseless.tgz', 'e92951f11f8d56e0d1bdea7026bb1087142c3ea054d9e7be44fea2b063c6f394', 17020),
        ('bbob/2009/BAYEDA_gallagher_noiseless.tgz', 'ed674ff71237cc020c9526b332e2d817d5cd82579920c7ff2d25ff064a57ed33', 12872),
        ('bbob/2009/BFGS_ros_noiseless.tgz', 'ca9dbeab9f7ecd7d3bb39596a6835a01f832178f57add98d95592143f0095c7a', 15026),
        ('bbob/2009/BIPOP-CMA-ES_hansen_noiseless.tgz', '6b1276dc15988dc71db0d48525ae8c41781aba8a171079159cdc845cc0f1932d', 16476),
        ('bbob/2009/Cauchy-EDA_posik_noiseless.tgz', 'd256677b215fe9a2bfc6f5a2b509b1adc0d76223e142bfe0775e70de9b5609e9', 16777),
        ('bbob/2009/CMA-ESPLUSSEL_auger_noiseless.tgz', 'b162d859071467091976877ff3ed46c4f6fd0aea89097a88691f08f5ada1e651', 15154),
        ('bbob/2009/DASA_korosec_noiseless.tgz', '2b98fbf25a6c92b597eb16b061aaf234a15c084753bae7fed9b6c6a86b5cea1d', 14629),
        ('bbob/2009/DE-PSO_garcia-nieto_noiseless.tgz', '796e7cf175cc68bd9475927a207dab05abb5c98054bbd852f272418225ddbdae', 10781),
        ('bbob/2009/DIRECT_posik_noiseless.tgz', '5cfe3e57d847a43d2b3e770fa81ffd462fdedfa38d284855f45930c28068f66f', 2065),
        ('bbob/2009/EDA-PSO_el-abd_noiseless.tgz', '0c97b91b9fd9656ca7ffba77449d9de3888e0be3fbe5bf515bb3dc00de47d8bd', 9864),
        ('bbob/2009/FULLNEWUOA_ros_noiseless.tgz', '5edafe995cd2bd9c02233c638bf61bb185993aee15c92b390231eb9d036ab236', 6324),
        ('bbob/2009/G3PCX_posik_noiseless.tgz', 'c9a943f839dccb9ef418adf0ad5e506eed2580866bd1bdb20b55f97e53608fcc', 15460),
        ('bbob/2009/GA_nicolau_noiseless.tgz', 'a49bec35b95916afdfa07c60c903cb40f316efba85aeb98fa92589dea084027b', 16736),
        ('bbob/2009/GLOBAL_pal_noiseless.tgz', 'eeeae4a60ab7e86bc27cc1772465ce4ac0a13961698b8e9a2071b9a06c282a80', 6819),
        ('bbob/2009/iAMALGAM_bosman_noiseless.tgz', '02cfa688d20710d90243be6b57f874d8b45386189da5fcb22ecaf01bb88f7564', 16899),
        ('bbob/2009/IPOP-SEP-CMA-ES_ros_noiseless.tgz', '8fc1af860fd3d46f4dcad70923f3ac610bf2542f306c9157a03494f3bd7630ef', 15099),
        ('bbob/2009/LSfminbnd_posik_noiseless.tgz', '2fbf1087a390a921e94da119644ebf87784c54267c745a1f0bb541ea97df78f6', 7603),
        ('bbob/2009/LSstep_posik_noiseless.tgz', 'a29629ea5c9b8d57d74ea6654d78c85539aa4e210ac1fb33f9d7cdae0095615b', 6375),
        ('bbob/2009/MA-LS-CHAIN_molina_noiseless.tgz', 'c9b354c892c167d377f45a68df0c09fb74c9b4c3c665a4498a3f62fde3e977ca', 10299),
        ('bbob/2009/MCS_huyer_noiseless.tgz', '14060fdefda9641f90c09f0f20c97ff7301f254811515d613515a26132b7502c', 4374),
        ('bbob/2009/NELDER_hansen_noiseless.tgz', '3c05507a05a4ed5c98a451b003907dfb60d03bd270f78475901815e0b1f19240', 17785),
        ('bbob/2009/NELDERDOERR_doerr_noiseless.tgz', '01db00bfaee07c26b5ea1b579b7760f32e245e5802e99b68f48296236af94e3d', 8078),
        ('bbob/2009/NEWUOA_ros_noiseless.tgz', '72e2b65d2ab6dbe3dfe9d8355d844bc6ea69e0c9cac07fa28f5f35e4939c1502', 13586),
        ('bbob/2009/ONEFIFTH_auger_noiseless.tgz', '0c260fa02a6d709ae69411ba887a122b48b9e3e94c032345ecd4cf6c2e0f5886', 18345),
        ('bbob/2009/POEMS_kubalik_noiseless.tgz', '2eb3ad99a56f1f5e122c861b477265d90e34cfdce0b5a6436e03c468a3da1daa', 7332),
        ('bbob/2009/PSO_Bounds_el-abd_noiseless.tgz', '34b11a175657dfcb199883207923dbba5e7225a0e842d195250abf67d1c87a78', 9933),
        ('bbob/2009/PSO_el-abd_noiseless.tgz', '206f1a618e35a7723ec0747aad40fd10001b5fbb2785910fa155b60f9c47f81a', 9355),
        ('bbob/2009/RANDOMSEARCH_auger_noiseless.tgz', '2c76caf6a069a2b5b2c32441c0fb5b267aa4d5cde8b8c40de5baaa06c16a33cf', 19292),
        ('bbob/2009/Rosenbrock_posik_noiseless.tgz', 'b74cf1e9909a5322c4f111fe5c9c5e5b147d6a25f428391e2de6a27efbd0a8f8', 9323),
        ('bbob/2009/VNS_garcia-martinez_noiseless.tgz', '3a9ba4c6305ef0216537684da1033ab1342cdd16d523a5a8c74dc9d20743729d', 9128),
        ('bbob/2010/1komma2_brockhoff_noiseless.tar.gz', 'b4a03b8ced22decfc52236637d572a4dbf7fc5320451bbfd38d2384c4faad4db', 9996),
        ('bbob/2010/1komma2mir_brockhoff_noiseless.tar.gz', '8135b6933984624ec7594cb880042a72e4d4a4a13b6f729a38bfe9f1398f4d7c', 9863),
        ('bbob/2010/1komma2mirser_brockhoff_noiseless.tar.gz', '78d4aa5e4db9aaf349e0069c50b390bd9c40a816b69f272bf2653853ad1c90d6', 9772),
        ('bbob/2010/1komma2ser_brockhoff_noiseless.tar.gz', 'f19ef264a0fa79249b8eb7ebaa811ca65867ecfd951b0c349c840b9fbc4aacc4', 10017),
        ('bbob/2010/1komma4_brockhoff_noiseless.tar.gz', '22aae2132d36946f0e4a82cb71eccde3549c63452f7f1bb167eac7553c4e1551', 9658),
        ('bbob/2010/1komma4mir_brockhoff_noiseless.tar.gz', '88d763cee3274d440764d68fc09d9a4e130019579a3e3b396bea07fdffa3da41', 9615),
        ('bbob/2010/1komma4mirser_brockhoff_noiseless.tar.gz', '56833eafae3e999cf28f75d7d4ada96b95e3da4981f2ffd29df4ee6ce54691ff', 9479),
        ('bbob/2010/1komma4ser_brockhoff_noiseless.tar.gz', '610dcb0d5654187b509605a3738cf909794de06fb09d2d2f9a8f871a765c331c', 9606),
        ('bbob/2010/1plus1_brockhoff_noiseless.tar.gz', '3a8a2a3d4a1931dd35d6c0c90be43084ab85c51d239acebbe3fe275c094a99c3', 10227),
        ('bbob/2010/1plus2mirser_brockhoff_noiseless.tar.gz', '9e6296b2c3b71a89a8436d4e1e5e755d3f280996bcb8397ea4e8018b2ff885ac', 10196),
        ('bbob/2010/ABC_elabd_noiseless.tar.gz', 'd210bb248d17dd1f1ff2eecd9e82a34206426bc31771168b367751ef708ab894', 11231),
        ('bbob/2010/AVGNEWUOA_ros_noiseless.tar.gz', '59434c7fa761dd5d818faa3d4705f91eee92da49f0b776d9d09c1b4974fb7284', 11211),
        ('bbob/2010/CMAEGS_finck_noiseless.tar.gz', '04ca2e0b6271fd12c7ff6ebe2cb8be6f9a9e46831af9cfd27b9090c9378e818f', 12250),
        ('bbob/2010/DE-F-AUC_fialho_noiseless.tar.gz', '74f7fa0f8b6214c52efb053eabdecfd008148e2709af9e509b983a79ea5dc8da', 11504),
        ('bbob/2010/DEuniform_fialho_noiseless.tar.gz', '7874cedf230fb74d087c452595e354d28069234c869f96a113e8190eadfe17a7', 11700),
        ('bbob/2010/IPOP-ACTCMA-ES_ros_noiseless.tar.gz', 'd01322a022ce42825c323629db6d78f9a5231482597aff0ce9fe6f151e26fe22', 11298),
        ('bbob/2010/IPOP-CMA-ES_ros_noiseless.tar.gz', '9a7637368f439e0b583b90f7d0fee92982e60a0dbf26f63663497ea604ce40ef', 11425),
        ('bbob/2010/MOS_torre_noiseless.tar.gz', 'aa98703ef569a2382ac4a031b47c46abd71b4ec63d506b595ecede2b02350434', 11106),
        ('bbob/2010/NBC-CMA_preuss_noiseless.tar.gz', 'de0f47dc795e1836a042ecbf597b275acc5c1474ac7e88340c5cd030887e5d7e', 11087),
        ('bbob/2010/oPOEMS_kubalic_noiseless.tar.gz', '82ebf7500de42853f33e0ae1067d1414b3163692a0c5b9b41a28317d0e02b40a', 8975),
        ('bbob/2010/PM-AdapSS-DE_fialho_noiseless.tar.gz', 'd9bc5512366efcf6fc21b269a6501096c17b89304ebc97deeabae035b04b59e8', 10268),
        ('bbob/2010/pPOEMS_kubalic_noiseless.tar.gz', '9708310e5953895e426f82c44016a1de914eeb3c103f3eb470398bd88b3ec86c', 10129),
        ('bbob/2010/RCGA_tran_noiseless.tar.gz', '0acecea3b470b764442b15588cdf09ce263f55896ebc47808a65d5a15165be7b', 12792),
        ('bbob/2010/SPSA_finck_noiseless.tar.gz', '9ba301e0abf35763013434fa90f1e830d390d0035644f5926fe3ec1e224b79ff', 11599),
        ('bbob/2012/ACOR_liao_noiseless.tgz', 'a246c4083da2962de98896ccc4a3ff0e44228c61de1bbb82d09e181570d9bcd9', 11624),
        ('bbob/2012/BIPOPaCMA_loshchilov_noiseless.tgz', '44889a44f5882a9b8cd56c9bd274926955dd384ebc0cd20eb54985dbe55210f1', 10049),
        ('bbob/2012/BIPOPsaACM_loshchilov_noiseless.tgz', 'ae63c5bebee4662aeb15b2b99e599ba1fe62476e5d1d3afdb40b1b727e428596', 8297),
        ('bbob/2012/CMA_brockhoff_noiseless.tgz', 'ca51e56a9c3679be8c169b4c36ce94b629565f5dce42fd0344c5d15df9feaa56', 9871),
        ('bbob/2012/CMAa_brockhoff_noiseless.tgz', '9ed207d4ddefafb38b5af14cdeab4acf7d5163054991670306171c54d2147f71', 9761),
        ('bbob/2012/CMAES_posik_noiseless.tgz', '85b3aedf3dab8f5d6bfc5f88e08abfc5746769cfc1b1e57a5e9a8a41b41e4d80', 15995),
        ('bbob/2012/CMAm_brockhoff_noiseless.tgz', '6f11f61fd76f7fc47719ebbe3d80e7b5ff381a0677a3cedcdc13c154479a0bf2', 9821),
        ('bbob/2012/CMAma_brockhoff_noiseless.tgz', '6473cd7f10c4ca8c203024c0dbb343c63e05ebe8c02b449231a274a5dd12465b', 9667),
        ('bbob/2012/CMAmah_brockhoff_noiseless.tgz', '4bb4c8bbf15cb98559ba2cd91b4061653b4a9e97612644f60e08e7ade351c5d4', 9700),
        ('bbob/2012/CMAmh_brockhoff_noiseless.tgz', '62cbcdd428d6bd148a012c26d9af9003011821abfae8ed587c79655db7928c56', 9770),
        ('bbob/2012/DBRCGA_chuang_noiseless.tgz', '5c4a5705e51414cf32cb45444876a389ec5b102b8763d45c9aeba00bfbbc5722', 11168),
        ('bbob/2012/DE-AUTO_voglis_noiseless.tgz', '016017067b599d9c8d18aea96a452c9b6363674713d1544ffd65209fe4f3d927', 8632),
        ('bbob/2012/DE-BFGS_voglis_noiseless.tgz', '44f2a6597e69d97eb0a447651c773721039975a8138bd8894226c6ce0827b6d7', 9821),
        ('bbob/2012/DE-ROLL_voglis_noiseless.tgz', 'b330ba2ba3c9e270c9e006f24abaddcc1fe2d22b7eeee2b10068df40ad2230a4', 9681),
        ('bbob/2012/DE-SIMPLEX_voglis_noiseless.tgz', '332794c8b1ded6c4a50fc0581c869b22c2357d51858c51b17b685557ca9f8c6d', 10869),
        ('bbob/2012/DE_posik_noiseless.tgz', 'db805dfa580af9c3eb4e93810b7b38ab943a73627081d8d6dff6a8f72ecab24b', 11928),
        ('bbob/2012/DEAE_posik_noiseless.tgz', '733a253bab6008b2c4bd948b64dd6eacf6d11a5173e0bc7d1bb5177ecf0f6d86', 11709),
        ('bbob/2012/DEb_posik_noiseless.tgz', 'd56cb6cd2ff12a91cffde6bd3ffac6989a5ff476f0a2d2ffd863a0386c9833cc', 10312),
        ('bbob/2012/DEctpb_posik_noiseless.tgz', '3fdb9fe751e472984f514453c5e37451dfd1b665ad6295c64ba37f65e999743f', 10327),
        ('bbob/2012/IPOPsaACM_loshchilov_noiseless.tgz', 'f6e890a3cfc4c32c88bc0a5fa7e20d20e72c4885ff32c7b70b1bc9bb342721a3', 8252),
        ('bbob/2012/JADE_posik_noiseless.tgz', 'eb2d85177f306b2d0e82da90dab1cc3eeeeffc3d415833c2e2ee81ef8e4ce1f8', 11813),
        ('bbob/2012/JADEb_posik_noiseless.tgz', '161616d7cf4ff75d225795c69ca94979209180d05f0950156bff4116f82f99f6', 10471),
        ('bbob/2012/JADEctpb_posik_noiseless.tgz', 'd934eea28bbb3902586b92c19db5fca0cac5967b06f5c10776dd655a79ca539f', 10240),
        ('bbob/2012/MVDE_melo_noiseless.tgz', '70add742758a86f7d90b3512c66869ad832b8da2a570de374e75d4e448cba2ec', 11226),
        ('bbob/2012/NBIPOPaCMA_loshchilov_noiseless.tgz', '17f16844bf3a689e425bd12ccfadb52e0583dc0708fe4178c72bd81451ad9af8', 10108),
        ('bbob/2012/NIPOPaCMA_loshchilov_noiseless.tgz', '1fcf9638981026da0becdd1c0a1cfedd27517dff813c78be8e3515f91301b448', 10093),
        ('bbob/2012/PSO-BFGS_voglis_noiseless.tgz', 'ccde8376bfbdf64295f15e10a4b36e24670c08d2aa74afd12a090e3427a44972', 9595),
        ('bbob/2012/SNES_schaul_noiseless.tgz', '9d227d96bb3eebbc5a94ec8086b88323949a349f1c92e6cabe30f902f78b84ce', 15945),
        ('bbob/2012/xNES_schaul_noiseless.tgz', '38779ce6003ed8b76f7cb067a6b481d9ecbf63cf40a0aa227969d2d27fe36f64', 11824),
        ('bbob/2012/xNESas_schaul_noiseless.tgz', '344ae21594e32a635c99e96f34460a83129d5f129c216e060befcf4bd73fd3ef', 14311),
        ('bbob/2013/BIPOP-aCMA-STEP_loshchilov_noiseless.tgz', 'daf898ca724e6bddb18153a61d8725f4ae5576a7cf484978d086ea68ffb7d5e0', 8958),
        ('bbob/2013/BIPOP-saACM-k_loshchilov_noiseless.tgz', '46914c801af2c87e9bb2a3f6309e8227f28d7b8f39108c32ca83793bf1d81452', 9536),
        ('bbob/2013/CGA-grid100_holtschulte_noiseless.tgz', 'bc56600f2eed365c6b3391f5e2cf9074b3946c77d93cdd7db84eab689b45f057', 8214),
        ('bbob/2013/CGA-grid16_holtschulte_noiseless.tgz', '3afab0f29c9b4a463ccf15280cabb3a3361f7d78dfd2548c3cc49c124dbd6990', 7663),
        ('bbob/2013/CGA-ring100_holtschulte_noiseless.tgz', '652265c56b789e246ec1c0c67bbf39657ec4590ae66397b1f0b137ead842042e', 9728),
        ('bbob/2013/CGA-ring16_holtschulte_noiseless.tgz', 'e8bde854dee3326d8ff7c4ee61129b16cbdfffd5ac315f2ca674e81be6509f4d', 7948),
        ('bbob/2013/CMAES_Hutter_hutter_noiseless.tgz', 'c9d94cb7fa86bc9866a41987dd6a5d2fdbcb5d7cb4560b10db4bb6302dbbd64a', 5491),
        ('bbob/2013/DE_Pal_pal_noiseless.tgz', '1b130dd4dfd80349f1083aab35632fb51ac9a5611a690c8aa22f249045e5b8a0', 8797),
        ('bbob/2013/fmincon_pal_noiseless.tgz', 'de1ffff6b2ada942c9b8f8ad715310c580e4cb2fe494eb4cfdb2d648628479f7', 6558),
        ('bbob/2013/fminunc_pal_noiseless.tgz', '576b61c96c0b0215345d9f59cbe611bae9de8fa2dbd3d92beba23eeb5bd72d55', 6617),
        ('bbob/2013/GA-100_holtschulte_noiseless.tgz', '31b8096c59ca917dff794e9ed4f8a10426488c4d2255416ec30cc5a597eb2948', 9174),
        ('bbob/2013/HCMA_loshchilov_noiseless.tgz', 'e6cff600fe006936ea75a99e29612365ef3084ca05dfb9ef3c00e9de1f6afa3e', 7277),
        ('bbob/2013/HILL_holtschulte_noiseless.tgz', '632500f8689653128e72b780f45289b512372bcc67aad18320c6101f6b5ac148', 6520),
        ('bbob/2013/HMLSL_pal_noiseless.tgz', '305b76bd7421dd583ac4bb202ebe80a951f1cd829946fac8d3879dbd7ece6776', 6800),
        ('bbob/2013/IP-10DDr_liao_noiseless.tgz', '2a7e151c62316118cb29387de549562002f1e89d15ffba82103c98250af75bf6', 9526),
        ('bbob/2013/IP-500_liao_noiseless.tgz', '76f3fd67e2c03500f55508110d7140ca96c2231de9c458bd01beb22bca9fedfa', 9453),
        ('bbob/2013/IP_liao_noiseless.tgz', '9b02da43cbc6a8905fe0101f783cff1b97f3761b4f5b8f059042f1e2fe54bf9d', 9515),
        ('bbob/2013/IPOP400D_auger_noiseless.tgz', '3b65562182c23aa0590ffbf1031529593bf84879d8d6806076d3b6ae38171fce', 6362),
        ('bbob/2013/lmm-CMA-ES_auger_noiseless.tgz', 'fae1d61c1bbe7a9524250637045b06fabf186d60186f9433853a4c373aa29b06', 6580),
        ('bbob/2013/MEMPSODE_voglis_noiseless.tgz', '4a684b92c63a72d84e859df32d22f9a22d6abd15bddfa70d00c2b7aa4a77b2b9', 9843),
        ('bbob/2013/MLSL_pal_noiseless.tgz', 'b2a655b87e4153065f1c92ea052001567aef9a2f51d17f14591dfec904639418', 6645),
        ('bbob/2013/OQNLP_pal_noiseless.tgz', 'cef9dd580b84f715a36492d75ba92ad20570f81431fa2c6994ad832c45dd93e6', 6552),
        ('bbob/2013/P-DCN_tran_noiseless.tgz', '525cad57a0b3abe38d5b0a02b75f71dbcad22bb1def90c5311aba8abe67b4b0c', 9440),
        ('bbob/2013/P-zero_tran_noiseless.tgz', '95b9e8ae070fd6e47cec9d971627467e211e2f53f4882da72fc712c8b4df90b9', 8815),
        ('bbob/2013/PRCGA_sawyerr_noiseless.tgz', 'cd2d68bf9a170c70bc29d1c3c0e29d5be0d9216f99b24460089b7580fa3c9f4c', 10830),
        ('bbob/2013/simplex_pal_noiseless.tgz', '40193939490ae16e6e75feaffd59a30736cec1fc2d1f32928edccacf59bd9943', 7761),
        ('bbob/2013/SMAC-BBOB_hutter_noiseless.tgz', '9c56f677e231b6229556aacf64cccbb4f611e9a2a3c43de566f96d443f8371d4', 4202),
        ('bbob/2013/tany_liao_noiseless.tgz', '79cd553a99e38929d05b21326d818ce57476ec98d3938422d54e697a38866d9d', 9691),
        ('bbob/2013/texp_liao_noiseless.tgz', '9ec405ca9dc45b795a60082d09e166cae711b6d452b4a9ab56942e61027ea5da', 9925),
        ('bbob/2013/U-DCN_tran_noiseless.tgz', 'ea8d3fc4ee7b28af2680b195cc2b7341a5563900af47a9af58816c73317d331f', 9585),
        ('bbob/2013/U-zero_tran_noiseless.tgz', '6286ca5bbc9624ac82c81d8550f93d75c2dce117ce2c06c9c01ae68b19b07f3f', 8279),
        ('bbob/2015-CEC/MATSUMOTO-Brockhoff-noiseless.tgz', 'eb2543fcfc6d41310edbbd523b877db2c759db481db0135e20b5c9f5c610108d', 4807),
        ('bbob/2015-CEC/R-DE-10e2-Tanabe-noiseless.tgz', '4bb88c605435f89aea4f3b26ad0b347a3b603bc1aa6a865584a31673ca82c120', 4509),
        ('bbob/2015-CEC/R-DE-10e5-Tanabe-noiseless.tgz', 'd843d9edf54cf637f134a990b6a6937ef72e487545ac1a6351514cb76b27052b', 10266),
        ('bbob/2015-CEC/R-SHADE-10e2-Tanabe-noiseless.tgz', '6d89e89876566b57758ee14dad07dbc6152e36e3971f62ae4789be44c29310ac', 4131),
        ('bbob/2015-CEC/R-SHADE-10e5-Tanabe-noiseless.tgz', '8d893f414f2f01197c3386bbed2fc8f7ce01b715560a930d56c614e39d06d589', 11598),
        ('bbob/2015-CEC/RL-SHADE-10e2-Tanabe-noiseless.tgz', '36a64007f60115cacbf54427182c9438177af3a62cd74510bc38ed3638be3fd2', 4518),
        ('bbob/2015-CEC/RL-SHADE-10e5-Tanabe-noiseless.tgz', 'cef2938e2d40803aef18f6294faaa88ac06d61bad8d6c204ef070ad2eb0bbcb1', 10729),
        ('bbob/2015-CEC/SOO-Derbel-noiseless.tgz', '0a9d505d8998f4886ca27847e26ec18d7f1e881e635ef5ec984b2180fc535cb0', 7013),
        ('bbob/2015-GECCO/BSif.tgz', '8cc18e2515990749743d2240dc73c99ffc66c4a8b347316384e029f6089971b7', 4520),
        ('bbob/2015-GECCO/BSifeg.tgz', '39bdafd9c150c700d0a347330934972c3d899711f7c1131477efb797f8a29909', 4857),
        ('bbob/2015-GECCO/BSqi.tgz', 'ccc3dd5c586677343271955cf3ba07754e2849a9e46cf478e5d65af7c86b113f', 4142),
        ('bbob/2015-GECCO/BSrr.tgz', '0c939e4d44c9b54abebf5600ee0b2e6213d998b9c47236b721ec20082660d040', 4970),
        ('bbob/2015-GECCO/CMA-CSA.tgz', '0118782525c3f26e9506837bd06a58953902f239bc3512a09f81fb88b2f20406', 9757),
        ('bbob/2015-GECCO/CMA-MSR.tgz', '940757df61d177e23b1bf9e1dce44301e4a16908508a2e4d960f17307229e786', 9703),
        ('bbob/2015-GECCO/CMA-TPA.tgz', '4da2a7922e76e9090f4e168a173cfbecf5a1511fb8f99d588a9ad83eb3329a9a', 9663),
        ('bbob/2015-GECCO/GP1-CMAES-2013instances.tgz', '5fed013afeba11a94bbcde0b49e0d3d1a1b39049fb01e36699b9f6c894d99b07', 5536),
        ('bbob/2015-GECCO/GP1-CMAES.tgz', 'e3dae53737db23112bbb8e7d6f241e7133406ea4b2927eb6009900a974d9ab9b', 5518),
        ('bbob/2015-GECCO/GP5-CMAES-2013instances.tgz', '6c071c0366f9ffe54f39b900f0961551129f1387fea46f893f4410bfdaf06096', 5459),
        ('bbob/2015-GECCO/GP5-CMAES.tgz', 'c7454076a6d8db04303d5c077fd3625041493d8afdf3945e19c7d46dd0bc1002', 5451),
        ('bbob/2015-GECCO/IPOPCMAv3p61-2013instances.tgz', '0e11c6fed04b3acb3f082b2f77136ac9b6c63075ecb0649b40368cc7d87b9452', 5604),
        ('bbob/2015-GECCO/IPOPCMAv3p61.tgz', '5da4f566cab5b86a6488b467d4d02ea6d4a867befa17bbd80bf479cc7b075fd9', 5601),
        ('bbob/2015-GECCO/LHD-10xDefault-MATSuMoTo.tgz', '1662210a01efde4ba89aa8ea7b52764ec854b78269808086d9f6d7258b341819', 4005),
        ('bbob/2015-GECCO/LHD-2xDefault-MATSuMoTo.tgz', '7c2100acaa27dc838f13fe8365f4f4cc347ddb8d843e34bf8be480d7220f0ed2', 3937),
        ('bbob/2015-GECCO/RAND-2xDefault-MATSuMoTo.tgz', '6179fd3bddae4b26c51e4f7ac9218cdbd110ffb45a5ecdf4e76e67337d06b108', 4044),
        ('bbob/2015-GECCO/RF1-CMAES-2013instances.tgz', 'f496cc6f9143bbd14b9284907c7251a18e8499073faccd5fc70729af44318a20', 5331),
        ('bbob/2015-GECCO/RF1-CMAES.tgz', 'e0b83f17a88c6e96c15999772660b53e4583772f998184392ba5365b9d3d8411', 5331),
        ('bbob/2015-GECCO/RF5-CMAES-2013instances.tgz', '69a8d9c23eed29ed042c90b35b25b5d6ce01ee062b912d760a19d3233d372b24', 4948),
        ('bbob/2015-GECCO/RF5-CMAES.tgz', 'bd7685a695d777266f5f09b32ae2c19973ef28f2fed3278828a80bf0f090abc6', 4942),
        ('bbob/2015-GECCO/Sif.tgz', '4f6fd80d85017945b957d265f36125b5bceb30352b3e0ef3582a23f146424c4e', 4934),
        ('bbob/2015-GECCO/Sifeg.tgz', '8610210ec2216ecec46a6aaaf45c992a894dc2860342d257b18d81a24690ac66', 5117),
        ('bbob/2015-GECCO/Srr.tgz', 'f343da9bb78596dbfd3baa56ef2f8ff55f0950c2b7e4df57362f07c4141be399', 5235),
        ('bbob/2016/PSAaLmC.tgz', '024b76720cca09bcf8960441bb474de436e878213a186a00e87711ff07f4c058', 10048),
        ('bbob/2016/PSAaLmD.tgz', 'd880d05fb2f6001205607acc7e5595ed65426881cf9fcecfe9662fc931ea36e7', 9866),
        ('bbob/2016/PSAaSmC.tgz', '900abf5ef358598cd63d4b71e107c58719aa0f965ddfee85d172c19c8a732614', 10035),
        ('bbob/2016/PSAaSmD.tgz', 'd6e5bf9ad50e5b2f9b9e3160f3ee9a1aeca6679d2f4ce98df7625bee80a2bb81', 9918),
        ('bbob/2017/CMAES-APOP.tgz', 'ebb833baec134f9b8976ee461b71ee9bdc5a0dd4f22f443358b5e45bf76f4572', 17931),
        ('bbob/2017/DTS-CMA-ES.tgz', '50d553800f78bb01f6be17604b1f21677e36a1adb1c1016326b9b948e6d34fa2', 13244),
        ('bbob/2017/EvoSpace-PSO-GA.tgz', 'c52056db1b4d7475c23796eaf2fbad6686e56375856343b3f8750eac9b723910', 2076),
        ('bbob/2017/KL-BIPOP.tgz', '1a2d42c4eca6e737752723d7fa19ec2a3706b0e7dabd6bb77bc32ad2240a1538', 14346),
        ('bbob/2017/KL-IPOP.tgz', '80f6b96c6bc8190a1665b75e38cfffe5b7465ed95d22e33375cf5c5ad25c015b', 10805),
        ('bbob/2017/KL-Restart.tgz', '8ac46074bb38cb31d4a5f540b295bf41eed020325ff7f06678c66643222e8b3b', 18401),
        ('bbob/2017/Ord-Q-DTS.tgz', '78dad617a76b9cddbdb3343f8b7b407a8d4e9f334ed5949c12951b4a6af55bb3', 16629),
        ('bbob/2017/SSEABC.tgz', '44732f7f9ac2ee23175b8ded90d106f3ded53866ec50dc555cbc08526f938cd2', 16435),
        ('bbob/2017-outsideGECCO/RS4-1e7D.tgz', 'dd216a1882900a1ff2f1460aa141aa85a9f9145f724bf3db23fa6513e6f7ecd5', 15402),
        ('bbob/2017-outsideGECCO/RS4p5-1e7D.tgz', '25ec9d4edc19ac6505969ef4d89e979f42cb5ed7e2372766f18f06291aa51040', 15541),
        ('bbob/2017-outsideGECCO/RS5-1e7D.tgz', '27d82fbaf14442ee46160e64c8ba4aa2fe5b9c6fedd08d37077b359c1e196c14', 15456),
        ('bbob-biobj/2016/DEMO.tgz', '3ea85ff39e92d3f4327c8f72fc8bd0e20dbedf59685ed1bf666a9f383bc82154', 9725),
        ('bbob-biobj/2016/GA-MULTIOBJ-NSGA-II.tgz', '4e7a550277583a331276b08b309c3168edebf3ec7e5b06d72365670c3a24e9cc', 9371),
        ('bbob-biobj/2016/HMO-CMA-ES.tgz', '6c2cc5db010974584577fdd4eff0538c68279a2d348c638423ed7c78423a4f14', 18850),
        ('bbob-biobj/2016/MAT-DIRECT.tgz', 'afc238e60e03d38e9d63feaced3883f53e405d09d6c9f26161b0edaab4af4c21', 742),
        ('bbob-biobj/2016/MAT-SMS.tgz', 'c96050e25a11b81bcb91fa8ae6fb0e032a6bfedfa39c30d79e0935142b5b86e0', 753),
        ('bbob-biobj/2016/MO-DIRECT-HV-Rank.tgz', '74a4671b31700c3ab6be7a32455051b713a79b11846773c62fcdc1f195a5dfa9', 3711),
        ('bbob-biobj/2016/MO-DIRECT-ND.tgz', '038f83e4f0d545daf6e5942816c6e5398c1698f3a9194b8d4f2618850671b964', 2425),
        ('bbob-biobj/2016/MO-DIRECT-Rank.tgz', '4f13c5005b7c680c7908477510abe316a3e8453d19cfc92f9d1cdedc1d941a35', 2376),
        ('bbob-biobj/2016/RM-MEDA.tgz', '97d89fc4e7e4fd350dcbc673caf412389bf581b44aed133057ac0a2f6a00e64e', 13484),
        ('bbob-biobj/2016/RS-100.tgz', '97ff31847c022156d73ba4730df057dc475e33aaae3a5616232f49140b175f99', 2352),
        ('bbob-biobj/2016/RS-4.tgz', '0d4066dcb0ef478166aaf3e717ed059bcacb032b542b5d3287bd07ad47a94da8', 5649),
        ('bbob-biobj/2016/RS-5.tgz', 'b417841eaeca4eecf2ae3e2205d671a46ed5dda2b6c7b47e71a5002ef7d23bb6', 7174),
        ('bbob-biobj/2016/SMS-EMOA-DE.tgz', 'c1348cf48d21680ca4e66012cc887e8d62e9ec40c8ece1cd8e81a42d579203b4', 10792),
        ('bbob-biobj/2016/SMS-EMOA-PM.tgz', 'd28c7e0e15ece9c3995884d1f52764cba10b3c53fc5f3263274a7ae5ba20a516', 9831),
        ('bbob-biobj/2016/UP-MO-CMA-ES.tgz', 'bb393da7f1f7213dc3335426a330eb63975cf2b5a324430cdad5b3bca286be3a', 17947),
        ('bbob-biobj/2017/SMS-EMOA-SA.tgz', '0886e00b241f621a175b197c3c845e790121bce0cb103363acf0bbb256e7f49c', 12279),
        ('bbob-noisy/2009/ALPS_hornby_noisy.tgz', 'e18f829dd313013f502305958df89c4a21cf0526e54c329272b226c9cf73a23b', 23257),
        ('bbob-noisy/2009/AMALGAM_bosman_noisy.tgz', '4c244655fd34451603cd0c78357aaff271a14f56004bff97ba65aa8ec9b4b655', 23319),
        ('bbob-noisy/2009/BAYEDA_gallagher_noisy.tgz', '2ec479f6cfeafaee05bfb7a8bbae28724124206c3f17623a4f9980bc4b3028d4', 16267),
        ('bbob-noisy/2009/BFGS_ros_noisy.tgz', '8ec01dcaa6bbce310865b0cce1b35d20eb8438fa151b89bbd1cab8444e557b59', 9253),
        ('bbob-noisy/2009/BIPOP-CMA-ES_hansen_noisy.tgz', 'cd0dfb1604149834a712d11f9336f9f60fe33af221c2fb72fc6794695d960988', 23987),
        ('bbob-noisy/2009/CMA-ESPLUSSEL_auger_noisy.tgz', 'dcd1a52abab0135aab20787fca0434d232a675b9e2e1d188db9dddb81488fb89', 16743),
        ('bbob-noisy/2009/DASA_korosec_noisy.tgz', '4b625fc887a9a93fae81773e8673d7e726da1b731c08d005b372972e3ab86ebb', 15775),
        ('bbob-noisy/2009/DE-PSO_garcia-nieto_noisy.tgz', 'fb25587540fe1303bfb3404b8494816604400da01b75ef49128f84320774ba46', 14006),
        ('bbob-noisy/2009/EDA-PSO_el-abd_noisy.tgz', '54b5b3d5cc6f4c35c09f711427dfba46c6b3bcf7294caf416e532b4ea90ee462', 12448),
        ('bbob-noisy/2009/FULLNEWUOA_ros_noisy.tgz', 'c45c8cc6a167aca75fb3bd28c0a185909cc729b5a87c6c438603350b6e64da30', 7383),
        ('bbob-noisy/2009/GLOBAL_pal_noisy.tgz', 'f333dff06e3f42f5044c965cc60beead61bb6c76b14b39230d0048e8cb67c527', 7560),
        ('bbob-noisy/2009/iAMALGAM_bosman_noisy.tgz', '8d00e9ef4da296a6dc5359bcd8aa4bec2845702a88342e5fea7ad6039872b7e3', 23872),
        ('bbob-noisy/2009/IPOP-SEP-CMA-ES_ros_noisy.tgz', '9d5f2425ce5dd501449bc73aace3183c0ead6d6f6ad33897ae7337303724cbb4', 18382),
        ('bbob-noisy/2009/MA-LS-CHAIN_molina_noisy.tgz', 'a452a81a2a46a60b76f64045761171f6945a2806a4a4543686db7eeb15d52290', 12471),
        ('bbob-noisy/2009/MCS_huyer_noisy.tgz', '7c010c165c0e27331106a3181f190c1da30c16ece69ec8a74e4644974250eabb', 5472),
        ('bbob-noisy/2009/ONEFIFTH_auger_noisy.tgz', 'b4d3a575313c30cbb4f75029e373f63ec3af0a6b2eb98ac5fc81a0b52b85d0ed', 24567),
        ('bbob-noisy/2009/PSO_Bounds_el-abd_noisy.tgz', 'e684648fa0faba9303d1f073233c8651452160420a3fdd5253c3960d391f6a44', 11958),
        ('bbob-noisy/2009/PSO_el-abd_noisy.tgz', '7cc3a0006254144502b6e2e0c416b32d58001b9774df5612d24f31ec9a0514d2', 12144),
        ('bbob-noisy/2009/RANDOMSEARCH_auger_noisy.tgz', 'd7f49cd7d44693977d09f917b4780cc8b8848587b442d18b3a41f329add359f2', 24468),
        ('bbob-noisy/2009/SNOBFIT_huyer_noisy.tgz', 'd960fa0570d117bc29409de6184a9745184d1048fbdf06f4be1cf4148423718d', 6568),
        ('bbob-noisy/2009/VNS_garcia-martinez_noisy.tgz', 'be3c913d160e141a22fca488ae3d119c17cb291cbae5e6a56b2fa0fe0ac0f96a', 12123),
        ('bbob-noisy/2010/1komma2_brockhoff_noisy.tar.gz', '26e8d5ea70be401d4757db6798c5d875ba3e0b73f0f841f340d5bcf91902f084', 12133),
        ('bbob-noisy/2010/1komma2mir_brockhoff_noisy.tar.gz', '4c7f0df3976e4ed2e8712ecdcbec6f048592a295fdb0a5fa0b4ce419d28ddaab', 12078),
        ('bbob-noisy/2010/1komma2mirser_brockhoff_noisy.tar.gz', '629a23e7437eafc050c8ec8a0f24faf974702dd5e6cb3297b16f887533052232', 12004),
        ('bbob-noisy/2010/1komma2ser_brockhoff_noisy.tar.gz', 'dce084161f283a779f0ffaddf89c0b143fea909980f70f1d9baa9f3e4f87d34a', 12081),
        ('bbob-noisy/2010/1komma4_brockhoff_noisy.tar.gz', 'be4b0ee3bbc0a7f3abe7218549f33535075677180c60a3dd3c3d7de57c69eb15', 11967),
        ('bbob-noisy/2010/1komma4mir_brockhoff_noisy.tar.gz', '9a810c5f9718fdc304184e04d90aa3e50c7e01ed95eff1b0686961ddd33cfe70', 11917),
        ('bbob-noisy/2010/1komma4mirser_brockhoff_noisy.tar.gz', 'ec0b9557f4879e4b61f670fc1a1e137427a602f7d512b50b3f0f28d3efb03db4', 12901),
        ('bbob-noisy/2010/1komma4ser_brockhoff_noisy.tar.gz', '471054217fb6733e07d1b146011e18322c7a974e850ef86d1799920262372861', 11872),
        ('bbob-noisy/2010/AVGNEWUOA_ros_noisy.tar.gz', '6cd51e8e1bcfe9f00fc65c6f2b775581f2164a688e0879325406bbb2872c85ed', 8952),
        ('bbob-noisy/2010/CMAEGS_finck_noisy.tar.gz', 'e6a8e8c10bd19f6ac6907cfc6bed91c483a5876ae5faf6cc81a6fc4ad5ac8f10', 16659),
        ('bbob-noisy/2010/IPOP-ACTCMA-ES_ros_noisy.tar.gz', '36d63234d99833b20e8f66cd27ae6e65d32774ba07ad8219fdfee859592ae402', 16219),
        ('bbob-noisy/2010/IPOP-CMA-ES_ros_noisy.tar.gz', '8dd6da09848e08e4269e584912ec07150c2d8fe76d4b6e80a114217ee83e4b79', 16496),
        ('bbob-noisy/2010/MOS_torre_noisy.tar.gz', '6227fa4186339ffc1200055cb08f6c8a77b74c5b8c03f059ec77780380d5791a', 14477),
        ('bbob-noisy/2010/NEWUOA_ros_noisy.tar.gz', '8366758198b0b692041734ce30a4db1de56e2b6d073d03c42b4803eff231ff18', 8929),
        ('bbob-noisy/2010/RCGA_tran_noisy.tar.gz', '8895fdac6cdcaa71c5e29efd996c572bd644006fcc637f1713048674463e79a4', 15980),
        ('bbob-noisy/2010/SPSA_finck_noisy.tar.gz', 'bbd1d82adc7c10d69457bf406eb6a8e10a8facf15c31bf69a7b79fc9fa315b4b', 15679),
        ('bbob-noisy/2012/IPOPsaACM_loshchilov_noisy.tgz', '8cd69fe16578c2fbcb5f9b362a7405ae9bd07a4e11f0544f81c099f4374a1493', 14374),
        ('bbob-noisy/2012/SNES_schaul_noisy.tgz', '9c45526124e3c8b000fb378f5b0e91f094aab40cff555dad7a57fb7102c1ad9c', 17453),
        ('bbob-noisy/2012/xNES_schaul_noisy.tgz', '7c0c1f452ea335fc6118e167687b8fb65e957d9a310a2fecc5bbd5775d90eccc', 20194),
        ('bbob-noisy/2012/xNESas_schaul_noisy.tgz', '907995075f026e9f702a74f64517809e1f43410c99a6e6bc39522f30e07a918b', 25492),
        ('bbob-noisy/2016/PSAaLmC-noisy.tgz', '0d7f3e30850c3884a18463a77a4d069263fe47e6d213f81f9a08c40ea16d5505', 13466),
        ('bbob-noisy/2016/PSAaLmD-noisy.tgz', 'dc2dc8401baa7818c2f895f3fadf0c7a2c6d4b9f1cf17ea034391c2f9e152a0e', 13319),
        ('bbob-noisy/2016/PSAaSmC-noisy.tgz', '41c69c3e6c9c7662a1321a63219ddd0c661d94750b569dbecab8d5c4523e532f', 13366),
        ('bbob-noisy/2016/PSAaSmD-noisy.tgz', '8eec30e968d60acd4cd8b21aecf9e0dd05c19ac93285e5e6a80137507357b75a', 13174),
        ('test/N-II.tgz', '4e7a550277583a331276b08b309c3168edebf3ec7e5b06d72365670c3a24e9cc', 9371),
        ('test/RS-4.zip', '1e9b0f4e63eaf934bdd819113c160b4663561f2d83059d799feb0c8cb5672978', 6158),
]

    @property
    def names_found(self):
        """names as (to be) used in `get` when called without argument.

        `names_found` is set in the `get` or `find` methods when called
        with a search or index argument.

        """
        return self._names_found

    def __init__(self,
                 local_path='~/.cocopp/data-archive',
                 url='http://coco.gforge.inria.fr/data-archive',
                 definition_file=None):
        """Arguments are a full local path and an URL.

        ``~`` refers to the user home folder. By default the archive is
        hosted at ``'~/.cocopp/data-archive'``.

        For the time being, `definition_file` may be a filename in
        `local_path` containing the `_all` list describing the (remote)
        archive, however this functionality has never been tested and
        needs most likely a few bug fixes.
        """
        self.local_data_path = os.path.expanduser(os.path.join(*local_path.split('/')))
        self.remote_data_path = url
        self._names_found = []  # names recently found
        self._print = print  # like this we can make it quiet for testing
        if definition_file:
            self._all = self._read_names_list(definition_file)
        # extract names (first column) from _all
        if not hasattr(self, '_all'):
            self._all = COCODataArchive._all
        list.__init__(self, (entry[0] for entry in self._all))


    def __call__(self, *substrs):
        """alias to `find`"""
        return self.find(*substrs)

    def find(self, *substrs):
        """return names of archived data that match all `substrs`.

        This method serves for interactive exploration of available data
        and is aliased to the shortcut of calling the instance itself.

        When given several `substrs` arguments the results match each
        substring (AND search, an OR can be simply achieved by appending
        the result of two finds). Upper/lower case is ignored.

        When given a single `substrs` argument, it may be

        - a list of matching substrings, used as several substrings as above
        - an index of `type` `int`
        - a list of indices

        Returned names correspond to the unique trailing subpath of
        data filenames. The next call to `get` without argument will
        retrieve the first found data and return the full data path. A
        call to `get_all` will call `get` on all found entries and
        return a `list` of full data paths which can be used with
        `cocopp.main`.

        Example:

        >>> import cocopp
        >>> cocopp.bbob.find('Auger', '2013')[1]
        '2013/lmm-CMA-ES_auger_noiseless.tgz'

        Sitting in front of a shell, we prefer using the shortcut to find
        via `__call__`:

        >>> cocopp.bbob('Auger', '2013') == cocopp.bbob.find('Auger', '2013')
        True

        Details: The list of matching names is stored in `names_found`.
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
            try:
                names = [name for name in names if s.lower() in name.lower()]
            except AttributeError:
                warnings.warn("arguments to `find` must be strings or a single integer or an interger list")
                raise
        self._names_found = names
        return StringList(names)

    def find_indices(self, *substrs):
        """same as `find` but returns indices instead of names"""
        return [self.index(name) for name in self.find(*substrs)]

    def print(self, *substrs):
        """print the result of ``find(*substrs)`` with indices.

        Details: does not change `names_found` and returns `None`.
        """
        current_names = list(self._names_found)
        for index in self.find_indices(*substrs):
            print("%4d '%s'" % (index, self[index]))
        self._names_found = current_names

    def get_all(self, indices=None, remote=True):
        """Return a `list` (`StringList`) of absolute pathnames,

        by repeatedly calling `get`. Elements of the `indices` list can
        be an index or a substring that matches one and only one name
        in the archive. If ``indices is None``, the results from the
        last call to `find` are used. Download the data if necessary.

        See also `get`.
        """
        if indices is None:
            names = self.names_found
        else:
            names = self.find(indices)
        return StringList(self.get(name, remote=remote)
                          for name in names)

    def get_first(self, substrs, remote=True):
        """get the first archived data matching all of `substrs`.

        `substrs` is a list of substrings.

        `get_first(substrs, remote)` is a shortcut for::

            self.find(*substrs)
            if self.names_found:
                return self.get(self.names_found[0], remote=remote)
            return None

        """
        self.find(*substrs)
        if self.names_found:
            return self.get(self.names_found[0], remote=remote)
        return None


    def get(self, substr=None, remote=True):
        """return the full data pathname of `substr` in the archived data.

        Retrieves the data from remote if necessary.

        `substr` can be a substring that matches one and only one name in
        the data archive or an integer between 0 and `len(self)`.

        Raises a `ValueError` if `substr` matches several archive entries
        on none.

        If ``substr is None`` (default), the first match of the last
        call to ``find*`` or ``get*`` is used like `self.names_found[0]``.

        If ``remote is True`` (default), the respective data are
        downloaded from the remote location if necessary. Otherwise
        return `None` for a match.
        """
        # handle different ways to pass arguments, create names
        if not isinstance(remote, bool):
            raise ValueError(
                "Second argument to `COCODataArchive.get` must be a "
                "`bool`,\n that is ``remote=True`` or `False`."
                "Use a `list` of `str` to define several substrings.")
        if substr is None:
            try:
                names = [self.names_found[0]]
            except IndexError:
                raise ValueError("nothing specified to `get`, use `find` first or give a name")
        elif isinstance(substr, int):
            names = [self[substr]]
        else:
            names = self.find(substr)

        # check that names has only one match
        if len(names) < 1:
                raise ValueError("'%s' has no match in data archive" % substr)
        elif len(names) > 1:
                raise ValueError("'%s' has multiple matches in data archive:\n   %s"
                                 % (substr,
                                    '\n   '.join(names)))
        # create full path
        full_name = self.full_path(names[0])
        if os.path.exists(full_name):
            self.check_hash(full_name)
            return full_name
        if not remote:
            if 22 < 3:  # suppressed, because remote is by default True
                warnings.warn('Name %s locally not found by COCODataArchive. Consider option "remote=True".'
                      ' %s' % full_name)
            return None

        # download
        if not os.path.exists(os.path.split(full_name)[0]):
            os.makedirs(os.path.split(full_name)[0])  # create path
        url = '/'.join((self.remote_data_path, names[0]))
        self._print("  downloading %s to %s" % (url, full_name))
        urlretrieve(url, full_name)
        self.check_hash(full_name)
        return full_name

    def get_one(self, *args, **kwargs):
        """depreciated, for backwards compatibility"""
        return self.get(*args, **kwargs)

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

        Check that all names are only once found in the data archive:

        >>> from cocopp import bbob
        >>> for name in bbob:
        ...     assert bbob.count(name) == 1, "%s counted %d times in data archive" % (name, bbob.count(name))
        ...     assert len(bbob.find(name)) == 1, "%s found %d times" % (name, bbob.find(name))

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
            return self._all[self.index(self.name(name))][1]
        except IndexError:
            return None

    def _read_names_list(self, filename='archive_info.txt'):
        """read data archive info for `_all`

        from a file generated with `_generate_names_list`

        TODO: never tested
        """
        with open(os.path.join(self.local_data_path, filename), 'rt') as file_:
            return ast.literat_eval(file_.read())

    def _generate_names_list(self, filename=None):
        """write an _all list of an existing archive including hashes and
        filesizes to `filename`.

        This may serve as new/improved/updated/final _all class
        attribute via manual copy-paste.

        May or may not need to be modified under Windows.
        """
        res = []
        for dirpath, dirnames, filenames in os.walk(filename):
            for filename in filenames:
                if '.extracted' not in dirpath \
                        and not filename.endswith(('dat', 'info')) \
                        and not ('BBOB' in filename and 'rawdata' in filename):
                    name = '/'.join([dirpath.replace(os.path.sep, '/'), filename])[len(filename) + 1:]
                    path = os.path.join(dirpath, filename)
                    res = res + [(
                        name,
                        self._hash(path),
                         os.path.getsize(path) // 1000)] # or os.stat(path).st_size
        if filename:
            with open(os.path.join(self.local_data_path, filename), 'rt') as file_:
                file_.write(repr(res))
        return res

class COCOBBOBDataArchive(COCODataArchive):
    """This class "contains" archived data for the 'bbob' suite.

    >>> import cocopp
    >>> cocopp.bbob  # doctest:+ELLIPSIS
    ['2009/ALPS_hornby_noiseless.tgz',...
    >>> isinstance(cocopp.bbob, cocopp.findfiles.COCOBBOBDataArchive)
    True

    While the data are specific to `COCOBBOBDataArchive`, all the
    functionality is inherited from the parent `class` `COCODataArchive`:
    """
    __doc__ += COCODataArchive.__doc__
    def __init__(self,
                 local_path='~/.cocopp/data-archive/bbob',
                 url='http://coco.gforge.inria.fr/data-archive/bbob'):
        """Arguments are a full local path and an URL.

        ``~`` refers to the user home folder.
        """
        self._all = [[line[0][5:]] + list(line[1:]) for line in COCODataArchive._all
                     if line[0].startswith('bbob/')]
        COCODataArchive.__init__(self, local_path, url)

class COCOBBOBNoisyDataArchive(COCODataArchive):
    """This class "contains" archived data for the 'bbob-noisy' suite.

    >>> import cocopp
    >>> cocopp.bbob_noisy  # doctest:+ELLIPSIS
    ['2009/ALPS_hornby_noisy.tgz',...
    >>> isinstance(cocopp.bbob_noisy, cocopp.findfiles.COCOBBOBNoisyDataArchive)
    True

    While the data are specific to `COCOBBOBNoisyDataArchive`, all the
    functionality is inherited from the parent `class` `COCODataArchive`:
    """
    __doc__ += COCODataArchive.__doc__
    def __init__(self,
                 local_path='~/.cocopp/data-archive/bbob-noisy',
                 url='http://coco.gforge.inria.fr/data-archive/bbob-noisy'):
        """Arguments are a full local path and an URL.

        ``~`` refers to the user home folder. By default the archive is
        hosted at ``'~/.cocopp/data-archive/bbob'``.
        """
        self._all = [[line[0][11:]] + list(line[1:]) for line in COCODataArchive._all
                     if line[0].startswith('bbob-noisy/')]
        COCODataArchive.__init__(self, local_path, url)

class COCOBBOBBiobjDataArchive(COCODataArchive):
    """This class "contains" archived data for the 'bbob-biobj' suite.

    >>> import cocopp
    >>> cocopp.bbob_biobj  # doctest:+ELLIPSIS
    ['2016/DEMO.tgz', '2016/GA-MULTIOBJ-NSGA-II.tgz',...
    >>> isinstance(cocopp.bbob_biobj, cocopp.findfiles.COCOBBOBBiobjDataArchive)
    True

    While the data are specific to `COCOBBOBBiobjDataArchive`, all the
    functionality is inherited from the parent `class` `COCODataArchive`:
    """
    __doc__ += COCODataArchive.__doc__
    def __init__(self,
                 local_path='~/.cocopp/data-archive/bbob-biobj',
                 url='http://coco.gforge.inria.fr/data-archive/bbob-biobj'):
        """Arguments are a full local path and an URL.

        ``~`` refers to the user home folder. By default the archive is
        hosted at ``'~/.cocopp/data-archive/bbob-biobj'``.
        """
        self._all = [[line[0][11:]] + list(line[1:]) for line in COCODataArchive._all
                     if line[0].startswith('bbob-biobj/')]
        COCODataArchive.__init__(self, local_path, url)
