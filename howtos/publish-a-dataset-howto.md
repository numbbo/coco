# How to Publish Your Data Within COCO

There are two ways to make data easily accessible to the community:

- propose inclusion of your data into [`cocopp.archives`](https://numbbo.github.io/data-archive), or
- host your own COCO archive with your data.

In both cases, first, the data need to be prepared. For this, for each
dataset (that is, each benchmarked algorithm variant):

  1. <details><summary><b>Zip the data folder.
     </b> (click to view)</summary>
     A data zipfile contains a single folder under which all data from a
     single full experiment was collected. The folder can contain subfolders
     (or subsub...folders), for example of data from different (sub)batches of
     the complete experiment. Valid formats are
     <tt>.gzip</tt> or <tt>.tgz</tt> or <tt>.zip</tt>
     </details>

  1. <details><summary><b>Rename the zip file.
     </b> (click to view)</summary>
     The name of the zipfile defines the name of the data set.
     The name should represent the benchmarked algorithm and may contain
     authors names (but rather not the name of the test suite).
     The name can have any length, but the first ten-or-so characters should
     be a meaningful algorithm "abbreviation".

## Propose Inclusion to the COCO Data Archive

This option is available if one or several datasets were used in a publication
or in a preprint available for example on [arXiv](https://arxiv.org) or
[HAL](https://hal.archives-ouvertes.fr).
For this:

  3. Upload the above data zipfile(s) to a file sharing site or to an accessible URL.
  4. Ask for the inclusion into [`cocopp.archives`](https://numbbo.github.io/data-archive).
     For this, open an [issue at the Gitlab repository of COCO](https://github.com/numbbo/coco/issues)
     (you need to have a Github account) with

     - the publication reference and a link to the paper
     - a very short description of each dataset including the name of
       - the algorithm
       - the test suite
       - the zip file
     - a link to the dataset zip file(s)
     - (optional) a link to the source code to reproduce the dataset

## Host an Archive

Hosting an archive means putting one or several data zipfiles with an added
"archive definition text file" online in a dedicated folder that can be
accessed under an URL, like http://cma-es.github.io/lq-cma/data-archives/lq-gecco2019.
For example, any folder under a personal homepage root will do.

For this:

  3. <details><summary><b>Move the above data zipfile(s) into a clean folder</b>,
     possibly with subfolders (click to see more).</summary>
     The folder name is only used as part of the URL and can be changed after
     creating the archive. If desired, subfolders can be created that become part
     of the names of the datasets under this subfolder. These can not be changed
     without repeating the following creation procedure:</details>

  1. <details><summary><b>Create the archive</b>
     (two lines of Python code, click to see more).</summary>
     Assume the data zipfiles are in the folder <tt>elisa_2020</tt> or its
     subfolders and <tt>cocopp</tt> is installed (<tt>pip install cocopp</tt>).
     In a Python shell, it suffices to type:

     ```python
     import cocopp
     cocopp.archiving.create('elisa_2020')
     ```

     thereby "creating" the archive locally by adding an archive
     definition file to the folder <tt>elisa_2020</tt>.
     Archives can contain other archives as subfolders or,
     the other way around, additional subarchives can be
     created in any archive subfolder. This is how
     https://numbbo.github.io/data-archive/ is organized.
     <details><summary>Alternative code (from a system shell, click to expand)</summary>
     <tt>python -c "import cocopp; cocopp.archiving.create('elisa_2020')"</tt>
     </details>
     </details>

  1. **Upload the archive folder** and its content to where it can be accessed
     via an URL. The archive is now accessible with `cocopp.archiving.get('URL')`
     (see below example).

  1. **Open an** [**issue** at the Github repository of COCO](https://github.com/numbbo/coco/issues)
     (you need to have a Github account) signalling the URL of the archive with
     a short description of the dataset(s) in the archive.

### Example of an resulting archive

For example, the `bbob-mixint` archive at
https://github.com/numbbo/data-archive/tree/gh-pages/data-archive/bbob-mixint
contains four datasets and the folder structure looks like
<font size="1">

```
bbob-mixint/
|-- 2019-gecco-benchmark/
|   |-- CMA-ES-pycma.tgz
|   |-- DE-scipy.tgz
|   |-- RANDOMSEARCH.tgz
|   `-- TPE-hyperopt.tgz
|-- coco_archive_definition.txt
```

</font>

The corresponding `coco_archive_definition.txt` file looks like
<details ><summary>(click to view)</summary><font size="1">

```python
[
('2019-gecco-benchmark/CMA-ES-pycma.tgz',
     '0d8e7f2c77f4e43176bc9424ee8f9a0bfe8e7f66fabc95b15ea7a56ad8b1d667',
     38514),
('2019-gecco-benchmark/DE-scipy.tgz',
     '494483b1bce9185f8977ce9abf6f6eac3a660efd6fa09321e305dfb79296cd18',
     35401),
('2019-gecco-benchmark/RANDOMSEARCH.tgz',
     '14b237093fd1f393871c578b6b28b6f9a6c3d8dc8921e3bdb024b3cc7cdd287d',
     26006),
('2019-gecco-benchmark/TPE-hyperopt.tgz',
     '34fede46a00c8adef4c388565c3b759c07a7d7d83366e115632b407764e64bf6',
     19633)]
```

</font>
with hashcodes and filesizes as additional entries.
</details>

### Example for using an archive

```python
import cocopp

url = 'http://cma-es.github.io/lq-cma/data-archives/lq-gecco2019'
arch = cocopp.archiving.get(url)
print(arch)  # `arch` "is" a `list` of relative filenames
['CMA-ES__2019-gecco-surr.tgz',
 'SLSQP+CMA_2019-gecco-surr.tgz',
 'SLSQP-11_2019-gecco-surr.tgz',
 'lq-CMA-ES_2019-gecco-surr.tgz']

# compare local result with data from lq-cma archive
# and from the cocopp.archives.bbob archive
cocopp.main([# 'exdata/my_local_results',  # in case
    arch.get('SLSQP-11'),  # downloads if necessary
    cocopp.archives.bbob.get_first('2010/IPOP-CMA'),
    arch.get('CMA-ES__2019')])
```
