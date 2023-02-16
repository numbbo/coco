import numpy as np
import os.path

from setuptools import Extension, setup

setup_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.join(setup_dir, "../../../"))
version_file = os.path.join(setup_dir, "src/cocoex/_version.py")

extensions = []
## COCO Cython Interface:
extensions.append(Extension("cocoex.interface",
                            sources=["src/cocoex/interface.pyx", "src/cocoex/coco.c"],
                            include_dirs=[np.get_include()],
                            ))

setup(ext_modules=extensions,
      ## FIXME: Once setuptools_scm works out issue #188, we should move this
      ##        to pyproject.toml
      use_scm_version={
          "root": root_dir,
          "write_to": os.path.join(setup_dir, "src/cocoex/_version.py"),
          }
      )
