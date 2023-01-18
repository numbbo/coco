import numpy as np

from setuptools import Extension, setup
   
extensions = []
extensions.append(Extension("cocoex.interface",
                            sources=["src/cocoex/interface.pyx", "src/cocoex/coco.c"],
                            include_dirs=[np.get_include()],
                            ))

setup(ext_modules=extensions)
