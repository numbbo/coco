
# Publish `cocopp` to [PyPI](https://pypi.org/project/cocopp)

0. Prepare requirements (in case):
   - `pip install colorama toml build`
1. create a tag `vX.Y.Z` (if not already done) which will be picked up by
   `git_version` in `scripts/fabricate`, e.g. like
   - `git tag v2.6.6`
2. possibly make a clean clone 
   - `git clone --local coco-root-folder coco-clean-v2.6.6` and
   - `cd coco-clean-v2.6.6`
3. `cd code-postprocessing`
4. run `../scripts/fabricate`
5. install `cocopp` (probably not necessary?):
   - repeat `pip uninstall -y cocopp` until none is installed
   - `pip install .`
6. make a distribution
   - `python -m build`
7. upload to [PyPI](https://pypi.org/project/cocopp) (possible for maintainers only)
   - `twine upload dist/*2.6.6*`
