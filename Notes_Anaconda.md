**version**
- conda --version
- conda update conda (perform as root)
- conda COMMAND --help

**Environment**
- conda create --name NEW_ENV [packages e.g. python=3.5 numpy ...]
- conda info --envs
- source activate ENV
- source deactivate
- conda create --name NEW_ENV --clone ENV

**Packages**
- (check packages) conda list [--name ENV]
- (reference) installable packages available at http://docs.continuum.io/anaconda/pkg-docs.html
- (check online) conda search [--full-name] PACKAGE
- (install) conda install [--name TARGET_ENV] PACKAGE
- (e.g.) conda install --channel https://conda.anaconda.org/pandas bottleneck
- (use pip in current ENV) pip install PACKAGE

**Removal**
- (del packages) conda remove [--name ENV] PACKAGE
- (del env) conda remove --name ENV --all
- (del conda) rm -rf ~/anaconda2
