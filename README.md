# Human in the Loop Topic Model with Word Embedding Latent Dirichlet Allocation (HLTM-WELDA)

## Getting Started

### Environment
This project used Python 3.6.8. The python package dependencies can be found in `requirements.txt`. Be sure to create an environment and activate it before proceeding. This project used [`pyenv`](https://github.com/pyenv/pyenv) and [`pyenv-virtualenv`](https://github.com/pyenv/pyenv-virtualenv) for python version and package management.


### Setup
If on a Mac, chances are that there might be issues with `gcc` due to [changes](https://developer.apple.com/documentation/xcode_release_notes/xcode_10_release_notes) Apple has made to Xcode. [This](https://stackoverflow.com/questions/52509602/ant-compile-c-program-on-a-mac-after-upgrade-to-mojave) is a good lead to how to deal with this.

After setting up the environment, run `bash setup.sh`. This sets up the `sqlite3` database and compiles Cython code into C.

Next, start up the Jupyter notebook: `jupyter notebook` and run it (after following direction within) to generate data files needed to get a demo running.


### Run
Next, run `bash run.sh` and navigate to `localhost:8050` in a browser to view the app. Currently, it takes about 30 seconds to load initially.
