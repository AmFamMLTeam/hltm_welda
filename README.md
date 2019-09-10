# Human in the Loop Topic Model Latent Dirichlet Allocation (HLTM-WELDA)

## Getting Started

### Environment
Be sure to have the environment set up running `conda env create -f environment.yml` from the base directory (`hltm_welda`). Also be sure to have the environment activated by running `conda activate hltm_welda`.


### Setup
After setting up the environment, run `bash setup.sh`. This sets up the `sqlite3` database and compiles Cython code into C.


### Run
Next, run `bash run.sh` and navigate to `localhost:8050` in a browser to view the app. Currently, it takes about 30 seconds to load initially.
