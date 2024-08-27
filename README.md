# maoii-workflow

Workflow for massive analysis of internet images

## Installing

1. Create a Python virtualenv
2. Clone this repository (e.g. `gh repo clone eamonnbell-dur/maoii-workflow`)
3. Initialize and update submodules, if not already done (e.g. `git submodule update --init --recursive`)
4. Install backend dependencies into the Python virtualenv using `maoii-backend/requirements.txt`
5. Install frontend dependencies in `maoii-frontend/package.json`, using `npm` or similar (e.g. `yarn`)
6. Build the frontend using `webpack`
7. Install `jq`

## Using 

1. Run the workflow using the following command:
    - `snakemake -c NUM_CORES --resources gpu_workers=NUM_GPUS`
    - where NUM_CORES is the number of available CPU cores, and NUM_GPUS is the number of CUDA devices available where `snakemake` is running.
2. When done, run a webserver from the root of this repository and navigate to the application.
