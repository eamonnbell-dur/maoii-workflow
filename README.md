# maoii-workflow

Workflow for massive analysis of internet images

## Installing

1. Create Python virtualenv
2. Install backend dependencies in `maoii-backend/requirements.txt`
3. Install frontend dependencies in `maoii-frontend/package.json`
4. Build the frontend using `webpack`
5. Install `jq`

## Using 

1. Run the workflow using the following command:
    - `snakemake -c NUM_CORES --resources gpu_workers=NUM_GPUS`
    - where NUM_CORES is the number of available CPU cores, and NUM_GPUS is the number of CUDA devices available where `snakemake` is running.
2. When done, run a webserver from the root of this repository and navigate to the application.
