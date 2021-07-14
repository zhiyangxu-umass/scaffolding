# Scaffolding: Syntactic Scaffolds for Semantic Structures

A repository based on the EMNLP 2018 [paper](https://arxiv.org/abs/1808.10485) for Frame-Semantic and PropBank Semantic Role Labeling with Syntactic Scaffolding. 


## Installation
This repository was built on an earlier version of [AllenNLP](https://github.com/allenai/allennlp).
Due to changes in the API, we recommended installing directly via steps below (adapted from the AllenNLP installation), as opposed to using an installed version of AllenNLP.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6

    ```
    conda create -n scaffold python=3.6
    ```

3.  Activate the Conda environment.  (You will need to activate the Conda environment in each terminal in which you want to use AllenNLP.

    ```
    source activate scaffold
    ```

4. Install in your environment

    ```
    git clone https://github.com/swabhs/scaffolding.git
    ```

5. Change your directory to where you cloned the files:

    ```
    cd scaffolding/
    ```

6.  Install the required dependencies.

    ```
    INSTALL_TEST_REQUIREMENTS="true" ./scripts/install_requirements.sh
    ```

7. Install PyTorch version 0.3 **via pip** ([modify](https://pytorch.org/previous-versions/) based on your CUDA environment).
Conda-based installation results in [slower rutime](https://github.com/pytorch/pytorch/issues/537) because of a CUDA issue.

    ```
    pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
    ```

## Step 1: Get Data

Download [FN data](https://drive.google.com/file/d/15n3M4AmURGdGqnNAjn352buUTV5S-fVI/view?usp=sharing) and place it under a `data/` directory under the root directory.

The [OntoNotes datasets](https://catalog.ldc.upenn.edu/LDC2013T19) can be obtained through LDC.

## Step 2: Get Pre-trained Models

[Frame-SRL baseline](https://drive.google.com/open?id=1f7ZLOBc65Y74hPQlYY8mGVHkCiN14dwH)

[Frame-SRL scaffold with common non-terminals](https://drive.google.com/open?id=1V1-U70U-wDKaG3zuONQN3eB77jjS3FSd)

[PropBank SRL baseline](https://drive.google.com/file/d/1ThTagaJeZkIACEYfDn6f4mjMHs0PMSAo/view?usp=sharing)

[PropBank SRL scaffold with common non-terminals](https://drive.google.com/file/d/1vv3KC_OLx0A7ItKqBz9yWwZLIsVZhR5B/view?usp=sharing)


## Step 3: Test

```
python -m allennlp.run evaluate \
    --archive-file log_final_pb_baseline/model.tar.gz \
    --evaluation-data-file data/fndata-1.5/test/fulltext/ \
    --cuda-device 0
```

For the syntactic scaffold model for PropBank SRL, use the `pbscaf` branch:
```
git checkout pbscaf
```
and then run evaluation as above.

## Training

For scaffolds, use `$command=train_m` and for baselines, `$command=train`.
```
python -m allennlp.run $command training_config/$config --serialization-dir log
```

## Acknowledgment

This is based on our [paper](https://arxiv.org/abs/1808.10485), please cite:

```
@inproceedings{swayamdipta:2018,
    author      = {Swabha Swayamdipta and Sam Thomson and Kenton Lee and Luke Zettlemoyer and Chris Dyer and Noah A. Smith},
    title       = {Syntactic Scaffolding for Semantic Structures},
    booktitle   = {Proc. of EMNLP},
    url         = {https://arxiv.org/abs/1808.10485},
    year        = {2018}
}
```
## Extra note on running wandb

1. Download dataset. 
    (skipping as it's unrelated to us)
    
3. Export enviroment variables

    ```
    export CUDA_DEVICE=0 # 0 for GPU, -1 for CPU
    export TEST=1 # for a dryrun and without uploading results to wandb
    export WANDB_IGNORE_GLOBS=*\*\*\*.th,*\*\*\*.tar.gz,*\*\*.th,*\*\*.tar.gz,*\*.th,*\*.tar.gz,*.tar.gz,*.th
    ```

3. Training single models

    1. Using slurm (on gypsum)

      Open `single_run.sh`, make modifications as needed, close and submit job using `sbatch single_run.sh`. Do not push local updates to this file to the repo.


    2. On you local machine

        1. Without sending output to wandb

        ```
        export TEST=1
        export CUDA_DEVICE=-1
        allennlp train <path_to_config> -s <path to serialization dir> --include_package structure_prediction_baselines
        ```

        2. With output to wandb (see [creating account and login into wandb](https://docs.wandb.ai/quickstart#2-create-account) for details on getting started with wandb.)

        ```
        export TEST=0
        export CUDA_DEVICE=-1
        wandb_allennlp --subcommand=train --config_file=model_configs/<path_to_config_file> --include-package=structured_prediction_baselines --wandb_run_name=<some_informative_name_for_run>  --wandb_project structure_prediction_baselines --wandb_entity score-based-learning --wandb_tags=baselines,as_reported
        ```

4. Running hyperparameter sweeps

    1. Create a sweep using a sweep config file. See `sweep_configs` directory for examples. Refer sweeps documentation [here](https://docs.wandb.ai/sweeps).

    ```
    wandb sweep -e score-based-learning -p baselines sweep_configs/path/to/config.yaml

    < you will see an alpha numeric sweep_id as output here. Copy it.>
    ```

    2. Start search agent on slurm using the following (This script will internally submit to sbatch. So you can run this command on the head node eventhough it is a python script because it exit withing seconds.)

    ```
    export TEST=0
    python slurm_wandb_agent.py <sweep_id> -p baselines -e score-based-learning --num-jobs 5 -f --edit-sbatch --edit-srun
    ```

    You can use `squeue` to see the running agents on nodes. You can rerun this command to start more agents.


