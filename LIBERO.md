
## Setup

Set up a conda environment (see instructions in [SETUP.md](SETUP.md)).

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO) and required packages:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r vat/libero_requirements.txt  
```

(Optional, if you plan to launch training) To download the [LIBERO datasets](https://huggingface.co/datasets/openvla/modified_libero_rlds) that we used in our fine-tuning
experiments, run the command below. This will download the LIBERO-Spatial, LIBERO-Object, LIBERO-Goal,
and LIBERO-10 datasets in RLDS data format (~10 GB total). You can use these to fine-tune OpenVLA or
train other methods. This step is optional since we provide pretrained vat checkpoints below.
Note that these are the same datasets used in the original OpenVLA project. If needed, see details on how to download the original non-RLDS datasets [here](https://github.com/openvla/openvla?tab=readme-ov-file#libero-setup).
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

## Fine-Tuning on LIBERO Datasets

```bash
sh scripts/train_vat.sh --data_root_dir ./libero_rlds --dataset_name libero_10_no_noops --run_root_dir ./ckpt/
```

## Launching LIBERO Evaluations

We train VAT on four LIBERO task suites independently: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 (also called LIBERO-Long).
The four VAT checkpoints for LIBERO are available on Hugging Face:

```bash
sh evaluation/eval_libero.sh --task_suite_name libero_spatial --pretrained_checkpoint ./ckpt
```

Notes:
* The evaluation script will run 500 trials by default (10 tasks x 50 episodes each). You can modify the number of
  trials per task by setting `--num_trials_per_task`. You can also change the random seed via `--seed`. There are
  other arguments in the script; we set them to the default values that work with the VAT checkpoints above.
* **NOTE: Setting `--center_crop True` is important** because we train VAT with random crop augmentations
  (we took a random crop with 90% area in every training sample, so at test time we simply take the center 90% crop).
If the discrepancy is large,
  please post a GitHub issue, and we will look into it.