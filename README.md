<h1 align="center">LingBot-VLA: A Pragmatic VLA Foundation Model</h1>

<p align="center">
  <a href="assets/LingBot-VLA.pdf"><img src="https://img.shields.io/static/v1?label=Paper&message=PDF&color=red&logo=arxiv"></a>
  <a href="https://technology.robbyant.com/lingbot-vla"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="https://huggingface.co/collections/robbyant/lingbot-vla"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=yellow"></a>
  <a href="https://modelscope.cn/collections/Robbyant/LingBot-VLA"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=purple"></a>
  <a href="https://huggingface.co/datasets/robbyant/gm100"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20GM-100&message=HuggingFace&color=yellow"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green"></a>
</p>


<p align="center">
  <img src="assets/Teaser.png" width="100%">
</p>

## 🥳 We are excited to introduce **LingBot-VLA**, a pragmatic Vision-Language-Action foundation model.

**LingBot-VLA** has focused on being **Pragmatic**:
- **Large-scale Pre-training Data**: 20,000 hours of real-world
data from 9 popular dual-arm robot configurations.
<p align="center">
  <img src="assets/scale_sr.png" width="45%" style="margin: 0 10px;">
  <img src="assets/scale_ps.png" width="45%" style="margin: 0 10px;">
</p>

- **Strong Performance**: Achieves clear superiority over competitors on simulation and real-world benchmarks.
- **Training Efficiency**: Represents a 1.5 ∼ 2.8× (depending on the relied VLM base model) speedup over existing VLA-oriented codebases.

## 🚀 News
- **[2026-04-30]** Update of Our codebase:

  - Add recommended post-training setting with real robot data.
  - Upgrade to LeRobot v3.0.
  - Support open-loop evaluation.
  - Optimize GPU memory usage during training.
  - Enable Torch Compile for inference.
- **[2026-01-27]** LingBot-VLA Technical Report is available on Arxiv.
- **[2026-01-27]** Weights and code released!


---


## 🛠️ Installation
Requirements
 - Python 3.12.3
 - Pytorch 2.8.0
 - CUDA 12.8

```bash
conda create -n lingbotvla python=3.12 -y
conda activate lingbotvla

git clone https://github.com/Robbyant/lingbot-vla.git
cd lingbot-vla
bash install.sh
```

> **Hardware support**
>
> LingBot-VLA runs on **NVIDIA CUDA, Intel GPUs (XPU), or CPU**. The deployment and evaluation scripts automatically select the available device in this order: CUDA, XPU, then CPU. No code or configuration changes are needed.
>
> **Intel GPU (XPU):** install a PyTorch build with XPU support instead of the CUDA wheels:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
> ```
>
> Skip the `flash-attn` step in `install.sh`. It is CUDA-only and is not required on XPU; the model automatically falls back to PyTorch's built-in `eager` attention on XPU and CPU.
>
> Outputs closely match the CUDA and CPU paths. Minor float32 flow-matching rounding drift is expected and is not a bug. `--use_compile` is validated on CUDA; on XPU, prefer running without `--use_compile`.

---

## 📦 Model Download
We release LingBot-VLA pre-trained weights in two configurations: depth-free version and a depth-distilled version.
#### Pretrained Checkpoints for Post-Training with and without depth

| Model Name | Huggingface | ModelScope | Description |
| :--- | :---: | :---: | :---: |
| LingBot-VLA-4B &nbsp; | [🤗 lingbot-vla-4b](https://huggingface.co/robbyant/lingbot-vla-4b) | [🤖 lingbot-vla-4b](https://modelscope.cn/models/Robbyant/lingbot-vla-4b) | LingBot-VLA *w/o* Depth|
| LingBot-VLA-4B-Depth | [🤗 lingbot-vla-4b-depth](https://huggingface.co/robbyant/lingbot-vla-4b-depth) | [🤖 lingbot-vla-4b-depth](https://modelscope.cn/models/Robbyant/lingbot-vla-4b-depth) | LingBot-VLA *w/* Depth |

```bash
# Download Pretrained Checkpoints
python3 scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-4b --local_dir lingbot-vla-4b 
```

> <details>
> <summary>⚠️ <strong>Note for users who downloaded before 2026/05/01 (click to expand)</strong></summary>
> 
> <br>
> 
> If you downloaded `LingBot-VLA-4B` or `LingBot-VLA-4B-Depth` before **2026/05/01**, you may encounter the following error when loading the model:
> 
> ```
> draccus.utils.DecodingError: The fields `resize_imgs_with_padding`, `adapt_to_pi_aloha`, `use_delta_joint_actions_aloha`, `proj_width`, `num_steps`, `use_cache`, `attention_implementation`, `freeze_vision_encoder`, `train_expert_only`, `train_state_proj` are not valid for PI0Config
> ```
> 
> This is caused by our migration from **LeRobot v2.1** to **v3.0**.  
> To fix this, please **re-download the latest checkpoint**, or manually remove the above fields from `config.json` in your local `lingbot-vla-4b/` or `lingbot-vla-4b-depth/` directory.
> 
> </details>


<br>

To train LingBot with our codebase, weights from [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [MoGe-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal), and [LingBot-Depth](https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14) are also required.


---

## 💻 Post-Training Example

### Data Preparation

Post-training requires three preparation steps. For a complete guide on customizing your own dataset, see the [Custom Data Guide](lingbotvla/data/vla_data/README.md).

| Step | Description | Output |
|------|-------------|--------|
| 1. Prepare LeRobot Dataset | Convert your demonstration data to [LeRobot v3.0](https://github.com/huggingface/lerobot) format | LeRobot dataset directory |
| 2. Prepare Robot Config | Define feature mapping (states / actions / images) from raw keys to unified feature space | `configs/robot_configs/<data_name>.yaml` |
| 3. Compute Norm Statistics | Calculate normalization statistics over your dataset | `assets/norm_stats/<name>.json` |

> **Note:** If you already have data in LeRobot v2.1 format, you can use [convert_dataset_v21_to_v30.py](https://github.com/huggingface/lerobot/blob/v0.4.2/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py) to quickly convert it to v3.0.

Below we use **RoboTwin 2.0** (5 tasks: "open_microwave", "click_bell", "stack_blocks_three", "place_shoe", "put_object_cabinet") as an example.

- **Step 1 &mdash; RoboTwin Data**: Follow [RoboTwin2.0 Preparation](experiment/robotwin/README.md) to download and convert.
- **Step 2 &mdash; Robot Config**: See [`configs/robot_configs/robotwin.yaml`](configs/robot_configs/robotwin.yaml) for the RoboTwin feature mapping.
- **Step 3 &mdash; Normalization**: Pre-computed stats are provided at `assets/norm_stats/robotwin_50.json`. To recompute for a custom task subset, see the [Custom Data Guide](lingbotvla/data/vla_data/README.md).

### Training

We provide a post-training example of LingBot-VLA on 5 RoboTwin 2.0 tasks ("open_microwave", "click_bell", "stack_blocks_three", "place_shoe", "put_object_cabinet"):

```bash
# without depth
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin_load20000h.yaml \
    --data.train_path /path/to/mixed_robotwin_5tasks \
    --data.data_name robotwin \
    --data.norm_stats_file assets/norm_stats/robotwin_50.json \
    --train.output_dir output/

# with depth
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin_load20000h_depth.yaml \
    --data.train_path /path/to/mixed_robotwin_5tasks \
    --data.data_name robotwin \
    --data.norm_stats_file assets/norm_stats/robotwin_50.json \
    --train.output_dir output/
```

### 🤖 Real-Robot Post-Training
Also, we provide recommended training configurations specifically tailored for **real-world scenarios**: [`real_load20000h.yaml`](configs/vla/real_load20000h.yaml) (w/o depth) and [`real_load20000h_depth.yaml`](configs/vla/real_load20000h_depth.yaml) (w/ depth). For a detailed explanation of all training configuration parameters (batch size, gradient accumulation, training duration, checkpointing, depth injection, etc.), see the [Training Configuration Guide](configs/vla/Training_Config.md).


### Evaluation

#### Open-Loop Eval

```bash
export QWEN25_PATH=Qwen/Qwen2.5-VL-3B-Instruct
python scripts/open_loop_eval.py --model_path path_to_posttraining_ckpt --data_path path_to_validation_data --use_length 50
# If `--data_path` is omitted, the script defaults to the training dataset specified in the YAML config (`data.train_path`).
```


> **Note:**  
> For inference, the model path (`path_to_posttraining_ckpt`, located in `train.output_dir/checkpoints/*/hf_ckpt`) must include:
> - weights in `.safetensors` format
> - `config.json`
> - `lingbotvla_cli.yaml`


#### Robotwin
```bash
export QWEN25_PATH=path_to_Qwen2.5-VL-3B-Instruct
python -m deploy.lingbot_vla_policy \
 --model_path path_to_posttraining_ckpt \
 --use_compile \
 --use_length 50 \
 --port port
```


#### Real-Robot Deployment
```bash
export QWEN25_PATH=path_to_Qwen2.5-VL-3B-Instruct
python -m deploy.lingbot_vla_policy \
 --model_path path_to_posttraining_ckpt \
 --use_compile \
 --use_length 25
# You can set --num_denoising_step to 5 if you want to speed up the evaluation.
```

---

## 🏗️ Efficiency
<p align="center">
  <img src="assets/QwenPI_PaliGemmaPI.png" width="85%">
</p>
We evaluate the training efficiency of our codebase against established baselines for both <b>Qwen2.5-VL-3B-π</b> and <b>PaliGemma-3B-pt-224-π</b> models. The results demonstrate that our codebase
achieved the fastest training speeds in both model settings. The above figures detail the training throughput across configurations of 8, 16, 32, 128, and 256 GPUs, alongside the theoretical linear scaling limit.

> **📢 Note on Throughput Metrics:** 
> All throughput values (e.g., 261 samples/sec) represent the **total aggregate throughput across all GPUs**, not per-GPU performance. 
> <br><sup>(Updated: Previously mislabeled as per-GPU in earlier versions. We apologize for the confusion.)</sup>

---

## 📊 Performance

Our LingBot-VLA achieves state-of-the-art results on real-world and simulation benchmarks:
- **GM-100 across 3 robot platforms**

<table>
  <thead>
    <tr>
      <th rowspan="2">Platform</th>
      <th colspan="2">WALL-OSS</th>
      <th colspan="2">GR00T N1.6</th>
      <th colspan="2">π<sub>0.5</sub></th>
      <th colspan="2">Ours w/o depth</th>
      <th colspan="2">Ours w/ depth</th>
    </tr>
    <tr>
      <th>SR</th><th>PS</th>
      <th>SR</th><th>PS</th>
      <th>SR</th><th>PS</th>
      <th>SR</th><th>PS</th>
      <th>SR</th><th>PS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Agibot G1</td>
      <td>2.99%</td><td>8.75%</td><td>5.23%</td><td>12.63%</td><td>7.77%</td><td>21.98%</td><td><b>12.82%</b></td><td>30.04%</td><td>11.98%</td><td><b>30.47%</b></td>
    </tr>
    <tr>
      <td>AgileX</td>
      <td>2.26%</td><td>8.16%</td><td>3.26%</td><td>10.52%</td><td>17.20%</td><td>34.82%</td><td>15.50%</td><td>36.31%</td><td><b>18.93%</b></td><td><b>40.36%</b></td>
    </tr>
    <tr>
      <td>Galaxea R1Pro</td>
      <td>6.89%</td><td>14.13%</td><td>14.29%</td><td>24.83%</td><td>14.10%</td><td>26.14%</td><td>18.89%</td><td>34.71%</td><td><b>20.98%</b></td><td><b>35.40%</b></td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td>4.05%</td><td>10.35%</td><td>7.59%</td><td>15.99%</td><td>13.02%</td><td>27.65%</td><td>15.74%</td><td>33.69%</td><td><b>17.30%</b></td><td><b>35.41%</b></td>
    </tr>
  </tbody>
</table>


- **RoboTwin 2.0 (Clean and Randomized)**

<table>
  <thead>
    <tr>
      <th rowspan="2" ><b>Simulation Tasks</b></th>
      <th colspan="2"><b>&pi;<sub>0.5</sub></b></th>
      <th colspan="2"><b>Ours w/o depth</b></th>
      <th colspan="2"><b>Ours w/ depth</b></th>
    </tr>
    <tr>
      <th><b>Clean</b></th>
      <th><b>Rand.</b></th>
      <th><b>Clean</b></th>
      <th><b>Rand.</b></th>
      <th><b>Clean</b></th>
      <th><b>Rand.</b></th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-top: 1px solid #ccc;"> <!-- \midrule -->
      <td><b>Average SR</b></td>
      <td>82.74%</td>
      <td>76.76%</td>
      <td>86.50%</td>
      <td>85.34%</td>
      <td>88.56%</td>
      <td>86.68%</td>
    </tr>
  </tbody>
</table>


📢 We have released our checkpoints of LingBot-VLA-Posttrain-Robotwin:
| Model Name | Huggingface | ModelScope | Description |
| :--- | :---: | :---: | :---: |
| LingBot-VLA-4B-Posttrain-Robotwin &nbsp; | [🤗 lingbot-vla-4b-posttrain-robotwin](https://huggingface.co/robbyant/lingbot-vla-4b-posttrain-robotwin) | [🤖 lingbot-vla-4b-posttrain-robotwin](https://modelscope.cn/models/Robbyant/lingbot-vla-4b-posttrain-robotwin) | LingBot-VLA-Posttrain-Robotwin *w/o* Depth|
| LingBot-VLA-4B-Depth-Posttrain-Robotwin | [🤗 lingbot-vla-4b-depth-posttrain-robotwin](https://huggingface.co/robbyant/lingbot-vla-4b-depth-posttrain-robotwin) | [🤖 lingbot-vla-4b-depth-posttrain-robotwin](https://modelscope.cn/models/Robbyant/lingbot-vla-4b-depth-posttrain-robotwin) | LingBot-VLA-Posttrain-Robotwin *w/* Depth |

> <details>
> <summary>⚠️ <strong>Note for users who downloaded before 2026/05/01 (click to expand)</strong></summary>
> 
> <br>
> 
> If you downloaded `lingbot-vla-4b-posttrain-robotwin` or `lingbot-vla-4b-depth-posttrain-robotwin` before **2026/05/01**, you  may encounter the following error when loading the model:  
> 
> ```
> draccus.utils.DecodingError: The fields `resize_imgs_with_padding`, `adapt_to_pi_aloha`, `use_delta_joint_actions_aloha`, `proj_width`, `num_steps`, `use_cache`, `attention_implementation`, `freeze_vision_encoder`, `train_expert_only`, `train_state_proj` are not valid for PI0Config
> ```
> 
> To fix this, please re-download the latest checkpoint, or manually remove the above fields from
`config.json` and add them to the `train` section of `lingbotvla_cli.yaml` in your local directory.
> 
> </details>

<br>


<p align="center">
  <img src="assets/exp-gm-100.png" width="45%" style="margin: 0 10px;">
  <img src="assets/exp-robotwin.png" width="45%" style="margin: 0 10px;">
</p>

---

## 📝 Citation

If you find our work useful in your research, feel free to give us a cite.

```bibtex
@article{wu2026pragmatic,
  title={A Pragmatic VLA Foundation Model},
  author={Wei Wu and Fan Lu and Yunnan Wang and Shuai Yang and Shi Liu and Fangjing Wang and Shuailei Ma and He Sun and Yong Wang and Zhenqi Qiu and Houlong Xiong and Ziyu Wang and Shuai Zhou and Yiyu Ren and Kejia Zhang and Hui Yu and Jingmei Zhao and Qian Zhu and Ran Cheng and Yong-Lu Li and Yongtao Huang and Xing Zhu and Yujun Shen and Kecheng Zheng},
  journal={arXiv preprint arXiv:2601.18692v1},
  year={2026}
}
```

---

## 📄 License Agreement
This project is licensed under the [Apache-2.0 License](LICENSE).

## 😊 Acknowledgement
We would like to express our sincere gratitude to the developers of [VeOmni](https://arxiv.org/abs/2508.02317), [LeRobot](https://github.com/huggingface/lerobot#), and Baidu Cloud for their technical support. This project benefits significantly from their outstanding work and contributions. Baidu Cloud's optimization solutions notably reduced our GPU memory consumption by **29.2%** during model training.