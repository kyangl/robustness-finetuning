# On the Robustness Tradeoff in Fine-Tuning

This repository contains the official code for our paper: 
> **On the Robustness Tradeoff in Fine-Tuning** [[Paper Link]](https://arxiv.org/abs/2503.14836) \
> *Kunyang Li, Jean-Charles Noirot Ferrand, Ryan Sheatsley, Blaine Hoak, Yohan
> Beugin, Eric Pauley, Patrick McDaniel* \
> 📍 *IEEE/CVF International Conference on Computer Vision (ICCV), Oct 19-23th,
> 2025, Honolulu, Hawaii*


## 📌 Overview
Fine-tuning, especially parameter-efficient fine-tuning (PEFT), has become the
standard approach for adapting pre-trained models to downstream tasks. However,
its implications on **robustness** (i.e., how robustness is inherited, gained,
and/or lost) are not well understood. This project
systematically studies the **robustness-accuracy trade-offs** during
fine-tuning. 

- ✅ 7 SOTA fine-tuning strategies (e.g., LoRA, Adapter, BitFit)
- 📊 6 benchmark datasets (e.g., CIFAR10, Caltech-256, DomainNet)
- 🔍 Continuous evaluation on adversarial robustness and out-of-distribution
  (OOD) generalization

We find that:
- The trade-off between adversarial robustness and accuracy is **consistent**
  and **quantifiable** across diverse
  downstream tasks with area under Pareto frontiers. 
- Lightweight fine-tuning (BitFit) excels on simpler tasks (75% above average), while information
  intensive fine-tuning (Compacter) is better for complex tasks (57.5% above
  average). 
  

## 📁 Project Structure (dummy)
<pre>
robustness-finetuning/ 
├── scripts/ # Training and evaluation scripts
│   ├── build.py # Modify model architecture based on different fine-tuning strategies 
│   ├── finetune.py # Fine-tune the pre-trained model with continuous evaluation (attacks)
│   ├── attack.py # Attack algorithms
│   ├── utils.py # Utilities with logging 
│   └── main.py # Continuous robustness evaluation during fine-tuning
├── configs/ # YAML configs for parameter-efficient fine-tuning strategies 
├── config_gen.py # Generate customized configs for fine-tuning strategies 
├── Dockerfile # Requirements and dependencies
├── README.md 
└── LICENSE 

</pre>


## 🧪 Experiments

### 1. Environment Setup
First, clone the repository:
```
git clone <link>
cd <folder>
```

We use Docker to manage dependencies and ensure reproducibility. Now, you can build
and run the container as follows: 
```
# Build the Docker image 
docker build robustness-finetuning . 

# Run the container with GPU support 
docker run --gpus all -it robustness-finetuning
``` 

Note: `--gpu all` is required for GPU support. Make sure [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
is installed. 

### 2. Running Experiments 
Our framework is designed to train (fine-tune) and evaluate robustness jointly
through a two-stage process: **build** and **train + attack**. This process is implemented
in `main.py`. Here is an example for fine-tuning the pre-trained model with LoRA
on CIFAR10. 
```
python3 main_self.py --config config0_lora --dataset cifar10 --epoch 20 --learning_rate 5e-4 --weight_decay 1e-2
```
The evaluation result will be saved to a json file. 


## 📎 Citation
If you find this work useful, please cite the following paper: 
```
@inproceedings{li_robustness_2025,
	title = {On the {Robustness} {Tradeoff} in {Fine}-{Tuning}},
    booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
	url = {https://arxiv.org/abs/2503.14836},
	author = {Li, Kunyang and Noirot Ferrand, Jean-Charles and Sheatsley, Ryan and Hoak, Blaine and Beugin, Yohan and Pauley, Eric and McDaniel, Patrick},
	month = oct,
	year = {2025},
}
```

## 📬 Contact
For questions or collaboration, you are welcome to contact us at
[email](kli253@wisc.edu). 
