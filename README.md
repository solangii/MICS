## MICS: Midpoint Interpolation to Learn Compact and Separated Representations for Few-Shot Class-Incremental Learning

> **MICS: Midpoint Interpolation to Learn Compact and Separated Representations for Few-Shot Class-Incremental Learning** 
>
> Solang Kim, Yuho Jeong, Joon Sung Park, Sung Whan Yoon
>
> In WACV 2024.

### Installation

1. Clone this repository.

   ```bash
   glt clone http://github.com/solang/mics.git
   ```

2. Install the required dependency. 

   ```bash
   conda create env -y -n mics python=3.9
   conda activate mics
   bash install.sh
   ```

   * **Recommend**: Check your CUDA version using the `nvcc -V` command and update the torch version in the `install.sh` script accordingly. You can find the compatible PyTorch versions for your CUDA release at [this link](https://pytorch.org/get-started/previous-versions/)

   * Our codes are tested on Ubuntu 18.04 with Python 3.9.5 and Pytorch 1.9.0. We utilized NVIDIA RTX A5000 for mini-ImageNet (CUDA 10.1) and GeForce RTX 3090 for CIFAR-100 and CUB-200-2011 (CUDA 11.1) 

### Datasets

Download FSCIL benchmark datasets.

1. CIFAR100[1]: https://www.cs.toronto.edu/~kriz/cifar.html
2. mini-ImageNet[2]: There is no official website for mini-ImageNet. You can utilize the *learn2learn*[4] python package or the unofficial Google Drive links for the download.
3. CUB-200-2011[3]: https://www.vision.caltech.edu/datasets/cub_200_2011/

### Train

```shell
cd scripts
bash [dataset]-mics.sh 
```

- Before execution the scripts, set your dataset path and pre-trained model path options in scripts.
- Also, you can download our pretrained weight: [Link](https://drive.google.com/drive/folders/18rcX2Vhva1lRtr_rUYcjWG6m2AshOcZ6?usp=sharing) (base session weight)

### Citation

```
@inproceedings{kim2024mics,
  title={MICS: Midpoint Interpolation to Learn Compact and Separated Representations for Few-Shot Class-Incremental Learning},
  author={Kim, Solang and Jeong, Yuho and Park, Joon Sung and Yoon, Sung Whan},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2024}
}
```



