# Underwater Ranker: Learn Which Is Better and How to Be Better (AAAI 2023)

![Python 3.6](https://img.shields.io/badge/python-3.6-g) ![pytorch 1.10.2](https://img.shields.io/badge/pytorch-1.10.2-blue.svg)

This repository contains the official implementation of the following paper:
> **Underwater Ranker: Learn Which Is Better and How to Be Better**<br>
> Chunle Guo<sup>#</sup>, Ruiqi Wu<sup>#</sup>, Xin Jin, Linghao Han, Zhi Chai, Weidong Zhang, Chongyi Li<sup>*</sup><br>
> Proceedings of the AAAI conference on artificial intelligence (AAAI), 2023<br>

[[Arxiv Paper](https://arxiv.org/abs/2208.06857)]  [中文版 (TBD)] [Project (TBD)]  [Dataset (TBD)] [[Video]()]

## Dependencies and Installation
1. Cloen Repo
    ```bash
    git clone https://github.com/RQ-Wu/UnderwaterRanker.git
    cd UnderwaterRanker
    ```

2. Create Conda Enviroment
    ```bash
    conda env create -f environment.yaml
    conda activate underwater_ranker
    ```

## Get Started
### Prepare pretrained models & dataset 

1. You are supposed to download our pretrained model first in the links below and put them in dir `./checkpoints/`:

<table>
<thead>
<tr>
    <th>Model</th>
    <th>:link: Download Links </th>
    <th> SRCC/ KRCC (PSNR / SSIM) </th>
</tr>
</thead>
<tbody>
<tr>
    <td>URanker</td>
    <th>
    [<a href="">Google Drive (TBD)</a>] 
    [<a href="">Baidu Disk (TBD)</a>]
    </th>
    <th>0.8655 / 0.7402</th>
</tr>
<tr>
    <td>NU<sup>2</sup>Net</td>
    <th>
    [<a href="">Google Drive (TBD)</a>] 
    [<a href="">Baidu Disk (TBD)</a>]
    </th>
    <th>22.669 / 0.9246</th>
</tr>
</tbody>
</table>

2. Two datasets used in our work can be downloaded in the links below:

    - URankerSet: [<a href="">Google Drive (TBD)</a>][<a href="">Baidu Disk (TBD)</a>]
    - Underwater Image Enhancement Benchmark (UIEB): [<a href="https://li-chongyi.github.io/proj_benchmark.html">Link</a>]

    The data is put in dir `./data/`.

**The directory structure will be arranged as**:
```
checkpoints
    |- URanker_ckpt.pth
    |- NU2Net_ckpt.pth
data
    |- UIEB
        |- raw-890
        |- reference-890
    |- UIERank
```

### Quick demo
Run demos to process the images in dir `./examples/` by following commands:

```bash
python ranker_demo.py \
     --opt_path options/URanker.yaml \
     --checkpoint_path checkpoints/URanker_ckpt.pth \
     --input_path examples/ranker_example \
     --save_path results/ranker_result.txt
```

```bash
python uie_demo.py \
     --opt_path options/NU2Net.yaml \
     --checkpoint_path checkpoints/NU2Net_ckpt.pth \
     --input_path examples/uie_example \
     --save_path results
```

### Training & Evaluation
Our training and evaluation confiures are provided in `options/URanker.yaml` (for URanker) and `options/NU2Net.yaml` (for NU<sup>2</sup>Net)

Run the following commands for training:

```bash
python ranker_main_train.py --opt_path options/URanker.yaml
python uie_main_train.py --opt_path options.NU2Net.yaml
```

Run the following commands for evaluation:
```bash
python uie_main_test.py --opt_path options.NU2Net.yaml --test_ckpt_path checkpoints/NU2Net_ckpt.pth --save_image
```

## Citation
If you find our repo useful for your research, please cite us:
```
@inproceedings{qin2020ffa,
  title={Underwater Ranker: Learn Which Is Better and How to Be Better},
  author={Guo, Chunle and Wu, Ruiqi and Jin, Xin and Han, Linghao and Chai, Zhi and Zhang, Weidong and Li, Chongyi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

## Acknowledgement
This repository is maintained by [Ruiqi Wu](https://rq-wu.github.io/).