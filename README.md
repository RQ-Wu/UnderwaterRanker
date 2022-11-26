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
Our pretrained models can be downloaded in the links below:
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
Then put the checkpoints into dir [`./checkpoints`](./checkpoints)