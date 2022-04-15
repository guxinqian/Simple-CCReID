### A Simple Codebase for Clothes-Changing Person Re-identification.
####  [Clothes-Changing Person Re-identification with RGB Modality Only (CVPR, 2022)](https://arxiv.org/abs/2204.06890)

#### Requirements
- Python 3.6
- Pytorch 1.6.0
- yacs
- apex

#### CCVID Dataset
- [[BaiduYun]](https://pan.baidu.com/s/1W9yjqxS9qxfPUSu76JpE1g) password: q0q2
- [[GoogleDrive]](https://drive.google.com/file/d/1vkZxm5v-aBXa_JEi23MMeW4DgisGtS4W/view?usp=sharing)

#### Get Started
- Replace `_C.DATA.ROOT` and `_C.OUTPUT` in `configs/default_img.py&default_vid.py`with your own `data path` and `output path`, respectively.
- Run `script.sh`


#### Citation

If you use our code/dataset in your research or wish to refer to the baseline results, please use the following BibTeX entry.
    
    @inproceedings{gu2022CAL,
        title={Clothes-Changing Person Re-identification with RGB Modality Only},
        author={Gu, Xinqian and Chang, Hong and Ma, Bingpeng and Bai, Shutao and Shan, Shiguang and Chen, Xilin},
        booktitle={CVPR},
        year={2022},
    }

#### Related Repos

- [Simple-ReID](https://github.com/guxinqian/Simple-ReID)
- [fast-reid](https://github.com/JDAI-CV/fast-reid)
- [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
- [Pytorch ReID](https://github.com/layumi/Person_reID_baseline_pytorch)

