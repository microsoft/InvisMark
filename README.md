# (WACV2025) InvisMark: Invisible and Robust Watermarking for AI-generated Image Provenance

This repo contains source code used for the paper: https://arxiv.org/pdf/2411.07795


## Environment Requirement


```
pip install -r requirements.txt
```


## Main Script

- Leverage the pretrained model ckpt
    - Download the pretrained model weights (with 100 encoded bits and no ECC included) from: https://1drv.ms/f/c/7882afab383c8474/Ei_Lasu5CrpHsrNIkYRLenYBmx662VSAovq5hD8r-NsB5A?e=gbHNVX
    - Follow the instruction in `Demo.ipynb`

- Train a watermark model from scratch with your own dataset
    - Prepare your own image dataset
    - Update `train_path` and `eval_path` in `train.sh`
    - ```./train.sh```


## Citation

If you find this paper or code useful, please cite by:
```txt
@article{xu2024invismark,
  title={InvisMark: Invisible and Robust Watermarking for AI-generated Image Provenance},
  author={Xu, Rui and Hu, Mengya and Lei, Deren and Li, Yaxi and Lowe, David and Gorevski, Alex and Wang, Mingyu and Ching, Emily and Deng, Alex and others},
  journal={arXiv preprint arXiv:2411.07795},
  year={2024}
}
```
