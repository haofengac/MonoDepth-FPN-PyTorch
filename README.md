# I2D: Single Image Depth Estimation with Feature Pyramid Networks

[![License][license]][license-url]

## Requirements

* Python 3
* Jupyter Notebook (for visualization)
* PyTorch 
  * Tested with PyTorch 0.3.0.post4
* CUDA 8 (if using CUDA)

## To Run

```
python3 main_fpn.py --cuda --bs=6
```
To continue training from a saved model, use
```
python3 main_fpn.py --cuda --bs=6 --r True --checkepoch 10
```
To visualize the reconstructed data, run the jupyter notebook in Vis.ipnb.

## To-dos

- [x] Add visualization ipynb
- [ ] Add code for CVPR 2018 ROB Challenge

<!-- Markdown link & img dfn's -->
[license]: https://img.shields.io/github/license/mashape/apistatus.svg
[license-url]: https://github.com/xanderchf/i2d/blob/master/LICENSE
