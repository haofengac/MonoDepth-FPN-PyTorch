# i2d

[![License][license]][license-url]

Single Image Depth Estimation with Feature Pyramid Networks

## Requirements

* Python 3

PyTorch Implementation:
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

- [x] Add visualization ipynb for PyTorch implementation
- [ ] Add visualization ipynb for CuPy and NumPy implementations
- [ ] Finish deformable convolution implementation in CuPy
- [ ] Start a project on CuPy automatic differentiation, which could possibly benefit this project

<!-- Markdown link & img dfn's -->
[license]: https://img.shields.io/github/license/mashape/apistatus.svg
[license-url]: https://github.com/xanderchf/pyCapsNet/blob/master/LICENSE
[Sabour, Sara, Nicholas Frosst, and Geoffrey E. Hinton. "Dynamic routing between capsules." Advances in Neural Information Processing Systems. 2017.]: https://arxiv.org/abs/1710.09829
[paper]: https://arxiv.org/abs/1710.09829
[with PyTorch]: https://github.com/gram-ai/capsule-networks
[TensorFlow]: https://github.com/ageron/handson-ml
[Keras]: https://github.com/XifengGuo/CapsNet-Keras
[video]: https://www.youtube.com/watch?v=2Kawrd5szHE
