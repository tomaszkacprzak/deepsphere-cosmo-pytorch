# DeepSphere: a spherical convolutional neural network

[Janis Fluri](http://www.da.inf.ethz.ch/people/JanisFluri),
[Nathanaël Perraudin](https://perraudin.info),
[Michaël Defferrard](https://deff.ch)

This is an implementation of DeepSphere using PyTorch.

## Resources

Code:
* [deepsphere-cosmo-pytorch](https://github.com/deepsphere/deepsphere-cosmo-pytorch): this PyTorch implementation for HEALPix cosmology experiments. \
  Use for new developments in PyTorch.

Papers:
* DeepSphere: Efficient spherical CNN with HEALPix sampling for cosmological applications, 2018.\
  [[paper][paper_cosmo], [blog](https://datascience.ch/deepsphere-a-neural-network-architecture-for-spherical-data), [slides](https://doi.org/10.5281/zenodo.3243380)]
* DeepSphere: towards an equivariant graph-based spherical CNN, 2019.\
  [[paper][paper_rlgm], [poster](https://doi.org/10.5281/zenodo.2839355)]
* DeepSphere: a graph-based spherical CNN, 2020.\
  [[paper][paper_iclr], [slides](https://doi.org/10.5281/zenodo.3777976), [video](https://youtu.be/NC_XLbbCevk)]

[paper_cosmo]: https://arxiv.org/abs/1810.12186
[paper_rlgm]: https://arxiv.org/abs/1904.05146
[paper_iclr]: https://arxiv.org/abs/2012.15000

## Installation

1. Clone this repository.
   ```sh
   git clone https://github.com/deepsphere/deepsphere-cosmo-pytorch.git
   cd deepsphere-cosmo-pytorch
   ```

2. Install the package and its dependencies.
   ```sh
   pip install -e .
   ```

   PyTorch (`torch`) is the primary deep learning dependency. Sparse graph support remains based on SciPy and PyGSP; `torch-geometric` is not required by the current package.

3. Install development tools if you want to run tests and linters.
   ```sh
   pip install pytest pytest-cov pre-commit black flake8
   ```

4. (Optional) Test the installation.
   ```sh
   pytest tests
   ```

5. Play with the Jupyter notebooks.
   ```sh
   jupyter notebook
   ```

## PyTorch module example

DeepSphere layers follow the `torch.nn.Module` style in the PyTorch port. A typical model composes layers in an `nn.Module` and implements `forward`:

```python
import torch
from torch import nn


class SphericalClassifier(nn.Module):
    def __init__(self, deepsphere_block, n_classes):
        super().__init__()
        self.features = deepsphere_block
        self.classifier = nn.Linear(deepsphere_block.out_channels, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=1)
        return self.classifier(x)
```

The notebooks below are kept as historical references until their full examples are ported to PyTorch.

## Tensor layout

The PyTorch port uses the public DeepSphere tensor layout
`(batch, nodes, channels)`. Public layers should accept and return that layout.

Internally, the implementation transposes only at the boundary of PyTorch
layers that require channel-first tensors, such as `torch.nn.Conv1d`,
`torch.nn.MaxPool1d`, and `torch.nn.AvgPool1d`. Use the internal layout helpers
`deepsphere.utils.nodes_channels_to_channels_nodes(x)` and
`deepsphere.utils.channels_nodes_to_nodes_channels(x)` for these boundaries
instead of adding ad hoc `permute(0, 2, 1)` calls.

## Notebooks

The below notebooks contain examples and experiments to play with the model.

1. [Quick Start.][whole_sphere]
   The easiest to play with the model by classifying data on the whole sphere.
2. [Advanced Tutorial.][advanced]
   An introduction to various layers, customized training loops, and custom survey masks.
3. [Generative Models.][generative]
   How to build an auto-encoder using spherical data and the transpose healpy pseudo convolutions.

[whole_sphere]: https://nbviewer.jupyter.org/github/deepsphere/deepsphere-cosmo-pytorch/blob/master/examples/quick_start.ipynb
[advanced]: https://nbviewer.jupyter.org/github/deepsphere/deepsphere-cosmo-pytorch/blob/master/examples/advanced_tutorial.ipynb
[generative]: https://nbviewer.jupyter.org/github/deepsphere/deepsphere-cosmo-pytorch/blob/master/examples/generative_models.ipynb

## License & citation

The content of this repository is released under the terms of the [MIT license](LICENCE.txt).\
Please consider citing our papers if you find it useful.

```
@article{deepsphere_cosmo,
  title = {{DeepSphere}: Efficient spherical Convolutional Neural Network with {HEALPix} sampling for cosmological applications},
  author = {Perraudin, Nathana\"el and Defferrard, Micha\"el and Kacprzak, Tomasz and Sgier, Raphael},
  journal = {Astronomy and Computing},
  volume = {27},
  pages = {130-146},
  year = {2019},
  month = apr,
  publisher = {Elsevier BV},
  issn = {2213-1337},
  doi = {10.1016/j.ascom.2019.03.004},
  archiveprefix = {arXiv},
  eprint = {1810.12186},
  url = {https://arxiv.org/abs/1810.12186},
}
```

```
@inproceedings{deepsphere_rlgm,
  title = {{DeepSphere}: towards an equivariant graph-based spherical {CNN}},
  author = {Defferrard, Micha\"el and Perraudin, Nathana\"el and Kacprzak, Tomasz and Sgier, Raphael},
  booktitle = {ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year = {2019},
  archiveprefix = {arXiv},
  eprint = {1904.05146},
  url = {https://arxiv.org/abs/1904.05146},
}
```

```
@inproceedings{deepsphere_iclr,
  title = {{DeepSphere}: a graph-based spherical {CNN}},
  author = {Defferrard, Michaël and Milani, Martino and Gusset, Frédérick and Perraudin, Nathanaël},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2020},
  url = {https://openreview.net/forum?id=B1e3OlStPB},
}
```
