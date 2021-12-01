Medical image datasets
======================

TorchIO offers tools to easily download publicly available datasets from
different institutions and modalities.

The interface is similar to :mod:`torchvision.datasets`.

If you use any of them, please visit the corresponding website (linked in each
description) and make sure you comply with any data usage agreement and you
acknowledge the corresponding authors' publications.

If you would like to add a dataset here, please open a discussion on the
GitHub repository:

.. raw:: html

   <a class="github-button" href="https://github.com/fepegar/torchio/discussions" data-icon="octicon-comment-discussion" aria-label="Discuss fepegar/torchio on GitHub">Discuss</a>
   <script async defer src="https://buttons.github.io/buttons.js"></script>


IXI
---

.. automodule:: torchio.datasets.ixi
.. currentmodule:: torchio.datasets.ixi

:class:`IXI`
~~~~~~~~~~~~~

.. autoclass:: IXI
    :show-inheritance:


:class:`IXITiny`
~~~~~~~~~~~~~~~~~

.. autoclass:: IXITiny
    :show-inheritance:


EPISURG
-------

.. currentmodule:: torchio.datasets.episurg

:class:`EPISURG`
~~~~~~~~~~~~~~~~

.. autoclass:: EPISURG
    :members:
    :show-inheritance:


RSNAMICCAI
----------

.. currentmodule:: torchio.datasets.rsna_miccai

:class:`RSNAMICCAI`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: RSNAMICCAI
    :show-inheritance:


MNI
---

.. automodule:: torchio.datasets.mni
.. currentmodule:: torchio.datasets.mni


:class:`ICBM2009CNonlinearSymmetric`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ICBM2009CNonlinearSymmetric
    :show-inheritance:


:class:`Colin27`
~~~~~~~~~~~~~~~~

.. autoclass:: Colin27
    :show-inheritance:

.. plot::

    import torchio as tio
    subject = tio.datasets.Colin27()
    subject.plot()


:class:`Pediatric`
~~~~~~~~~~~~~~~~~~

.. autoclass:: Pediatric
    :show-inheritance:

.. plot::

    import torchio as tio
    subject = tio.datasets.Pediatric((4.5, 8.5))
    subject.plot()


:class:`Sheep`
~~~~~~~~~~~~~~

.. autoclass:: Sheep
    :show-inheritance:

.. plot::

    import torchio as tio
    subject = tio.datasets.Sheep()
    subject.plot()


.. currentmodule:: torchio.datasets.bite

:class:`BITE3`
~~~~~~~~~~~~~~

.. autoclass:: BITE3
    :show-inheritance:


.. currentmodule:: torchio.datasets.visible_human


Visible Human Project
---------------------

The `Visible Human Project <https://www.nlm.nih.gov/research/visible/visible_human.html>`_
is an effort to create a detailed data set of cross-sectional photographs of
the human body, in order to facilitate anatomy visualization applications.
It is used as a tool for the progression of medical findings, in which these
findings link anatomy to its audiences.
A male and a female cadaver were cut into thin slices which were then
photographed and digitized (from `Wikipedia <https://en.wikipedia.org/wiki/Visible_Human_Project>`_).

:class:`VisibleMale`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VisibleMale
    :show-inheritance:

.. plot::

    import torchio as tio
    tio.datasets.VisibleMale('Shoulder').plot()


:class:`VisibleFemale`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VisibleFemale
    :show-inheritance:

.. plot::

    import torchio as tio
    tio.datasets.VisibleFemale('Shoulder').plot()


ITK-SNAP
--------

.. automodule:: torchio.datasets.itk_snap
.. currentmodule:: torchio.datasets.itk_snap


:class:`BrainTumor`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: BrainTumor
    :show-inheritance:

.. plot::

    import torchio as tio
    tio.datasets.BrainTumor().plot()


:class:`T1T2`
~~~~~~~~~~~~~

.. autoclass:: T1T2
    :show-inheritance:

.. plot::

    import torchio as tio
    subject = tio.datasets.T1T2()
    subject.plot()


:class:`AorticValve`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AorticValve
    :show-inheritance:

.. plot::

    import torchio as tio
    subject = tio.datasets.AorticValve()
    subject.plot()


3D Slicer
---------

.. automodule:: torchio.datasets.slicer
.. currentmodule:: torchio.datasets.slicer

:class:`Slicer`
~~~~~~~~~~~~~~~

.. autoclass:: Slicer
    :show-inheritance:

.. plot::

    import torchio as tio
    subject = tio.datasets.Slicer()
    subject.plot()


FPG
---

.. currentmodule:: torchio.datasets.fpg

.. autoclass:: FPG
    :show-inheritance:

.. plot::

    import torchio as tio
    subject = tio.datasets.FPG()
    subject.plot()

.. plot::

    import torchio as tio
    subject = tio.datasets.FPG(load_all=True)
    subject.plot()


MedMNIST
--------

.. currentmodule:: torchio.datasets.medmnist


.. autoclass:: OrganMNIST3D

.. plot::

    import torch
    import torchio as tio
    from einops import rearrange
    rows, cols = 16, 28
    dataset = tio.datasets.OrganMNIST3D('train')
    loader = torch.utils.data.DataLoader(dataset, batch_size=rows * cols)
    batch = tio.utils.get_first_item(loader)
    tensor = batch['image'][tio.DATA]
    pattern = '(b1 b2) c x y z -> c x (b1 y) (b2 z)'
    tensor = rearrange(tensor, pattern, b1=rows, b2=cols)
    sx = tensor.shape[1]
    plt.imshow(tensor[0, sx // 2], cmap='gray')


.. autoclass:: NoduleMNIST3D

.. plot::

    import torch
    import torchio as tio
    from einops import rearrange
    rows, cols = 16, 28
    dataset = tio.datasets.NoduleMNIST3D('train')
    loader = torch.utils.data.DataLoader(dataset, batch_size=rows * cols)
    batch = tio.utils.get_first_item(loader)
    tensor = batch['image'][tio.DATA]
    pattern = '(b1 b2) c x y z -> c x (b1 y) (b2 z)'
    tensor = rearrange(tensor, pattern, b1=rows, b2=cols)
    sx = tensor.shape[1]
    plt.imshow(tensor[0, sx // 2], cmap='gray')


.. autoclass:: AdrenalMNIST3D

.. plot::

    import torch
    import torchio as tio
    from einops import rearrange
    rows, cols = 16, 28
    dataset = tio.datasets.AdrenalMNIST3D('train')
    loader = torch.utils.data.DataLoader(dataset, batch_size=rows * cols)
    batch = tio.utils.get_first_item(loader)
    tensor = batch['image'][tio.DATA]
    pattern = '(b1 b2) c x y z -> c x (b1 y) (b2 z)'
    tensor = rearrange(tensor, pattern, b1=rows, b2=cols)
    sx = tensor.shape[1]
    plt.imshow(tensor[0, sx // 2], cmap='gray')


.. autoclass:: FractureMNIST3D

.. plot::

    import torch
    import torchio as tio
    from einops import rearrange
    rows, cols = 16, 28
    dataset = tio.datasets.FractureMNIST3D('train')
    loader = torch.utils.data.DataLoader(dataset, batch_size=rows * cols)
    batch = tio.utils.get_first_item(loader)
    tensor = batch['image'][tio.DATA]
    pattern = '(b1 b2) c x y z -> c x (b1 y) (b2 z)'
    tensor = rearrange(tensor, pattern, b1=rows, b2=cols)
    sx = tensor.shape[1]
    plt.imshow(tensor[0, sx // 2], cmap='gray')


.. autoclass:: VesselMNIST3D

.. plot::

    import torch
    import torchio as tio
    from einops import rearrange
    rows, cols = 16, 28
    dataset = tio.datasets.VesselMNIST3D('train')
    loader = torch.utils.data.DataLoader(dataset, batch_size=rows * cols)
    batch = tio.utils.get_first_item(loader)
    tensor = batch['image'][tio.DATA]
    pattern = '(b1 b2) c x y z -> c x (b1 y) (b2 z)'
    tensor = rearrange(tensor, pattern, b1=rows, b2=cols)
    sx = tensor.shape[1]
    plt.imshow(tensor[0, sx // 2], cmap='gray')


.. autoclass:: SynapseMNIST3D

.. plot::

    import torch
    import torchio as tio
    from einops import rearrange
    rows, cols = 16, 28
    dataset = tio.datasets.SynapseMNIST3D('train')
    loader = torch.utils.data.DataLoader(dataset, batch_size=rows * cols)
    batch = tio.utils.get_first_item(loader)
    tensor = batch['image'][tio.DATA]
    pattern = '(b1 b2) c x y z -> c x (b1 y) (b2 z)'
    tensor = rearrange(tensor, pattern, b1=rows, b2=cols)
    sx = tensor.shape[1]
    plt.imshow(tensor[0, sx // 2], cmap='gray')
