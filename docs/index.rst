MLTK
====

A small toolkit of building blocks on top of `PyTorch Lightning
<https://lightning.ai/>`_. ``AbstractModel`` is a ``LightningModule``;
everything else (datasets, schedulers, ``SuperModel``, hyperparameter
search) layers on top.

Layout
------

.. list-table::
   :widths: 18 82
   :header-rows: 1

   * - Subpackage
     - What lives there
   * - :mod:`mltk.learning`
     - Lightning lifecycle base (:class:`~mltk.AbstractModel`), losses,
       schedulers, samplers, hyperparameter search.
   * - :mod:`mltk.building_blocks`
     - Reusable ``nn.Module`` primitives — modern MLP blocks (SwiGLU,
       LayerScale, DropPath), classifier heads, gradient-checkpointed
       wrappers.
   * - :mod:`mltk.models`
     - Composed architectures — :class:`~mltk.SuperModel` (shared trunk +
       per-key heads), abstract :class:`~mltk.ModelFactory` protocol.
   * - :mod:`mltk.data`
     - Dataset bases and wrappers — :class:`~mltk.AbstractImageDataset`,
       :class:`~mltk.MapDataset`, :class:`~mltk.ReplicatedDataset`,
       :class:`~mltk.DiskCachedDataset`, :class:`~mltk.DeviceLRUCache`.
   * - :mod:`mltk.hierarchical`
     - Tree-structured dispatch — hierarchy metadata, per-level samplers,
       beam-search inference.
   * - :mod:`mltk.devices`
     - Cross-platform accelerator selection (CUDA / ROCm / DirectML / MPS / CPU).

Design contracts
----------------

**Datasets own randomization, models own normalization.** Augmentation
lives in the dataset because it's part of how an item is drawn;
normalization lives on the model because per-backbone mean/std varies.
:class:`~mltk.AbstractModel` auto-applies ``self.normalize`` inside
``compute_loss`` and ``predict``; datasets return un-normalized tensors.

**Lightning is the backend.** MLTK does not maintain its own training
loop, optimizer registry, or checkpoint format. Train with
``lightning.Trainer.fit`` and restore with
``cls.load_from_checkpoint``. ``self.optimizer`` and ``self.scheduler``
are conveniences that the default ``configure_optimizers`` returns to
Lightning; override the hook directly when you need bespoke behavior.

**Augmentation presets are first-class constants.** ``LIGHT_AUGMENT``,
``STANDARD_AUGMENT``, ``STRONG_AUGMENT`` are module-level
``transforms.Compose`` objects. Pass them as ``augment=...`` or build
your own callable.

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/learning
   api/building_blocks
   api/models
   api/data
   api/hierarchical
   api/devices

.. toctree::
   :maxdepth: 1
   :caption: Project

   readme
   lightning
