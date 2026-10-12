"""Data layer: sample records, manifests, patient-level splits, masks, dedup and probes.

Nothing in this package downloads or stores data. Datasets live outside the
repository under ``paths.data_root`` (see ``docs/data_policy.md``); this package
only describes how they are read, split and turned into yes/no finding probes.

Modules
-------
base       ``VQASample`` record and the ``VQADataset`` interface.
manifests  File manifests with content hashes (kept outside git).
splits     Patient-level splits and nested calibration subsets (A4).
masks      Lesion masks, lesion area and the frozen size strata (primary endpoint).
dedup      Near-duplicate detection between training data and external test sets.
probes     Fixed yes/no finding probes used by the clinical criterion and the audit.
"""
