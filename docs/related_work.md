# Related work (closest prior work)

> **Re-check before submission.** Verify titles, author lists, venues and arXiv
> versions against the sources, and search again for newer work. Entries are
> preprints unless a venue is listed. Fields marked TODO were not available when
> this list was written; do not fill them from memory.

| # | Work | Venue / status | arXiv | Relevance to this project | Re-checked |
|---|---|---|---|---|---|
| 1 | Khan et al., *Compression Strategies for Efficient Multimodal LLMs in Medical Contexts* | preprint | 2507.21976 | Basis of our output-similarity layer-pruning baseline (E6, "Khan et al. style"). | TODO |
| 2 | Lin et al., *MoPE-CLIP* (full title TODO) | CVPR 2024 | 2403.07839 | TODO: summarise after reading. | TODO |
| 3 | Holste et al., *How Does Pruning Impact Long-Tailed Multi-Label Medical Image Classifiers?* | MICCAI 2023 | 2308.09180 | Effect of pruning on long-tailed medical classifiers (from the title); related to our rare-finding analysis. | TODO |
| 4 | Khalid et al., title TODO (topic: model pruning, long-tail forgetting and explanation reliability in medical imaging) | preprint | 2609.07803 | Topic: pruning, long-tail forgetting and explanation reliability in medical imaging. | TODO |
| 5 | GHOST-Q, authors and title TODO | preprint | 2609.29999 | TODO: summarise after reading. | TODO |
| 6 | Kim, Kim, Goebel, *When Quantization Preserves Accuracy but Not Evidence* | preprint | 2609.24799 | Accuracy vs evidence under quantization (from the title); related to our "hidden harm" framing. | TODO |
| 7 | Huang et al., title TODO (topic: structural pruning and recovery of large vision-language models) | preprint | 2604.24380 | Topic: structural pruning and recovery of large VLMs; related to E5 and our recovery design. | TODO |
| 8 | Cai et al., *LLaVA-KD* (full title TODO) | ICCV 2025 | 2410.16236 | TODO: summarise after reading; listed as related to our distillation arm (E2). | TODO |
| 9 | Qian et al., *Med-OPD* (full title TODO) | preprint | 2607.16303 | TODO: summarise after reading. | TODO |
| 10 | Chen et al., *GRACE* (full title TODO) | ICML 2026 | 2601.22709 | Template for our quantization-aware distillation arm (E4). | TODO |
| 11 | MedTokenBudget, authors and title TODO | preprint | 2609.30613 | TODO: summarise after reading. | TODO |
| 12 | Colon-X (ColonVQA, ColonPert), authors and title TODO | preprint | 2512.03667 | Source of candidate rare-finding / stress-test datasets. | TODO |
| 13 | Sara Hooker, Aaron Courville, Gregory Clark, Yann Dauphin, Andrea Frome, *What Do Compressed Deep Neural Networks Forget?* | arXiv preprint (2019) | 1911.05248 | The cost of compression falls on a small subset of examples, hidden by top-line accuracy. Basis of our pre-registered secondary flip analysis (2026-10-31): the test probes whose answer flips between E0 and each compressed arm, by finding and lesion-size stratum. | TODO |

## Paper structure for related work

Planned subsections (assign each work after reading it; TODO):

1. Compression of vision-language models.
2. Medical vision-language models and their compression.
3. What compression forgets.

BibTeX keys are in `paper/references.bib`.
