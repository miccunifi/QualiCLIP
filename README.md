# QualiCLIP

### Quality-aware Image-Text Alignment for Real-World Image Quality Assessment

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2403.11176)
[![GitHub Stars](https://img.shields.io/github/stars/miccunifi/QualiCLIP?style=social)](https://github.com/miccunifi/QualiCLIP)

This is the **official repository** of the [**paper**](https://arxiv.org/abs/2403.11176) "*Quality-aware Image-Text Alignment for Real-World Image Quality Assessment*".

## Overview

### Abstract
No-Reference Image Quality Assessment (NR-IQA) focuses on designing methods to measure image quality in alignment with human perception when a high-quality reference image is unavailable. The reliance on annotated Mean Opinion Scores (MOS) in the majority of state-of-the-art NR-IQA approaches limits their scalability and broader applicability to real-world scenarios. To overcome this limitation, we propose QualiCLIP (Quality-aware CLIP), a CLIP-based self-supervised opinion-unaware method that does not require labeled MOS. In particular, we introduce a quality-aware image-text alignment strategy to make CLIP generate representations that correlate with the inherent quality of the images. Starting from pristine images, we synthetically degrade them with increasing levels of intensity. Then, we train CLIP to rank these degraded images based on their similarity to quality-related antonym text prompts, while guaranteeing consistent representations for images with comparable quality. Our method achieves state-of-the-art performance on several datasets with authentic distortions. Moreover, despite not requiring MOS, QualiCLIP outperforms supervised methods when their training dataset differs from the testing one, thus proving to be more suitable for real-world scenarios. Furthermore, our approach demonstrates greater robustness and improved explainability than competing methods.

![](assets/qualiclip_method.png "Overview of the proposed quality-aware image-text alignment strategy")

Overview of the proposed quality-aware image-text alignment strategy. Starting from a pair of two random overlapping crops from a pristine image, we synthetically degrade them with $L$ increasing levels of intensity, resulting in $L$ pairs. Then, given two quality-related antonym prompts, we fine-tune the CLIP image encoder by ranking the similarity between the prompts and the images, according to their corresponding level of degradation. At the same time, for each pair of equally distorted crops, we force the similarity between the crops and the prompts to be comparable.

## Citation

```bibtex
@article{agnolucci2024qualityaware,
      title={Quality-Aware Image-Text Alignment for Real-World Image Quality Assessment}, 
      author={Agnolucci, Lorenzo and Galteri, Leonardo and Bertini, Marco},
      journal={arXiv preprint arXiv:2403.11176},
      year={2024}
}
```

## To be released
- [ ] Pre-trained model
- [ ] Testing code
- [ ] Training code

## Authors

* [**Lorenzo Agnolucci**](https://scholar.google.com/citations?user=hsCt4ZAAAAAJ&hl=en)
* [**Leonardo Galteri**](https://scholar.google.com/citations?user=_n2R2bUAAAAJ&hl=en)
* [**Marco Bertini**](https://scholar.google.com/citations?user=SBm9ZpYAAAAJ&hl=en)

## Acknowledgements

This work was partially supported by the European Commission under European Horizon 2020 Programme, grant number 951911 - AI4Media.

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />All material is made available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** that you've made.
