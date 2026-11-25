## Overview

Deep learning approach for dermoscopic skin lesion classification across two tasks:
- **Binary Classification:** Distinguishing `nevus` vs. `others`[span_5](start_span)[span_5](end_span)[span_6](start_span)[span_6](end_span)[span_7](start_span)[span_7](end_span).
- **Three-Class Classification:** Classifying lesions into `bcc`, `mel`, and `scc`[span_8](start_span)[span_8](end_span)[span_9](start_span)[span_9](end_span).

---

## Preprocessing & Evaluated Architectures

- **Image Preprocessing:** Hair removal and black contour removal[span_10](start_span)[span_10](end_span).
- **Architectures Evaluated:** DenseNet121, ResNet50, ResNet101, ResNeXt-101, and EfficientNet V2[span_11](start_span)[span_11](end_span)[span_12](start_span)[span_12](end_span)[span_13](start_span)[span_13](end_span)[span_14](start_span)[span_14](end_span).

---

## Proposed Pipelines

### 1. Binary Classification (`nevus` vs. `others`)
- **Model:** **DenseNet121** (leveraging dense connections for feature reuse and gradient flow)[span_15](start_span)[span_15](end_span)[span_16](start_span)[span_16](end_span).
- **Fine-Tuning:** Early/basic blocks frozen; roughly 50% of the network (middle and deep layers) unfrozen to learn lesion textures without overfitting[span_17](start_span)[span_17](end_span).
- **Inference:** Test-Time Augmentation (TTA) averaging predictions across augmentations[span_18](start_span)[span_18](end_span).

### 2. Three-Class Classification (`bcc`, `mel`, `scc`)
- **Model:** **EfficientNet V2 - Small** (using Fused blocks in early layers and progressive compound scaling)[span_19](start_span)[span_19](end_span)[span_20](start_span)[span_20](end_span).
- **Fine-Tuning & Head:** Layers 0–5 frozen (or last two convolutional blocks unfrozen with a custom Multi-Layer Perceptron head using Dropout and Batch Normalization) to adapt to dermoscopic patterns[span_21](start_span)[span_21](end_span)[span_22](start_span)[span_22](end_span).
