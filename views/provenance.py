"""Model card, data lineage, licences and what this project contributes."""
from __future__ import annotations

import streamlit as st

from xraylens import ui
from xraylens.app_state import get_metrics

M = get_metrics()

ui.html_block("""
<header class="xl-mast">
  <div class="xl-kicker">Provenance · where every pixel and weight came from</div>
  <div class="xl-title" style="font-size:clamp(3rem,7.5vw,6.2rem)">Read the <em>label</em><br>before the result</div>
</header>
""")

ui.section("01", "The model", "We did not train this network. We audited it.")
a, b = st.columns([7, 5], gap="large")
with a:
    st.markdown(f"""
**[Purino/resnet50-chest-xray-pneumonia](https://huggingface.co/Purino/resnet50-chest-xray-pneumonia)**, pinned at revision
`{M['model']['revision'][:12]}`. It is a torchvision ResNet-50 (ImageNet V2 initialisation) fine-tuned for two classes,
NORMAL and PNEUMONIA, at 320 × 320 px with aspect-preserving black padding.

Notes from the model card, quoted or paraphrased:
- Final weights were **refit on all 5,824 cleaned Kermany images**, including the original test folder. *“It has no independent test evaluation.”*
- Its reported metrics (ROC-AUC 0.998) come from a **selection checkpoint** on an internal split, and *“do not evaluate `model.safetensors`.”*
- Source images are from **young children at one centre**. Performance on adults, other hospitals and other diseases is unknown.
- Patient grouping was inferred from filenames. *“Near-duplicate or undocumented patient overlap may remain.”*
- *“Softmax outputs are not clinically calibrated probabilities.”*
- Research only. Not a clinical diagnostic device.
""")
with b:
    ui.note("<b>Why this model?</b> Its card is unusually candid about training scope, leakage and limits. That made it "
            "a good subject for an audit. Each warning above is something XRayLens measures rather than repeats.", "cool")
    ui.stats_row([("Parameters", "23.5 M", "ResNet-50 + 2-way head"), ("Input", "320 px", "RGB, letterboxed"),
                  ("Threshold", f"{M['model']['threshold']}", "fixed, from config.json")])

ui.section("02", "The data", "Four cohorts, used for different questions. Images stay out of the repository.")
st.markdown("""
| Cohort | Who | Used here for | Licence / terms | In the repo? |
|---|---|---|---|---|
| **Kermany et al. 2018**, test folder (624) | Children aged 1–5, Guangzhou | Showing the leakage gap. **Seen in training.** | CC BY 4.0 | Gallery examples only |
| **RSNA Pneumonia Detection Challenge 2018**, sample of 600 | Adults, NIH ChestX-ray14 | External sensitivity and specificity, robustness | RSNA/Kaggle challenge terms, non-commercial research | **No.** IDs and scores only |
| **Montgomery County CXR set** (138) | Adults, USA, TB programme | Adult specificity, Grad-CAM vs lung masks | NLM public release, citation required | Gallery examples only |
| **Shenzhen CXR set**, sample of 100 | Adults, China, TB programme | Out-of-scope disease probe | NLM public release, citation required | Gallery examples only |

Samples are drawn with a fixed seed (`20261020`), and every source is pinned to a Hugging Face commit in
`scripts/fetch_eval_data.py`. One dataset we **rejected**: a Hugging Face set labelled as African patients turned out to
be Kermany images under a new name (identical `IM-xxxx-xxxx` filenames). Testing on it would have repeated the leakage trap.
""")

ui.section("03", "What XRayLens adds", "Our contribution is the instrument, not the classifier.")
st.markdown("""
1. **Grad-CAM from logits, not softmax.** This model's softmax saturates, so softmax gradients vanish exactly where the model is most confident.
2. **Area-normalised attribution.** Heat on lungs, the frame and padding is always reported against each region's pixel share, so “40% of heat on the lungs” can't hide a 45% area.
3. **Randomisation sanity check** (Adebayo et al. 2018), live for any image.
4. **Perturbation stress lab.** Eight anatomy-preserving edits per image, plus population flip rates.
5. **An evaluation that refuses the easy number.** Leakage made visible, external adult cohorts, Wilson and bootstrap 95% intervals, calibration, and a live threshold explorer.
""")

ui.section("04", "Citations")
st.markdown("""
- Kermany D, Goldbaum M, Cai W, et al. Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell* 2018;172(5):1122–1131. Data: Mendeley Data, doi:10.17632/rscbjbr9sj.2 (CC BY 4.0).
- Jaeger S, Candemir S, Antani S, et al. Two public chest X-ray datasets for computer-aided screening of pulmonary diseases. *Quant Imaging Med Surg* 2014;4(6):475–477.
- Shih G, Wu CC, Halabi SS, et al. Augmenting the National Institutes of Health chest radiograph dataset with expert annotations of possible pneumonia. *Radiol Artif Intell* 2019;1(1):e180041.
- Selvaraju RR, et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. *ICCV* 2017.
- Adebayo J, et al. Sanity checks for saliency maps. *NeurIPS* 2018.
- Zech JR, et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs. *PLOS Medicine* 2018;15(11):e1002683.
""")

ui.html_block('<div class="xl-foot">XRayLens AI is an educational research prototype. It is not a medical device, '
              'and nothing it displays is medical advice.</div>')
