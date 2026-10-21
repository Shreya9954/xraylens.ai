# XRayLens AI: demo video script (≈ 2:45)

Record from a local run (`streamlit run app.py`) at 1440 px wide, browser zoom 90%. Warm the app up first (load
every gallery case once) so nothing spins on camera. Times are targets.

| Time | Screen | Say |
|---|---|---|
| **0:00–0:25** · Problem | Reliability page hero: struck-through **1.000 AUC** next to **38.5% specificity** | "This is one pneumonia model and two honest-looking numbers. The perfect one comes from testing on images the model was trained on. The other comes from adults it has never seen. Medical AI is usually sold with the first number. I built a lab for finding the second." |
| **0:25–0:50** · Introduce | Examine page masthead, slow scroll over the contact sheet | "XRayLens wraps a real, public ResNet-50 classifier in a four-step investigation: choose, read, explain, stress-test. Every case on this contact sheet was picked by a rule from my own evaluation of 1,462 images." |
| **0:50–1:05** · Read | Load **Textbook pneumonia**. Point at the readout and the RSNA context line | "It says pneumonia, with a score of 1.0000. But that's a model output, not a probability. The app tells you how often healthy adults score this high. And this image was in the training set." |
| **1:05–1:40** · Explain | Load **Healthy adult, flagged**. Step 03: point at the heat on the "PA ERECT" marker, the teal lung outline and the 12% inside-lungs stat. Open the sanity check and click *Run randomisation test* | "Here's a healthy adult called pneumonia at 0.997. Grad-CAM shows where it looked: the text marker in the corner, not the lungs. Only 12% of the heat is inside the radiologist's lung outline. And the map isn't an artefact. When I destroy the trained weights, it decorrelates, so this really is what the model used." |
| **1:40–1:55** · Stress | Step 04: point at *Tighter collimation* crossing the threshold | "Crop the frame away and the verdict flips to normal. Across 838 external images, adding a text marker flipped 91 labels, every one toward pneumonia." |
| **1:55–2:20** · Evaluate | Reliability page: drag the threshold slider from 0.5 to the 80%-specificity point. Hover the strip chart | "On the dashboard every number is recomputed live. The model ranks well, with an AUC of 0.91, but the shipped threshold is wrong for adults. To reach 80% specificity you need a threshold of 0.99999. And it's badly calibrated: 76% of its outputs are pinned at 0 or 1." |
| **2:20–2:35** · Limits | Scroll to "What this evaluation cannot tell you", then the Provenance page | "I'm explicit about what I can't claim: no clean pediatric test exists for these weights, and my external tests confound age with generalisation." |
| **2:35–2:45** · Close | Back to the masthead | "XRayLens doesn't make a model better. It makes the model accountable. Next steps: lung-mask-guided Grad-CAM metrics for every cohort, a pediatric external set, and a model comparison mode." |

**Checks before recording**
- The "Healthy adult, flagged" crop flip is deterministic (0.997 → 0.104). Re-verify after any model or code change.
- Keep the research-only stamp visible in the first frame.
