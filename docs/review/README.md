# Paper review (CSE 754 assignment, 10 marks)

Review of **Karras et al., *Alias-Free Generative Adversarial Networks*
(StyleGAN3), NeurIPS 2021** — A\* venue, GAN paper.

| Section | Marks |
|---|---:|
| Problem statement | 1 |
| Methodology and analysis | 2 |
| Results and outcomes | 2 |
| Limitations and research gaps | 2 |
| Concluding remarks — suggested improvements | 3 |

The mark allocation is printed beside each section heading in the PDF so the
mapping to the rubric is visible at a glance.

## Building

```bash
cd docs/review
tectonic -X compile main.tex          # -> main.pdf  (10 pages)
pandoc main.tex -o main.docx --bibliography=references.bib --citeproc
```

Or produce an Overleaf bundle from the repository root:

```bash
./scripts/make_overleaf_zip.sh review   # -> umud-review-overleaf.zip
```

## Notes on the content

The heaviest-weighted section (3 marks) proposes six concrete improvements,
each tied to specific evidence in the paper rather than offered generically:

1. Make the discriminator equivariant too, and test the authors' own untested
   hypothesis about the FFHQ teeth artefact.
2. Replace the paper's self-proposed equivariance metrics with an independent
   downstream measurement (optical flow / FVD), breaking a closed validation loop.
3. Make band-limiting adaptive per layer rather than uniform, exploiting the
   Pareto frontier visible in the paper's own parameter ablation.
4. Target local deformation equivariance, closing the gap between what the paper
   motivates (a nose moving with a head) and what it measures (global rigid motion).
5. **Selective equivariance** via a two-stream generator — the central suggestion.
   Motivated by B-mode ultrasound, where speckle is bound to the transducer rather
   than to tissue, so "texture sticking" is physically correct and StyleGAN3's
   architecture forbids representing it.
6. Buy rotation equivariance only in the low-resolution layers, where the paper's
   own analysis says global pose is decided.
