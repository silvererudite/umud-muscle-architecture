# umud-src

Source package for the CSE 754 course project on the UMUD Challenge
(geometry-aware muscle-architecture estimation in B-mode ultrasound).

`umud/` here is a **staging copy** of `src/umud/` and is not tracked in git —
`scripts/run_pipeline.sh` populates it immediately before publishing, so the
repository has a single source of truth. To refresh it by hand:

```bash
cp src/umud/*.py kaggle/src/umud/
cp src/umud/assets/*.json kaggle/src/umud/assets/
kaggle datasets version -p kaggle/src -m "update" -r zip
```

Kernels mount it and import from it:

```python
import sys; sys.path.append('/kaggle/input/umud-src')
from umud.train import train
```
