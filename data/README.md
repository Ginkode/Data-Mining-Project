# Dataset setup

This project uses the **OASIS-2 Longitudinal MRI Data in Nondemented and Demented Older Adults** dataset.

The dataset is intentionally **not redistributed in this repository**. OASIS requires users to accept its Data Use Agreement and to acknowledge the dataset appropriately when publicly presenting results derived from it.

## Setup

1. Visit the official OASIS website: https://www.oasis-brains.org/
2. Obtain access to **OASIS-2** and accept the applicable Data Use Agreement.
3. Download the demographic/longitudinal CSV.
4. Save it locally as:

```text
data/oasis_longitudinal.csv
```

5. From the repository root, run:

```bash
python src/model_comparison.py
```

## Citation

Marcus, D. S., Fotenos, A. F., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2010). Open Access Series of Imaging Studies: Longitudinal MRI Data in Nondemented and Demented Older Adults. *Journal of Cognitive Neuroscience, 22*(12), 2677–2684. https://doi.org/10.1162/jocn.2009.21407

When using OASIS data, also follow the current acknowledgement requirements published by OASIS.
