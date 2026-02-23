# London Crime Analysis Dashboard

An end-to-end data pipeline and interactive dashboard analysing 3.4 million Metropolitan Police crime records from 2023–2025. Built to explore the structural drivers of London's crime geography, with a focus on the shoplifting surge, deprivation patterns, and stop and search effectiveness.

**[Live Dashboard →](https://uk-crime-analysis.streamlit.app/)**

---

## Key Findings

**Housing environment predicts crime geography more strongly than income.** A Random Forest model trained on IMD 2025 deprivation data found that Living Environment and Barriers to Housing & Services together account for roughly half the model's predictive power — outperforming Income and Employment deprivation, which are the factors most commonly discussed in relation to crime. This points toward housing quality and service access as more tractable policy levers than income transfers alone.

**London's shoplifting surge is sustained, not cyclical.** Shoplifting rose substantially between 2023 and 2025 across nearly all boroughs. STL decomposition separates a clear upward trend from seasonal variation. The relationship with food inflation is negative at lag-0 (the two series diverged as inflation fell while shoplifting kept rising), consistent with accumulated financial damage rather than a concurrent price-driven effect.

**Stop and search activity is heavily concentrated and shows significant ethnic disparity.** Black Londoners are stopped at a rate roughly 4× their share of the population. Arrest rates on drug searches — the most common search type — are below 10%, raising questions about the targeting accuracy of the tool. A hypothesis table tests three competing explanations for the August 2024 drugs offence spike against observable data implications.

**Borough vulnerability is driven by a combination of deprivation, crime trend, and policing intensity.** A composite vulnerability index combining four components (weighted 35/30/20/15) identifies the boroughs under most sustained structural pressure. Risk tiers are assigned by tertile rather than fixed thresholds, ensuring meaningful differentiation across all 33 boroughs.

---

## Dashboard Sections

| Section | What it covers |
|---|---|
| **Economic Crime** | Shoplifting vs food inflation, lag correlation with bootstrapped CIs, STL decomposition, drugs changepoint analysis, borough trend map |
| **Deprivation & Crime** | IMD 2025 domain correlations, borough outlier classification, crime-deprivation mismatch |
| **Policing Response** | Stop and search ethnicity disparity, effectiveness by search type, drugs hypothesis table, borough map |
| **Where is London Headed?** | Random Forest feature importance, borough vulnerability index, weight sensitivity analysis, 12-month shoplifting scenarios |

---

## Technical Overview

### Data sources
- **Metropolitan Police street crime data** — police.uk, 2023–2025, ~3.4M records
- **Stop and search data** — police.uk, 2023–2025, ~400k records
- **IMD 2025** — MHCLG English Indices of Deprivation (released 2025), replacing 2019 vintage
- **ONS CPI food inflation** — monthly series D7G8, fetched from ONS API
- **ONS population estimates** — Mid-2022 LSOA-level, sapelsoasyoa20222024.xlsx

### Pipeline

```
01_clean_street_data.py       Force extraction, deduplication, LSOA validation
02_economic_analysis.py       Lag correlations (bootstrapped), STL, changepoint detection
03_deprivation_correlations.py  IMD domain correlations, borough outlier classification
04_train_model.py             Random Forest with spatial block cross-validation
05_vulnerability_index.py     Composite vulnerability index, weight sensitivity analysis
06_process_stop_search.py     BallTree borough assignment, ethnicity disparity, hypothesis table
```

**To run the full pipeline:**
```bash
python run_all.py
```

**To run from a specific script:**
```bash
python run_all.py --from 3
```

### Modelling decisions worth noting

**Spatial cross-validation.** The model uses grid-based spatial blocking rather than random train/test splits. LSOAs are assigned to ~0.05° grid cells (~3.5km at London's latitude), and each fold holds out a geographically contiguous region. This prevents spatially autocorrelated LSOAs from leaking between train and test sets, producing a more honest R² estimate.

**Rank-based normalisation.** The vulnerability index uses rank normalisation rather than min-max scaling. City of London is an extreme outlier on crime rate — min-max scaling compresses all other boroughs toward zero and destroys variation in the composite score. Rank normalisation is insensitive to the magnitude of outliers while preserving ordinal relationships.

**Tertile tier assignment.** Vulnerability risk tiers (Higher / Medium / Lower) are assigned by tertile cut-points computed from the actual score distribution, not fixed thresholds. Fixed thresholds produced a 26/7/0 split on the first pipeline run — near-useless for differentiation. Tertile assignment guarantees ~11 boroughs per tier.

**LSOA boundary bridging.** IMD 2025 uses 2021 LSOA boundaries; police.uk crime data uses 2011 codes. The ONS 2011→2021 correspondence table is applied at load time so that ~10% of LSOAs whose boundaries changed between revisions still join correctly.

### Testing

86 tests across schema validation, analytical correctness, and pipeline integrity:

```bash
pytest tests/test_pipeline.py -v
```

Test classes cover: street data cleaning, economic outputs, deprivation outputs, model outputs, stop and search outputs, and analytical correctness (changepoint date, correlation direction, feature importance plausibility, vulnerability rank ordering).

---

## Project Structure

```
├── processing/
│   ├── 01_clean_street_data.py
│   ├── 02_economic_analysis.py
│   ├── 03_deprivation_correlations.py
│   ├── 04_train_model.py
│   ├── 05_vulnerability_index.py
│   └── 06_process_stop_search.py
├── sections/
│   ├── economic_crime.py
│   ├── deprivation.py
│   ├── policing_response.py
│   └── where_headed.py
├── utils/
│   ├── data_loaders.py
│   ├── constants.py
│   ├── charts.py
│   ├── helpers.py
│   └── lsoa_lookup.py
├── tests/
│   └── test_pipeline.py
├── data/
│   ├── raw/          ← source files (not tracked)
│   └── processed/    ← pipeline outputs (not tracked)
├── models/           ← trained model (not tracked)
├── app.py
├── run_all.py
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/yourusername/uk-crime-analysis
cd uk-crime-analysis
pip install -r requirements.txt
```

**Data files required in `data/raw/`:**
- Police.uk street crime CSVs (2023–2025)
- Police.uk stop and search CSVs (2023–2025)
- `imd_2025.csv` — [MHCLG English Indices of Deprivation 2025](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025)
- `lsoa_2011_to_2021_lookup.csv` — [ONS LSOA lookup](https://geoportal.statistics.gov.uk)
- `sapelsoasyoa20222024.xlsx` — [ONS population estimates](https://www.ons.gov.uk)
- `ons_cpi_food_d7g8.csv` — fetched automatically from ONS API on first run

```bash
python run_all.py     # build all processed data
streamlit run app.py  # launch dashboard
```

---

## Stack

Python · Pandas · scikit-learn · statsmodels · SciPy · Plotly · Streamlit · pytest