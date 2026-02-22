# London Crime & the Cost of Living
### An interactive data dashboard | 2023–2025

An end-to-end data analysis project examining how economic pressures, deprivation, and policing patterns have shaped crime across London since 2023. Built in Python and Streamlit, the dashboard synthesises 3.4 million police records with ONS inflation data, IMD deprivation indices, and stop and search records.

---

## Dashboard sections

**The Story** — Overview of London crime trends 2023–2025 with headline statistics and force-level breakdowns.

**Economic Crime** — Correlation analysis between food CPI inflation and crime types, seasonal decomposition of shoplifting, and a drugs offence changepoint analysis. ONS Series D7G8 is fetched automatically on first run.

**Crime & Deprivation** — Borough-level correlations between IMD 2019 deprivation domains and crime rates. Boroughs classified by quadrant (deprived/affluent × high/expected crime).

**Policing Response** — Stop and search outcomes by ethnicity, object of search, and borough. Arrest rates compared against population share using ONS Census 2021 figures.

**Where is London Headed?** — Borough vulnerability index combining deprivation, shoplifting trend, crime-deprivation mismatch, and policing intensity. Crime trajectory and three-scenario shoplifting projections for 2026.

---

## Data sources

| Dataset | Source | File |
|---|---|---|
| Police recorded crime (street) | [police.uk](https://data.police.uk/data/) | `data/raw/**/*street*.csv` |
| Stop and search | [police.uk](https://data.police.uk/data/) | `data/raw/**/*stop*search*.csv` |
| IMD 2019 | [MHCLG](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019) | `data/raw/2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv` |
| LSOA population estimates | [ONS](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates) | `data/raw/sapelsoasyoa20222024.xlsx` |
| CPI food inflation (D7G8) | [ONS](https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7g8/mm23) | Fetched automatically on first run |

Data covers Metropolitan Police and City of London Police, January 2023 – December 2025.

---

## Project structure

```
uk-crime-analysis/
├── app.py                          # Streamlit entry point
├── run_all.py                      # Pipeline orchestrator
├── requirements.txt
│
├── processing/
│   ├── 01_clean_street_data.py     # Load + clean police.uk street CSVs
│   ├── 02_economic_analysis.py     # Food inflation correlations, decomposition
│   ├── 03_deprivation_correlations.py  # IMD joins, borough outlier classification
│   ├── 04_train_model.py           # Random Forest crime rate prediction
│   ├── 05_vulnerability_index.py   # Borough vulnerability index + projections
│   └── 06_stop_search.py           # Stop and search processing
│
├── sections/
│   ├── the_story.py
│   ├── economic_crime.py
│   ├── crime_deprivation.py
│   ├── policing_response.py
│   └── where_headed.py
│
├── utils/
│   └── data_loaders.py             # Cached data loading functions
│
├── data/
│   ├── raw/                        # Source files (not tracked in git)
│   └── processed/                  # Pipeline outputs (not tracked in git)
│
└── models/
    └── crime_rate_model.pkl        # Trained RF model (not tracked in git)
```

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/yourusername/uk-crime-analysis.git
cd uk-crime-analysis
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

**2. Download raw data**

Download Metropolitan Police and City of London Police data (Jan 2023 – Dec 2025) from [data.police.uk](https://data.police.uk/data/) and place the CSV files anywhere inside `data/raw/`. The pipeline uses recursive glob patterns so subfolder depth does not matter.

Download the IMD 2019 scores file and the ONS LSOA population estimates and place them in `data/raw/` with the filenames shown in the table above. The food inflation data is fetched automatically.

**3. Run the pipeline**

```bash
python run_all.py
```

This runs all six processing scripts in order. If a script fails, the pipeline stops and tells you how to resume from that point:

```bash
python run_all.py --from 03     # resume from script 03
python run_all.py --only 02 05  # rerun specific scripts
```

**4. Launch the dashboard**

```bash
streamlit run app.py
```

---

## Methods

**Crime rate normalisation** — Borough-level crime rates are calculated per 1,000 residents using ONS mid-2022 LSOA population estimates aggregated to borough level.

**Deprivation classification** — Boroughs are classified into quadrants using London median crime rate and London median IMD score as thresholds. A residual approach (as used nationally) was not applied because London's high-footfall wealthy boroughs (Westminster, City of London) distort the deprivation-crime regression line.

**Stop and search disparity** — Stop rates by ethnicity are compared against ONS Census 2021 London population shares. Arrest rates are shown separately to avoid conflating enforcement intensity with outcome effectiveness.

**Vulnerability index** — Composite score (0–100) weighted as: deprivation 35%, shoplifting trend 30%, crime-deprivation mismatch 20%, policing intensity 15%. Each component is min-max normalised before weighting.

**Random Forest model** — Trained on 10,247 London LSOAs with seven IMD domain scores as features, predicting log crime rate per 1,000 residents. R²=0.695 on held-out test set.

---

## Notes on data quality

- Stop and search records missing GPS coordinates (~6%) are excluded from borough-level analysis but included in aggregate ethnicity statistics.
- The drugs offence spike in 2024–2025 is partly attributable to changed recording practices (Operation Yamata) rather than a real increase in drug use.
- IMD 2019 is the most recent available index; deprivation changes since 2019 are not captured.
- Crime data covers recorded offences only. Dark figure and reporting rate variation across boroughs and crime types are not accounted for.