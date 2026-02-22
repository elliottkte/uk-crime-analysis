# London Crime Analysis

An end-to-end data science project analysing crime patterns across London 
using three years of police data (2023–2025), ONS population estimates, 
and the 2019 Index of Multiple Deprivation.

---

## What this project looks at

- How crime rates vary across London LSOAs when normalised by population
- Whether deprivation scores can predict local crime rates
- Geographic clustering of areas by crime profile
- Stop and search patterns including demographic breakdowns

---

## Data sources

- [police.uk Open Data](https://data.police.uk/) — Metropolitan and City 
  of London Police, January 2023 to December 2025
- [ONS LSOA Population Estimates](https://www.ons.gov.uk/) — mid-2022 
  small area population estimates
- [Ministry of Housing, Communities & Local Government](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019) 
  — Index of Multiple Deprivation 2019, LSOA level scores

---

## Approach

Crime counts alone are misleading — an area with more residents will 
naturally have more recorded crimes. All analysis uses crimes per 1,000 
residents to allow fair comparison across areas of different sizes.

A Random Forest model is trained to predict LSOA-level crime rates from 
deprivation indicators including income, employment, health, education, 
and living environment scores. K-means clustering groups LSOAs by their 
crime profile, visualised on an interactive map.

A live data feed via the police.uk API pulls the most recent month's 
crimes for Central London.

---

## Tools used

Python (pandas, numpy, scikit-learn, plotly, streamlit), Jupyter Notebooks

---

## Running the project
```bash
git clone https://github.com/yourusername/uk-crime-analysis.git
cd uk-crime-analysis
pip install -r requirements.txt
```

Download the raw data files listed above and place them in `data/raw/` 
before running the notebooks. Notebooks are numbered and should be run 
in order.

To launch the dashboard:
```bash
streamlit run app.py
```

---

## Known limitations

The IMD data uses 2011 LSOA boundaries while the population estimates 
use 2021 boundaries. A small number of LSOAs were redrawn between those 
years and are excluded from the joined dataset where codes don't align.

---

## About

I'm a data analyst with a background spanning policing, defence, and 
government. This project was partly motivated by that experience. I 
wanted to explore what the publicly available data actually shows about 
crime and deprivation in London, using analytical methods rather 
than headlines.

[LinkedIn](https://www.linkedin.com/in/kate-elliott/)
