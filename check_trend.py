import pandas as pd

decomp = pd.read_csv('data/processed/shoplifting_decomposition.csv')
decomp['month'] = pd.to_datetime(decomp['month'])
decomp_clean = decomp.dropna(subset=['trend'])

trend_min = decomp_clean['trend'].min()
trend_max = decomp_clean['trend'].max()
trend_start = decomp_clean.iloc[0]['trend']
trend_end = decomp_clean.iloc[-1]['trend']

min_month = decomp_clean.loc[decomp_clean['trend'].idxmin(), 'month'].strftime('%b %Y')
max_month = decomp_clean.loc[decomp_clean['trend'].idxmax(), 'month'].strftime('%b %Y')
start_month = decomp_clean.iloc[0]['month'].strftime('%b %Y')
end_month = decomp_clean.iloc[-1]['month'].strftime('%b %Y')

print(f'Trend start: {trend_start:.1f} ({start_month})')
print(f'Trend end:   {trend_end:.1f} ({end_month})')
print(f'Trend min:   {trend_min:.1f} ({min_month})')
print(f'Trend max:   {trend_max:.1f} ({max_month})')
print()
print(f'Min to max (current dashboard):  +{(trend_max - trend_min) / trend_min * 100:.1f}%')
print(f'Start to end (more honest):      +{(trend_end - trend_start) / trend_start * 100:.1f}%')
print(f'Raw annual 2023 to 2025:         +{(88134 - 57426) / 57426 * 100:.1f}%')