import pandas as pd

url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv"
df = pd.read_csv(url)

caracteristicas = [
    'koi_period', 'koi_depth', 'koi_duration', 'koi_prad',
    'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad'
]

print(df[caracteristicas].describe())
print(df[['koi_prad', 'koi_model_snr']].describe())