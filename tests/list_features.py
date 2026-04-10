import joblib
meta = joblib.load('models/model_metadata.joblib')
cols = meta['feature_columns']['log_Ionic_Conductivity']
orig = [c for c in cols if c.startswith('orig_') and c != 'orig_Temp_K']
print(f'Physical features ({len(orig)}):')
for c in orig:
    print(f'  {c}')
