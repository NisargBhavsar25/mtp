import pandas as pd, numpy as np
s = pd.read_csv('data/raw/Sendek_OP.csv')
s = s[[c for c in s.columns if not c.startswith('Unnamed')]]
print(f'Sendek: N={len(s)}, std(y)={s["log10_target"].std():.4f}, range=[{s["log10_target"].min():.2f}, {s["log10_target"].max():.2f}]')

l = pd.read_csv('data/processed/LLZO_OP_py.csv')
l = l[[c for c in l.columns if not c.startswith('Unnamed')]]
l['log_cond'] = np.log10(l['conductivity'].clip(lower=1e-15))
key = ['compound','temperature']
dup = l.duplicated(subset=key, keep=False)
uniq = l[~dup]; dups = l[dup]
agg = {col: ('median' if l[col].dtype in ['float64','int64'] else 'first') for col in l.columns if col not in key}
merged = dups.groupby(key, as_index=False).agg(agg)
la = pd.concat([uniq, merged], ignore_index=True)
la['log_cond'] = np.log10(la['conductivity'].clip(lower=1e-15))
print(f'LLZO:   N={len(la)}, std(y)={la["log_cond"].std():.4f}, range=[{la["log_cond"].min():.2f}, {la["log_cond"].max():.2f}]')
print(f'  Need std(y)={0.592/np.sqrt(0.2774):.4f} for R2_adj=0.725 with RMSE=0.592')

li = pd.read_csv('data/raw/LiIon_OP.csv')
li['log10_target'] = np.log10(li['target'].clip(lower=1e-15))
print(f'LiIon:  N={len(li)}, std(y)={li["log10_target"].std():.4f}, range=[{li["log10_target"].min():.2f}, {li["log10_target"].max():.2f}]')
