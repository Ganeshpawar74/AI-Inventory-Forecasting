from scripts.simulate_reorders import load_models, compute_residual_stats, calibrate_and_simulate, DATA
import pandas as pd

print('Loading models...')
models, global_model = load_models()
print('Models loaded:', len(models), 'global:', global_model is not None)
print('Reading sample data...')
df = pd.read_csv(DATA, parse_dates=['date'])
print('Rows:', len(df))
df_test = df.dropna(subset=['future_14d_sum'])
print('Test rows:', len(df_test))
print('Computing residual stats...')
sku_stats = compute_residual_stats(df_test, models, global_model)
print('SKU stats rows:', len(sku_stats))
print('Calibrating and simulating...')
df_out = calibrate_and_simulate(sku_stats)
print('Out rows:', len(df_out))
print(df_out.head().to_string(index=False))
# write outputs
out_path = 'reports/reorder_suggestions_90pct.csv'
df_out[df_out['service_level']==0.9].to_csv(out_path, index=False)
print('Wrote', out_path)
