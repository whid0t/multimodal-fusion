import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

# 1) List your log.json files in the order they were produced
log_files = [
    'path/to/log.json'   
]

# 2) Read each into a DataFrame, concatenate, and sort by `iter`
dfs = []
for file in log_files:
    with open(file, 'r') as f:
        records = [json.loads(line) for line in f]
    df = pd.DataFrame(records)
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)
# Remove duplicates in case iterations overlap, keeping the first occurrence
full_df = full_df.drop_duplicates(subset='iter', keep='first')
full_df = full_df.sort_values(by='iter').reset_index(drop=True)

# 3) Filter only the training records
train_df = full_df[full_df['mode'] == 'train'].copy()

# 4) (Optional) smooth with a rolling window
train_df['loss_smooth']   = train_df['loss'].rolling(100, min_periods=1).mean()
train_df['decode_smooth'] = train_df['decode.loss_ce'].rolling(100, min_periods=1).mean()
train_df['aux_smooth']    = train_df['aux.loss_ce'].rolling(100, min_periods=1).mean()

# 5) Plot
plt.figure(figsize=(10, 6))
plt.plot(train_df['iter'], train_df['loss_smooth'],   label='Total Loss (smoothed)')
plt.plot(train_df['iter'], train_df['decode_smooth'], label='Decode Loss (smoothed)')
plt.plot(train_df['iter'], train_df['aux_smooth'],    label='Aux Loss (smoothed)')

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Combined MMSeg Training Loss Curve - 40k iters fine-tune')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 6) Save
output_path = 'path/to/output.png'
plt.savefig(output_path)
print(f"Saved combined training loss curve to {output_path}")
