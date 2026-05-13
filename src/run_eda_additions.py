"""Additional EDA: date_time, train vs test sanity, summary stats, NDCG baselines."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

DATA = Path('data')
FIG = Path('figures')
NOTES = Path('notes')
extras = {}

print('Loading...', flush=True)
USE = ['srch_id', 'date_time', 'site_id', 'prop_id', 'prop_starrating',
       'prop_review_score', 'price_usd', 'srch_length_of_stay',
       'srch_adults_count', 'click_bool', 'booking_bool', 'position', 'random_bool']
train = pd.read_csv(DATA / 'training_set_VU_DM.csv', usecols=USE)
test_use = [c for c in USE if c not in {'click_bool', 'booking_bool', 'position'}]
test = pd.read_csv(DATA / 'test_set_VU_DM.csv', usecols=test_use)

# === 1. date_time analysis ===
print('\n=== date_time ===', flush=True)
train['date_time'] = pd.to_datetime(train['date_time'])
test['date_time'] = pd.to_datetime(test['date_time'])

extras['train_date_min'] = str(train['date_time'].min())
extras['train_date_max'] = str(train['date_time'].max())
extras['test_date_min'] = str(test['date_time'].min())
extras['test_date_max'] = str(test['date_time'].max())
print(f"train range: {extras['train_date_min']} to {extras['train_date_max']}")
print(f"test  range: {extras['test_date_min']} to {extras['test_date_max']}")

train['month'] = train['date_time'].dt.month
train['weekday'] = train['date_time'].dt.weekday  # 0 = Monday
train['hour'] = train['date_time'].dt.hour

cr_by_month = train.groupby('month')['click_bool'].mean()
br_by_month = train.groupby('month')['booking_bool'].mean()
cr_by_weekday = train.groupby('weekday')['click_bool'].mean()
br_by_weekday = train.groupby('weekday')['booking_bool'].mean()
cr_by_hour = train.groupby('hour')['click_bool'].mean()

print('click rate by month:'); print(cr_by_month.round(4).to_string())
print('click rate by weekday (0=Mon):'); print(cr_by_weekday.round(4).to_string())
print('click rate by hour:'); print(cr_by_hour.round(4).to_string())

extras['cr_by_month'] = {int(k): float(v) for k, v in cr_by_month.items()}
extras['cr_by_weekday'] = {int(k): float(v) for k, v in cr_by_weekday.items()}
extras['cr_by_hour'] = {int(k): float(v) for k, v in cr_by_hour.items()}
extras['highest_month_click'] = int(cr_by_month.idxmax())
extras['lowest_month_click'] = int(cr_by_month.idxmin())

# Plot: click rate by month and weekday side by side
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
cr_by_month.plot(ax=axes[0], marker='o', color='#4c72b0')
axes[0].set_title('Click rate by month')
axes[0].set_xlabel('month')
axes[0].set_ylabel('click rate')
axes[0].set_xticks(range(1, 13))

weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
cr_by_weekday.index = weekday_names
cr_by_weekday.plot.bar(ax=axes[1], color='#dd8452')
axes[1].set_title('Click rate by weekday')
axes[1].set_xlabel('day of week')
axes[1].set_ylabel('click rate')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(FIG / '11_temporal.png', dpi=130, bbox_inches='tight')
plt.close()

# === 2. train vs test sanity check ===
print('\n=== train vs test sanity ===', flush=True)
def quick_dist(s):
    return {
        'mean': float(s.mean()),
        'median': float(s.median()),
        'p99': float(s.quantile(0.99)),
        'missing_pct': float(s.isna().mean()),
    }

cmp_cols = ['price_usd', 'prop_starrating', 'prop_review_score', 'srch_length_of_stay', 'srch_adults_count']
sanity = {}
for c in cmp_cols:
    sanity[c] = {'train': quick_dist(train[c]), 'test': quick_dist(test[c])}
    print(f'\n{c}:'); print('  train:', sanity[c]['train']); print('  test :', sanity[c]['test'])

extras['train_vs_test'] = sanity

# Site_id share check
train_sites = set(train['site_id'].unique())
test_sites = set(test['site_id'].unique())
extras['sites_train_only'] = sorted(int(s) for s in (train_sites - test_sites))
extras['sites_test_only'] = sorted(int(s) for s in (test_sites - train_sites))
extras['sites_shared'] = len(train_sites & test_sites)

# Search-id overlap (should be zero)
overlap_srch = len(set(train['srch_id'].unique()) & set(test['srch_id'].unique()))
extras['srch_id_overlap'] = int(overlap_srch)
print(f'\nsearch_id overlap between train and test: {overlap_srch}')
print(f'sites in train only: {extras["sites_train_only"]}')
print(f'sites in test only : {extras["sites_test_only"]}')

# Search size comparison
train_sz = train.groupby('srch_id').size()
test_sz = test.groupby('srch_id').size()
extras['search_size_train_median'] = int(train_sz.median())
extras['search_size_test_median'] = int(test_sz.median())
extras['search_size_train_mean'] = float(train_sz.mean())
extras['search_size_test_mean'] = float(test_sz.mean())

# === 3. NDCG baselines ===
print('\n=== NDCG baselines ===', flush=True)

def dcg_at_k(rels, k=5):
    rels = np.asarray(rels)[:k]
    if len(rels) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(rels) + 2))
    gains = (2 ** rels - 1)
    return float(np.sum(gains * discounts))

def ndcg_at_k(rels, k=5):
    actual = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0

# Build relevance: 5 if booked, 1 if clicked, 0 otherwise
train['relevance'] = np.where(train['booking_bool'] == 1, 5,
                              np.where(train['click_bool'] == 1, 1, 0))

# Sample searches for speed (200k searches is too many to evaluate per-query)
rng = np.random.default_rng(42)
all_sids = train['srch_id'].unique()
sample_sids = rng.choice(all_sids, size=20000, replace=False)
sample = train[train['srch_id'].isin(sample_sids)].copy()

# Random baseline: shuffle within each search
def random_ndcg(group, seed=0):
    rels = group['relevance'].sample(frac=1, random_state=seed).values
    return ndcg_at_k(rels, k=5)

random_scores = sample.groupby('srch_id', group_keys=False).apply(lambda g: ndcg_at_k(
    g['relevance'].sample(frac=1, random_state=int(g['srch_id'].iloc[0]) % 2**31).values, k=5))

# Position baseline: how the data was actually shown (use position order)
def position_ndcg(group):
    g = group.sort_values('position')
    return ndcg_at_k(g['relevance'].values, k=5)

position_scores = sample.groupby('srch_id', group_keys=False).apply(position_ndcg)

# Popularity baseline: per-property mean booking rate from train, sort by it
prop_pop = train.groupby('prop_id')['booking_bool'].mean()
sample['pop_score'] = sample['prop_id'].map(prop_pop)
def pop_ndcg(group):
    g = group.sort_values('pop_score', ascending=False)
    return ndcg_at_k(g['relevance'].values, k=5)
pop_scores = sample.groupby('srch_id', group_keys=False).apply(pop_ndcg)

# Price-only baseline: rank by price ascending
def price_ndcg(group):
    g = group.sort_values('price_usd', ascending=True)
    return ndcg_at_k(g['relevance'].values, k=5)
price_scores = sample.groupby('srch_id', group_keys=False).apply(price_ndcg)

extras['ndcg_random'] = float(random_scores.mean())
extras['ndcg_position'] = float(position_scores.mean())
extras['ndcg_popularity_book'] = float(pop_scores.mean())
extras['ndcg_price_asc'] = float(price_scores.mean())
print(f'NDCG@5 random         : {extras["ndcg_random"]:.4f}')
print(f'NDCG@5 keep position  : {extras["ndcg_position"]:.4f}')
print(f'NDCG@5 popularity sort: {extras["ndcg_popularity_book"]:.4f}')
print(f'NDCG@5 price ascending: {extras["ndcg_price_asc"]:.4f}')

# === 4. Summary stats table for top features ===
print('\n=== summary stats table ===', flush=True)
TOP = ['price_usd', 'prop_starrating', 'prop_review_score',
       'prop_location_score2', 'srch_length_of_stay', 'srch_booking_window']
table_rows = []
for c in TOP:
    if c not in train.columns:
        continue
    s = train[c]
    table_rows.append({
        'feature': c,
        'min': float(s.min()),
        'median': float(s.median()),
        'p99': float(s.quantile(0.99)),
        'max': float(s.max()),
        'missing_pct': float(s.isna().mean()),
    })
extras['summary_table'] = table_rows
for r in table_rows:
    print(r)

# Save
with open(NOTES / 'eda_extras.json', 'w') as f:
    json.dump(extras, f, indent=2, default=str)

print('\nDONE. Extras saved to notes/eda_extras.json')
