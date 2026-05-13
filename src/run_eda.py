"""Run the EDA end to end and print every number Task 2 needs."""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA = Path('data')
FIG = Path('figures')
FIG.mkdir(exist_ok=True)
NOTES = Path('notes')
NOTES.mkdir(exist_ok=True)

# Set headless matplotlib so we can save figures without display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

results = {}


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / f'{name}.png', dpi=130, bbox_inches='tight')
    plt.close()


def section(title):
    print('\n' + '=' * 60)
    print(title)
    print('=' * 60)


# Use efficient dtypes to keep memory in check
DTYPES_INT8 = ['prop_brand_bool', 'promotion_flag', 'srch_saturday_night_bool',
               'random_bool', 'click_bool', 'booking_bool',
               'srch_adults_count', 'srch_children_count', 'srch_room_count',
               'prop_starrating', 'position']
DTYPES_INT32 = ['srch_id', 'site_id', 'visitor_location_country_id',
                'prop_country_id', 'prop_id', 'srch_destination_id',
                'srch_length_of_stay', 'srch_booking_window']

dtype_map = {c: 'Int32' for c in DTYPES_INT32}
dtype_map.update({c: 'Int8' for c in DTYPES_INT8})

section('1. Loading data')
print('Loading training set ...', flush=True)
train = pd.read_csv(DATA / 'training_set_VU_DM.csv', dtype=dtype_map)
print('train:', train.shape, 'memory:', round(train.memory_usage(deep=True).sum() / 1e9, 2), 'GB', flush=True)
print('Loading test set ...', flush=True)
test = pd.read_csv(DATA / 'test_set_VU_DM.csv', dtype=dtype_map)
print('test :', test.shape, flush=True)

results['n_train_rows'] = int(len(train))
results['n_test_rows'] = int(len(test))
results['n_train_searches'] = int(train['srch_id'].nunique())
results['n_test_searches'] = int(test['srch_id'].nunique())
results['n_hotels'] = int(train['prop_id'].nunique())
results['train_only_cols'] = sorted(set(train.columns) - set(test.columns))

section('2. Target rates and search-level overview')
results['click_rate'] = float(train['click_bool'].mean())
results['book_rate'] = float(train['booking_bool'].mean())
results['book_given_click'] = float(train.loc[train['click_bool'] == 1, 'booking_bool'].mean())
results['random_share'] = float(train['random_bool'].mean())
print('click_rate         :', results['click_rate'])
print('book_rate          :', results['book_rate'])
print('book given click   :', results['book_given_click'])
print('random sort share  :', results['random_share'])

search_sizes = train.groupby('srch_id').size()
results['search_size_median'] = int(search_sizes.median())
results['search_size_mean'] = float(search_sizes.mean())
results['search_size_min'] = int(search_sizes.min())
results['search_size_max'] = int(search_sizes.max())
print('search size stats:', search_sizes.describe().to_dict())

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(search_sizes.clip(upper=40), bins=range(0, 41), edgecolor='white')
ax.set_xlabel('hotels per search (clipped at 40)')
ax.set_ylabel('number of searches')
ax.set_title('Distribution of search sizes')
savefig('01_search_sizes')

section('3. Missingness')
missing = train.isna().mean().sort_values(ascending=False)
missing = missing[missing > 0]
print(missing.to_string())
results['missing'] = {k: float(v) for k, v in missing.items()}

fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(missing))))
missing.plot.barh(ax=ax, color='#4c72b0')
ax.invert_yaxis()
ax.set_xlabel('fraction missing')
ax.set_title('Missing values per column (train)')
savefig('02_missingness')

section('4. Position bias')
by_pos = train.groupby('position').agg(
    click_rate=('click_bool', 'mean'),
    book_rate=('booking_bool', 'mean'),
    n=('click_bool', 'size'),
).reset_index()
print(by_pos.head(15).to_string(index=False))
results['cr_pos1'] = float(by_pos.loc[by_pos['position'] == 1, 'click_rate'].iloc[0])
results['cr_pos5'] = float(by_pos.loc[by_pos['position'] == 5, 'click_rate'].iloc[0])
results['cr_pos20'] = float(by_pos.loc[by_pos['position'] == 20, 'click_rate'].iloc[0])
results['br_pos1'] = float(by_pos.loc[by_pos['position'] == 1, 'book_rate'].iloc[0])

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(by_pos['position'], by_pos['click_rate'], label='click rate', marker='o')
ax.plot(by_pos['position'], by_pos['book_rate'], label='book rate', marker='s')
ax.set_xlabel('position on results page')
ax.set_ylabel('rate')
ax.set_title('Click and book rate by position')
ax.legend()
savefig('03_position_bias')

by_pos_rand = train.groupby(['random_bool', 'position'])['click_bool'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 4.5))
for r, sub in by_pos_rand.groupby('random_bool'):
    label = 'random sort' if r == 1 else 'normal sort'
    ax.plot(sub['position'], sub['click_bool'], marker='o', label=label)
ax.set_xlabel('position')
ax.set_ylabel('click rate')
ax.set_title('Click rate by position, normal vs random sort')
ax.legend()
savefig('04_position_bias_random')

random_at_1 = float(by_pos_rand[(by_pos_rand['random_bool'] == 1) & (by_pos_rand['position'] == 1)]['click_bool'].iloc[0])
random_at_5 = float(by_pos_rand[(by_pos_rand['random_bool'] == 1) & (by_pos_rand['position'] == 5)]['click_bool'].iloc[0])
results['cr_random_at_1'] = random_at_1
results['cr_random_at_5'] = random_at_5

section('5. Price distribution')
desc = train['price_usd'].describe([0.5, 0.9, 0.95, 0.99, 0.999, 0.9999])
print(desc)
results['price_max'] = float(train['price_usd'].max())
results['price_99'] = float(train['price_usd'].quantile(0.99))
results['price_median'] = float(train['price_usd'].median())
results['price_mean'] = float(train['price_usd'].mean())
results['price_above_5000'] = int((train['price_usd'] > 5000).sum())
results['price_above_100k'] = int((train['price_usd'] > 100000).sum())

low, high = train['price_usd'].quantile([0.01, 0.99])
trimmed = train['price_usd'].clip(low, high)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(np.log1p(train['price_usd'].clip(0, 1e7)), bins=80, edgecolor='white')
axes[0].set_title('log(1 + price_usd), full range')
axes[0].set_xlabel('log(1 + price)')
axes[1].hist(trimmed, bins=80, edgecolor='white')
axes[1].set_title(f'price_usd, clipped to [{low:.0f}, {high:.0f}]')
axes[1].set_xlabel('price (USD)')
savefig('05_price_distribution')

med_by_site = train.groupby('site_id')['price_usd'].median().sort_values()
print('\nMedian price per site_id:')
print(med_by_site.to_string())
results['price_median_by_site'] = {int(k): float(v) for k, v in med_by_site.items()}
fig, ax = plt.subplots(figsize=(8, 4))
med_by_site.plot.bar(ax=ax, color='#4c72b0')
ax.set_xlabel('site_id')
ax.set_ylabel('median price_usd')
ax.set_title('Median displayed price by site_id')
savefig('06_price_by_site')

section('6. Star and review effects')
star_clicks = train.groupby('prop_starrating')['click_bool'].mean()
review_clicks = train.groupby('prop_review_score')['click_bool'].mean()
print('click rate by star rating:')
print(star_clicks.to_string())
print('\nclick rate by review score:')
print(review_clicks.to_string())
results['cr_by_star'] = {int(k): float(v) for k, v in star_clicks.items()}
results['cr_by_review'] = {float(k): float(v) for k, v in review_clicks.items() if pd.notna(k)}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
star_clicks.plot.bar(ax=axes[0], color='#4c72b0')
axes[0].set_title('Click rate by star rating')
axes[0].set_xlabel('prop_starrating')
axes[0].set_ylabel('click rate')
review_clicks.plot.bar(ax=axes[1], color='#dd8452')
axes[1].set_title('Click rate by review score')
axes[1].set_xlabel('prop_review_score')
savefig('07_star_review_clickrate')

brand_promo = train.groupby(['prop_brand_bool', 'promotion_flag'])[['click_bool', 'booking_bool']].mean()
print('\nclick/book by brand and promotion:')
print(brand_promo)
results['brand_promo'] = brand_promo.reset_index().to_dict(orient='records')

section('7. Within-search price decile')
tmp = train[['srch_id', 'price_usd', 'click_bool', 'booking_bool']].copy()
tmp['price_q'] = tmp.groupby('srch_id')['price_usd'].transform(
    lambda s: pd.qcut(s.rank(method='first'), q=min(10, max(1, s.nunique())), labels=False, duplicates='drop')
)
rel = tmp.groupby('price_q')[['click_bool', 'booking_bool']].mean()
print(rel)
results['within_search_price'] = rel.reset_index().to_dict(orient='records')
results['within_cheapest_click'] = float(rel.loc[0, 'click_bool'])
results['within_expensive_click'] = float(rel.iloc[-1]['click_bool'])

fig, ax = plt.subplots(figsize=(7, 4))
rel.plot.bar(ax=ax)
ax.set_xlabel('price decile within search (0 = cheapest)')
ax.set_ylabel('rate')
ax.set_title('Click and book rate by within-search price decile')
savefig('08_within_search_price')

section('8. Search context')
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, col, label in zip(
    axes.ravel(),
    ['srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count'],
    ['length of stay (nights)', 'booking window (days)', 'adults', 'children'],
):
    upper = train[col].quantile(0.99)
    ax.hist(train[col].clip(upper=upper), bins=30, edgecolor='white')
    ax.set_title(label)
savefig('09_search_context')

results['stay_median'] = int(train['srch_length_of_stay'].median())
results['stay_mean'] = float(train['srch_length_of_stay'].mean())
results['booking_window_median'] = int(train['srch_booking_window'].median())
results['booking_window_mean'] = float(train['srch_booking_window'].mean())
results['adults_mode'] = int(train['srch_adults_count'].mode().iloc[0])
results['children_zero_share'] = float((train['srch_children_count'] == 0).mean())

section('9. Correlation table')
num_cols = [
    'prop_starrating', 'prop_review_score', 'prop_brand_bool',
    'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price',
    'price_usd', 'promotion_flag',
    'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
    'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
    'click_bool', 'booking_bool',
]
num_cols = [c for c in num_cols if c in train.columns]
corr = train[num_cols].corr(numeric_only=True)
print(corr[['click_bool', 'booking_bool']].sort_values('booking_bool', ascending=False).to_string())
results['corr_with_book'] = corr['booking_bool'].drop(['click_bool', 'booking_bool']).to_dict()
results['corr_with_book'] = {k: float(v) for k, v in results['corr_with_book'].items() if pd.notna(v)}

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=True, fmt='.2f', annot_kws={'size': 7}, ax=ax)
ax.set_title('Correlation among numeric features')
savefig('10_correlation')

# Save aggregated results JSON
with open(NOTES / 'eda_numbers.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print('\n\nDONE. Numbers saved to notes/eda_numbers.json. Figures in figures/.')
