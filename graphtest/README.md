# S&P 500 Scatter Matrix & Feature Analysis

## Amaç

500 S&P 500 şirketinin earnings verisi üzerinde EPS beat prediction için feature analysis:
- Scatter matrix ile feature-feature ilişkileri
- Decile charts ile feature-beat rate ilişkileri
- Point-in-time safe (leak prevention)

## Veri Kaynağı

`../data/raw/*_earnings_with_q4.csv` dosyalarından 498 şirket:
- Her CSV bir şirketin quarterly earnings verileri
- ~35 satır/şirket, toplamda ~17,000+ gözlem
- Target: `eps_beat` (1=beat, 0=miss)

## Kullanım

```bash
cd graphtest
python scatter_matrix_analysis.py
```

## Çıktılar

`reports/` klasöründe:

1. **feature_deciles_all.png** - Tüm 500 şirket için decile charts
   - Her feature 10 dilime bölünür
   - Line plot: beat rate (primary y-axis)
   - Bar plot: observation count (secondary y-axis)
   - Monoton artış/azalış → güçlü sinyal

2. **pairgrid_global.png** - Stratified scatter matrix
   - Dark theme, 12 top feature
   - Orange (miss) vs Blue (beat)
   - Her şirketten max 10 satır (balanced sampling)

3. **feature_signal_summary.csv** - Feature ranking
   - Mutual Information scores
   - Spearman correlation (absolute)
   - Top 12 feature sıralı

## Feature Selection Metodolojisi

1. **Data cleaning:**
   - Sadece sayısal kolonlar
   - %85+ doluluk oranı
   - Leak prevention (rapor sonrası kolonlar çıkarıldı)

2. **Signal detection:**
   - Mutual Information: Non-linear relationships
   - Spearman correlation: Monotonic relationships
   - Combined ranking: (MI_rank + Spearman_rank)

3. **Stratified sampling:**
   - Overweight prevention: her symbol'den max 10 satır
   - Random seed=42 (reproducibility)

## Leak Prevention

Rapor sonrası kolonlar çıkarıldı:
- `actual_eps`, `eps_delta`
- `elo_after`, `elo_change`
- `price_change_1m_pct`, `price_change_3m_pct`

Kullanılabilir: `elo_before`, `eps_estimate_*`, growth metrics (lag1)

