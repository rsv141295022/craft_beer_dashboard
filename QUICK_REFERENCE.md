# Quick Reference - Running Scripts

## Quick Start (Recommended)

```bash
# Navigate to project root
cd /Users/famepatcharapol/Desktop/Learning/craft_beer_v2

# Run preprocessing (ready to use!)
python src/preprocess_recipes.py
```

**Done!** Check `dataset/processed/` for 7 analytics-ready tables.

---

## All Available Commands 

### Extract Recipes (optional - already done with 1,395 recipes)
```bash
# Test with 5 recipes
python src/extract_recipes.py --sample 5

# Process all recipes
python src/extract_recipes.py --all

# Resume from index 500
python src/extract_recipes.py --resume 500

# Enable preprocessing during extraction
python src/extract_recipes.py --all --preprocess
```

### Preprocess Data (main use)
```bash
# Process extracted data into 7 analytics tables
python src/preprocess_recipes.py
```

### Scrape Recipes (optional - already done)
```bash
# Get recipe URLs
python src/get_url.py

# Scrape recipe text from URLs
python src/get_all_recipes.py
```

---

## Output Files

### After Extraction
- `dataset/recipes_extracted.json` - Full nested data
- `dataset/recipes_extracted_flat.csv` - Flattened data
- `dataset/recipes_checkpoint.json` - Resume checkpoint

### After Preprocessing ✅
Ready for dashboard:
1. `recipes_normalized.csv` - Main recipe data
2. `malts_normalized.csv` - Malt details
3. `hops_normalized.csv` - Hop details
4. `yeast_normalized.csv` - Yeast data
5. `water_normalized.csv` - Water chemistry
6. `timeseries.csv` - Trend analysis
7. `ingredient_frequency.csv` - Word clouds

All in: `dataset/processed/`

---

## Data Stats

| Metric | Value |
|--------|-------|
| Total Recipes | 1,395 ✅ |
| Styles | 12 |
| Malts | 5,678 |
| Hops | 4,467 |
| Yeast Strains | 891 |
| Gold Medals | 623 (44.7%) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No such file or directory" | Run from: `/craft_beer_v2/` |
| API rate limits | Reduce `time.sleep()` in script |
| Missing packages | `pip install -r requirements.txt` |
| API key error | Check `.env` file has valid `OPENAI_API_KEY` |

---

## Next Steps

1. ✅ Scripts are configured
2. ✅ Data is preprocessed
3. **→ Build dashboard with processed tables**

Use `dataset/processed/timeseries.csv` and `ingredient_frequency.csv` for visualizations!

---
