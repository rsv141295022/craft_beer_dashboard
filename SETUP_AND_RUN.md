# Craft Beer Recipe Pipeline - Setup & Execution Guide

## Project Structure

```
craft_beer_v2/
├── src/                           # All Python scripts
│   ├── get_url.py                # Scrape recipe URLs from HBA website
│   ├── get_all_recipes.py         # Scrape recipe details from URLs
│   ├── extract_recipes.py         # Extract structured data using OpenAI API
│   └── preprocess_recipes.py      # Normalize and prepare data for analysis
│
├── dataset/                       # All data files
│   ├── scraped/                   # Raw scraped HTML data
│   │   ├── all_urls.csv           # List of recipe URLs
│   │   ├── all_recipes_new.csv    # Raw recipe text data
│   │   └── all_recipes_new1.csv   # Additional raw recipe data
│   │
│   ├── recipes_extracted.json     # Full extracted data (JSON)
│   ├── recipes_extracted_flat.csv # Flattened extraction (CSV)
│   ├── recipes_checkpoint.json    # Checkpoint for resuming extraction
│   │
│   ├── recipes_extracted_malts.csv    # Normalized malts table
│   ├── recipes_extracted_hops.csv     # Normalized hops table
│   │
│   └── processed/                 # Analytics-ready tables
│       ├── recipes_normalized.csv     # Main recipe data
│       ├── malts_normalized.csv       # Malt details
│       ├── hops_normalized.csv        # Hop details
│       ├── yeast_normalized.csv       # Yeast data
│       ├── water_normalized.csv       # Water chemistry
│       ├── timeseries.csv             # Time-series ready
│       └── ingredient_frequency.csv   # Ingredient usage %
│
├── .env                           # Environment variables (API keys)
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
└── SETUP_AND_RUN.md               # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd /Users/famepatcharapol/Desktop/Learning/craft_beer_v2
pip install -r requirements.txt
```

**Required packages:**
```
openai
pydantic
python-dotenv
pandas
numpy
tqdm
openpyxl
selenium
```

### 2. Set Environment Variables

Create `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. Verify Data Directory Structure

All scripts automatically create necessary directories. No manual setup needed!

---

## Execution Workflow

### **Stage 1: Data Collection (Optional - Already Done)**

If you need to re-scrape recipes:

#### Step 1a: Get Recipe URLs
```bash
python src/get_url.py
```
**Output:** `dataset/scraped/all_urls.csv`

#### Step 1b: Scrape Recipe Details
```bash
python src/get_all_recipes.py
```
**Input:** `dataset/scraped/all_urls.csv`
**Output:** `dataset/scraped/all_recipes_new.csv`

---

### **Stage 2: Data Extraction (Already Done - 1,395 recipes)**

Extract structured data from raw recipe text using OpenAI API:

```bash
# Test with 5 recipes
python src/extract_recipes.py --sample 5

# Process all recipes
python src/extract_recipes.py --all

# Resume from recipe #100
python src/extract_recipes.py --resume 100

# Enable preprocessing during extraction
python src/extract_recipes.py --sample 5 --preprocess
```

**Outputs:**
- `dataset/recipes_extracted.json` - Full nested JSON
- `dataset/recipes_extracted_flat.csv` - Flattened CSV
- `dataset/recipes_checkpoint.json` - Resume checkpoint

**Options:**
- `--sample N` - Process first N recipes (for testing)
- `--all` - Process entire dataset
- `--resume N` - Resume from index N (after interrupt)
- `--preprocess` - Enable preprocessing (medals, categories, years)
- `--export FILE.json` - Export existing results to CSV/Excel formats

---

### **Stage 3: Data Preprocessing (Ready for Dashboard)**

Normalize and prepare data for visualization:

```bash
python src/preprocess_recipes.py
```

**Inputs:**
- `dataset/recipes_extracted_flat.csv` (from Stage 2)

**Outputs (in `dataset/processed/`):**
1. `recipes_normalized.csv` (1,395 rows)
   - Main recipe data with computed metrics
   - Columns: title, style, style_group, year, og, fg, abv, ibu, srm, medal
   - Plus: base_malt_pct, crystal_pct, bittering_oz_gal, dry_hop_oz_gal, water profiles

2. `malts_normalized.csv` (5,678 rows)
   - Ingredient-level malt details
   - Grist composition percentages
   - Malt type classification

3. `hops_normalized.csv` (4,467 rows)
   - Hop additions with timing
   - oz/gal calculations for trends
   - Biotransformation flags

4. `yeast_normalized.csv` (1,354 rows)
   - Normalized yeast strains
   - Fermentation temperatures
   - Canonical strain names

5. `water_normalized.csv` (1,395 rows)
   - Water chemistry profiles
   - PPM estimates from salt additions
   - Sulfate:Chloride ratios

6. `timeseries.csv` (1,395 rows)
   - Ready for trend analysis
   - All metrics filled (NaN → mean)
   - Year + all brewing parameters

7. `ingredient_frequency.csv` (4,439 rows)
   - Ingredient usage % by style group
   - For word clouds and ingredient prominence

---

## Key Features

### Path Handling
- ✅ All scripts use **relative paths** from project root
- ✅ Works when run from anywhere: `python src/extract_recipes.py`
- ✅ No hardcoded absolute paths

### Data Flow
```
Raw HTML
   ↓
get_url.py & get_all_recipes.py
   ↓
recipes_extracted_flat.csv (raw extraction)
   ↓
preprocess_recipes.py
   ↓
7 Analytics-Ready Tables → Dashboard Visualizations
```

### Checkpointing & Resume
- Extraction saves every 50 recipes
- Resume from any checkpoint: `python src/extract_recipes.py --resume 500`
- No data loss on interruption

### Data Quality
- Automatically detects missing values
- Fills gaps (NaN → mean for numeric fields)
- Normalizes ingredient names across recipes
- Classifies malt types and hop timing

---

## Example Workflows

### Complete Fresh Start
```bash
# 1. Extract 5 test recipes
python src/extract_recipes.py --sample 5

# 2. Preprocess test data
python src/preprocess_recipes.py

# 3. Check outputs
ls -lh dataset/processed/
```

### Process All 1,400 Recipes
```bash
# Start extraction (takes ~10-15 minutes per 100 recipes)
python src/extract_recipes.py --all

# After extraction completes
python src/preprocess_recipes.py

# View results
head -20 dataset/processed/recipes_normalized.csv
```

### Resume Interrupted Processing
```bash
# Check checkpoint
tail dataset/recipes_checkpoint.json

# Resume from index 500
python src/extract_recipes.py --resume 500

# Then preprocess
python src/preprocess_recipes.py
```

---

## Script Parameters

### extract_recipes.py
```bash
--sample N        # Process first N recipes (default: none)
--all             # Process entire dataset
--resume N        # Resume from recipe index N
--export FILE     # Export existing JSON to CSV/Excel
--preprocess      # Enable data preprocessing
```

### preprocess_recipes.py
- No parameters needed
- Auto-detects input files
- Creates `dataset/processed/` directory

---

## Troubleshooting

### "No such file or directory" Error
**Solution:** Ensure you're running from project root:
```bash
cd /Users/famepatcharapol/Desktop/Learning/craft_beer_v2
python src/extract_recipes.py --sample 5
```

### API Rate Limit Errors
**Solution:** Reduce sleep time in script or:
```python
# In extract_recipes.py, line 441
time.sleep(0.1)  # Reduce from 0.5
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Missing .env File
Create `.env` in project root:
```
OPENAI_API_KEY=sk-...your-key-here...
```

---

## Next Steps

1. ✅ Scripts are configured with proper paths
2. ✅ Data is preprocessed and ready
3. **Next:** Build dashboard with processed tables
   - Use `dataset/processed/timeseries.csv` for trend analysis
   - Use `ingredient_frequency.csv` for word clouds
   - Use normalized ingredient tables for detailed breakdowns

---

## Data Statistics (Full Dataset - 1,395 recipes)

| Metric | Value |
|--------|-------|
| Total Recipes | 1,395 |
| Style Groups | 12 |
| Malt Entries | 5,678 |
| Hop Entries | 4,467 |
| Yeast Strains | 891 |
| Gold Medals | 623 (44.7%) |
| Data Quality | 85-95% complete |

---

For questions or issues, check individual script docstrings:
```bash
head -50 src/extract_recipes.py
head -50 src/preprocess_recipes.py
```
