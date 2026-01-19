# Craft Beer Recipe Extraction

Extract structured data from craft beer recipes using OpenAI GPT-4o.

## Setup

1. **Install dependencies:**
```bash
pip install pandas numpy openai pydantic python-dotenv tqdm openpyxl
```

2. **Configure OpenAI API key:**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

### Python Script

```bash
# Test on 5 sample recipes
python extract_recipes.py --sample 5

# Process all recipes (~1395 recipes, will take time and cost money)
python extract_recipes.py --all

# Resume from a specific index (if interrupted)
python extract_recipes.py --resume 100

# Export existing JSON results to other formats
python extract_recipes.py --export dataset/recipes_extracted.json
```

### Jupyter Notebook

Open `notebooks/dev.ipynb` and run the cells sequentially:

1. **Cell 10**: Install packages (uncomment and run once)
2. **Cell 11**: Import libraries and initialize OpenAI client
3. **Cell 12**: Define Pydantic schema models
4. **Cell 13**: Define extraction function
5. **Cell 14**: Load data
6. **Cell 15**: Test on 5 samples
7. **Cell 16**: View sample result
8. **Cell 17**: Process all recipes (uncomment to run)
9. **Cell 18**: View flattened DataFrame
10. **Cell 19**: Export to multiple formats

## Output Files

After extraction, you'll get:

- `recipes_extracted.json` - Full nested JSON structure
- `recipes_extracted_flat.csv/xlsx` - Flattened table for analysis
- `recipes_extracted_malts.csv` - Normalized malts table
- `recipes_extracted_hops.csv` - Normalized hops table
- `recipes_extracted_errors.json` - Failed extractions (if any)
- `recipes_checkpoint.json` - Checkpoint file for resuming

## Schema

The extracted data follows this structure:

```python
{
  "title": str,
  "style": str,
  "brewer": str | None,
  "year": int | None,
  "competition": {
    "name": str,
    "year": int,
    "award": str,
    "category": str
  },
  "ingredients": {
    "malts": [{"name": str, "amount": float, "unit": str, "color_L": float}],
    "hops": [{"name": str, "amount": float, "unit": str, "alpha_acid_pct": float, "time_min": int, "usage": str}],
    "yeast": {"name": str, "brand": str, "product_code": str},
    "water": {"description": str},
    "adjuncts": [{"item": str, "amount": float, "unit": str, "purpose": str}]
  },
  "specs": {
    "batch_size_gal": float,
    "og": float,
    "fg": float,
    "abv_pct": float,
    "ibu": float,
    "srm": float,
    "efficiency_pct": float
  },
  "directions": {
    "mash": {"temp_F": float, "time_min": int, "notes": str},
    "boil": {"time_min": int, "notes": str},
    "fermentation": {"temp_F": float, "duration_days": int, "notes": str}
  },
  "extract_version": str
}
```

## Cost Estimation

Using GPT-4o:
- ~1395 recipes in dataset
- Average ~1000 tokens per recipe (input + output)
- Estimated cost: ~$5-10 for full dataset

Always test on samples first!

## Features

- ✅ Structured output using Pydantic models
- ✅ Automatic retry with exponential backoff
- ✅ Progress tracking with tqdm
- ✅ Checkpoint system (saves every 50 recipes)
- ✅ Multiple export formats (JSON, CSV, Excel)
- ✅ Normalized tables for database import
- ✅ Error handling and logging
