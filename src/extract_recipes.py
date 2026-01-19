"""
Craft Beer Recipe Extraction using OpenAI 

This script extracts structured data from craft beer recipe text using OpenAI's API.
It processes raw recipe text and outputs structured JSON with ingredients, specifications, and directions.

Usage:
    python extract_recipes.py --sample 5  # Test on 5 recipes
    python extract_recipes.py --all       # Process all recipes
    python extract_recipes.py --resume 100  # Resume from index 100
"""

import os
import json
import argparse
import time
import re
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'
EXTRACTED_DIR = DATASET_DIR / 'extracted'
OUTPUT_FILE = EXTRACTED_DIR / 'recipes_extracted.json'
CHECKPOINT_FILE = EXTRACTED_DIR / 'recipes_checkpoint.json'
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

EXTRACTION_PROMPT = """You are an expert at extracting structured data from craft beer recipes.

Extract all information from the provided beer recipe text into the specified JSON schema.

CORE GUIDELINES:
- Extract amounts with their units (lb, oz, kg, g for malts; oz, g for hops)
- Identify yeast brand (Wyeast, White Labs, Fermentis, Lallemand, etc.) and product codes
- Extract specifications: OG, FG, ABV, IBU, SRM, efficiency, batch size
- Extract year: look for any year mentioned in the recipe text (competition year, publication year, etc.)
- If a field is not present in the text, leave it as null
- For gravity values, use decimal format (e.g., 1.050 not 50)

HOP EXTRACTION:
- Identify usage: "boil" (with time), "flameout" (0 min), "dry hop", "FWH" (first wort hop), "whirlpool", "hop stand"
- Extract alpha acid percentages (look for "a.a.", "AA", or "%")
- For whirlpool/hop stand additions: extract temperature (if stated) and duration in minutes
- Identify biotransformation hops (look for keywords like "geraniol", "linalool", "biotransform")
- Note any special hopping techniques or considerations

MASH EXTRACTION:
- Identify mash type: "Single Infusion", "Step Mash", or "Decoction"
- Extract all mash steps with:
  * Step name (Acid Rest, Protein Rest, Beta Amylase, Alpha Amylase, Mash Out, etc.)
  * Temperature in Fahrenheit
  * Duration in minutes
- For Single Infusion: extract single temperature and time

FERMENTATION EXTRACTION:
- Identify all fermentation stages: Primary, Secondary, Conditioning, Cold Crash, etc.
- For each stage extract:
  * Starting temperature in Fahrenheit
  * Ending temperature (if temperature is ramped)
  * Temperature ramp rate (°F per day) if specified
  * Duration in days
- Example: "Primary at 62°F for 7 days, then ramp to 68°F at 1°F/day" → start_temp: 62, end_temp: 68, ramp_per_day: 1, duration: 7

ADJUNCT EXTRACTION:
- Extract ALL non-malt, non-hop, non-yeast additions including:
  * Sugars (table sugar, brown sugar, honey, molasses, candi sugar, etc.)
  * Spices (cinnamon, cardamom, ginger, etc.)
  * Fruits (oranges, cherries, apples, etc.)
  * Processing aids (Whirlfloc, Irish Moss, Yeast Nutrient, etc.)
  * Salts (table salt, Kosher salt, sea salt, etc.)
  * Flavorings (vanilla, chocolate, coconut, etc.)
- For each adjunct, specify:
  * Amount and unit
  * Category (sugar, spice, fruit, processing_aid, salt, etc.)
  * Timing (mash, boil, flameout, primary, secondary, bottling)
  * Purpose (clarity, priming, flavor, body, etc.)

WATER CHEMISTRY:
- PRIORITY 1: Extract explicit target PPM values (Ca, Mg, Na, Cl, SO4, HCO3) if stated directly
- PRIORITY 2: If only salt additions are listed (e.g., "1 tsp calcium chloride"):
  * Extract salt type, amount, and unit
  * Extract total water volume if available (usually in directions)
  * DO NOT calculate PPM unless both salt amount and total water volume are explicitly confirmed
- Include water source/treatment description if mentioned

DATASET VS. RECIPE DISTINCTION:
- If the text contains statistical data (e.g., "average gravity range 1.040-1.072"), distinguish this from the specific target recipe
- Extract the SPECIFIC RECIPE specifications, not the dataset averages
- If both statistical and specific data exist, use the specific recipe data for the JSON fields
"""


# ============================================================================
# Preprocessing Functions
# ============================================================================

def map_img_src_to_medal(img_src: pd.Series) -> pd.Series:
    """Map image source URLs to medal types."""
    medal_mapping = {
        'https://cdn.homebrewersassociation.org/wp-content/uploads/2021/02/04134634/medal-1-1.svg': 'NHC GOLD',
        'https://cdn.homebrewersassociation.org/wp-content/uploads/2021/02/04134237/medal-0-1.svg': 'NORMAL MEDAL',
        'https://cdn.homebrewersassociation.org/wp-content/uploads/2021/02/04134238/clone-flag.svg': 'CLONE',
        'https://cdn.homebrewersassociation.org/wp-content/uploads/2021/02/04134239/pro-am-flag.svg': 'PRO AM',
        'https://cdn.homebrewersassociation.org/wp-content/uploads/2021/02/04134635/medal-3-1.svg': 'NHC COPPER',
        'https://cdn.homebrewersassociation.org/wp-content/uploads/2021/02/04134633/medal-2-1.svg': 'NHC SILVER',
        'Not Found': 'No Medal'
    }
    return img_src.map(medal_mapping)


def find_index_topics(ingredient: list) -> dict:
    """Find the line index for each ingredient topic."""
    idx_topics = {
        'MALTS': np.nan,
        'Fermentable': np.nan,
        'EXTRACT': np.nan,
        'HOPS': np.nan,
        'WATER': np.nan,
        'YEAST': np.nan,
        'ADDITIONAL': np.nan,
        'Specifications': np.nan,
        'Yield': np.nan,
        'Original Gravity': np.nan,
        'Final Gravity': np.nan,
        'ABV': np.nan,
        'IBU': np.nan,
        'SRM': np.nan,
        'Efficiency': np.nan,
    }

    for i, text in enumerate(ingredient):
        for topics in idx_topics.keys():
            if text.lower().find(topics.lower()) == 0:
                idx_topics[topics] = i

    return idx_topics


def find_index_not_nan(keys: list, idx_topics: dict) -> float:
    """Find the first non-NaN index from a list of keys."""
    for key in keys:
        if not np.isnan(idx_topics[key]):
            return idx_topics[key]
    return np.nan


def extract_ingredients(start_keys: list, end_keys: list, idx_topics: dict, ingredient: list) -> Optional[str]:
    """Extract ingredients between start and end topics."""
    i_start = find_index_not_nan(keys=start_keys, idx_topics=idx_topics)

    if np.isnan(i_start):
        return None

    filtered_values = [v for v in idx_topics.values() if v > i_start]

    if not filtered_values:
        return None

    i_end = min(filtered_values, key=lambda v: v - i_start)

    if not np.isnan(i_start) and not np.isnan(i_end):
        return '\n'.join(ingredient[int(i_start) + 1:int(i_end)])
    return ''


def recheck_and_finalize_category(df: pd.DataFrame) -> pd.DataFrame:
    """Recheck and finalize recipe category (beer, cider, mead) based on ingredients."""
    # Recheck category based on ingredients
    type_ = []
    for desc in df['ingredients']:
        types = [t for t in ['hops', 'malt', 'honey', 'cider', 'mead', 'apple'] if t in desc.lower()]
        type_.append(types)
    df['recheck_category'] = type_

    # Finalize category
    category_final = []
    for cat, recheck in df[['category', 'recheck_category']].values:
        if 'Mead' in cat:
            if ('malt' in recheck) or ('hops' in recheck):
                cat_final = 'Beer'
            else:
                cat_final = 'Mead'
        elif 'Cider' in cat:
            if ('malt' in recheck) or ('hops' in recheck):
                cat_final = 'Beer'
            else:
                cat_final = 'Cider'
        else:
            cat_final = 'Beer'

        category_final.append(cat_final)

    df['final_category'] = category_final
    return df


def extract_year_from_text(df: pd.DataFrame) -> pd.DataFrame:
    """Extract year from recipe description using regex."""
    year_in_text = []
    for descrip in df['description'].values:
        year = [str(i) for i in np.arange(1900, 2040) if str(i) in descrip]
        year_in_text.append(year)

    df['year'] = year_in_text
    df['year'] = df['year'].apply(lambda x: max(x) if len(x) > 0 else np.nan)
    return df


# ============================================================================
# Pydantic Models for Schema
# ============================================================================

class Malt(BaseModel):
    name: str = Field(description="Name of the malt/grain")
    amount: float = Field(description="Amount of malt")
    unit: str = Field(description="Unit: lb, oz, kg, or g")
    color_L: Optional[float] = Field(default=None, description="Color in Lovibond")


class Hop(BaseModel):
    name: str = Field(description="Name of the hop variety")
    amount: float = Field(description="Amount of hops")
    unit: str = Field(description="Unit: oz or g")
    alpha_acid_pct: Optional[float] = Field(default=None, description="Alpha acid percentage")
    time_min: Optional[int] = Field(default=None, description="Boil time in minutes, 0 for flameout")
    usage: str = Field(description="Usage: boil, flameout, dry hop, FWH (first wort hop), whirlpool, hop stand")
    whirlpool_temp_F: Optional[float] = Field(default=None, description="Whirlpool/hop stand temperature in Fahrenheit")
    whirlpool_duration_min: Optional[int] = Field(default=None, description="Whirlpool/hop stand duration in minutes")
    biotransformation: Optional[bool] = Field(default=None, description="Whether hop is used for biotransformation")
    notes: Optional[str] = Field(default=None, description="Additional notes about the hop usage")


class Yeast(BaseModel):
    name: str = Field(description="Name of the yeast strain")
    brand: Optional[str] = Field(default=None, description="Brand: Wyeast, White Labs, Fermentis, Lallemand, etc.")
    product_code: Optional[str] = Field(default=None, description="Product code: 1056, WLP001, US-05, etc.")
    starter_volume_L: Optional[float] = Field(default=None, description="Yeast starter volume in liters")


class MineralProfile(BaseModel):
    """Mineral profile with specific fields for common brewing minerals."""
    model_config = {"extra": "forbid"}

    Ca: Optional[float] = Field(default=None, description="Calcium in ppm")
    Mg: Optional[float] = Field(default=None, description="Magnesium in ppm")
    Na: Optional[float] = Field(default=None, description="Sodium in ppm")
    Cl: Optional[float] = Field(default=None, description="Chloride in ppm")
    SO4: Optional[float] = Field(default=None, description="Sulfate in ppm")
    HCO3: Optional[float] = Field(default=None, description="Bicarbonate in ppm")


class WaterSalt(BaseModel):
    salt_type: str = Field(description="Type of salt: CaCl2, CaSO4, NaCl, NaHCO3, MgSO4, etc.")
    amount: float = Field(description="Amount of salt")
    unit: str = Field(description="Unit: g, oz, tsp")
    notes: Optional[str] = Field(default=None, description="Notes about the salt addition")


class Water(BaseModel):
    description: Optional[str] = Field(default=None, description="Water source and treatment description")
    minerals_ppm: Optional[MineralProfile] = Field(default=None, description="Target mineral levels in ppm (if explicitly stated)")
    salt_additions: list[WaterSalt] = Field(default_factory=list, description="Salt additions (if PPM not explicitly stated)")
    total_water_volume_gal: Optional[float] = Field(default=None, description="Total water volume in gallons (for PPM calculations)")


class Adjunct(BaseModel):
    item: str = Field(description="Name of the adjunct/addition")
    amount: Optional[float] = Field(default=None, description="Amount")
    unit: Optional[str] = Field(default=None, description="Unit: oz, g, tsp, lb, ml, etc.")
    category: Optional[str] = Field(default=None, description="Category: sugar, spice, fruit, processing_aid, salt, etc.")
    timing: Optional[str] = Field(default=None, description="When added: mash, boil, flameout, primary, secondary, bottling")
    purpose: Optional[str] = Field(default=None, description="Purpose: clarity, priming, flavor, body, etc.")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class Ingredients(BaseModel):
    malts: list[Malt] = Field(default_factory=list, description="List of malts and fermentables")
    hops: list[Hop] = Field(default_factory=list, description="List of hops")
    yeast: Optional[Yeast] = Field(default=None, description="Yeast information")
    water: Optional[Water] = Field(default=None, description="Water profile")
    adjuncts: list[Adjunct] = Field(default_factory=list, description="Additional ingredients")


class Specs(BaseModel):
    batch_size_gal: Optional[float] = Field(default=None, description="Batch size in gallons")
    boil_time_min: Optional[int] = Field(default=None, description="Boil time in minutes")
    og: Optional[float] = Field(default=None, description="Original gravity (e.g., 1.050)")
    fg: Optional[float] = Field(default=None, description="Final gravity (e.g., 1.010)")
    abv_pct: Optional[float] = Field(default=None, description="Alcohol by volume percentage")
    ibu: Optional[float] = Field(default=None, description="International Bitterness Units")
    srm: Optional[float] = Field(default=None, description="Standard Reference Method (color)")
    efficiency_pct: Optional[float] = Field(default=None, description="Mash efficiency percentage")


class MashStep(BaseModel):
    name: Optional[str] = Field(default=None, description="Step name: Acid Rest, Protein Rest, Beta Amylase, Alpha Amylase, Mash Out, etc.")
    temp_F: float = Field(description="Temperature in Fahrenheit")
    time_min: int = Field(description="Duration in minutes")


class MashDirections(BaseModel):
    type: Optional[str] = Field(default=None, description="Mash type: Single Infusion, Step Mash, Decoction")
    steps: list[MashStep] = Field(default_factory=list, description="List of mash steps with temperatures and times")
    notes: Optional[str] = Field(default=None, description="Additional mash notes")


class BoilDirections(BaseModel):
    time_min: Optional[int] = Field(default=None, description="Boil time in minutes")
    notes: Optional[str] = Field(default=None, description="Additional boil notes")


class FermentationStage(BaseModel):
    stage: str = Field(description="Stage name: Primary, Secondary, Conditioning, Cold Crash, etc.")
    start_temp_F: Optional[float] = Field(default=None, description="Starting temperature in Fahrenheit")
    end_temp_F: Optional[float] = Field(default=None, description="Ending temperature in Fahrenheit (for ramps)")
    duration_days: Optional[int] = Field(default=None, description="Duration in days")
    ramp_per_day_F: Optional[float] = Field(default=None, description="Temperature increase per day during ramp")


class FermentationDirections(BaseModel):
    stages: list[FermentationStage] = Field(default_factory=list, description="Fermentation stages with temperature schedules")
    notes: Optional[str] = Field(default=None, description="Additional fermentation notes")


class Directions(BaseModel):
    mash: Optional[MashDirections] = Field(default=None, description="Mash directions")
    boil: Optional[BoilDirections] = Field(default=None, description="Boil directions")
    fermentation: Optional[FermentationDirections] = Field(default=None, description="Fermentation directions")


class Competition(BaseModel):
    name: Optional[str] = Field(default=None, description="Competition name")
    year: Optional[int] = Field(default=None, description="Competition year")
    award: Optional[str] = Field(default=None, description="Award: Gold Medal, Silver Medal, Bronze Medal")
    category: Optional[str] = Field(default=None, description="Competition category")


class Recipe(BaseModel):
    title: str = Field(description="Recipe title/name")
    style: Optional[str] = Field(default=None, description="Beer style")
    brewer: Optional[str] = Field(default=None, description="Brewer name")
    source: Optional[str] = Field(default=None, description="Source publication and date")
    year: Optional[int] = Field(default=None, description="Recipe year")
    competition: Optional[Competition] = Field(default=None, description="Competition information if any")
    description: Optional[str] = Field(default=None, description="Short description of the beer")
    ingredients: Ingredients = Field(description="Recipe ingredients")
    specs: Specs = Field(description="Beer specifications")
    directions: Directions = Field(description="Brewing directions")
    extract_version: Optional[str] = Field(default=None, description="Extract brewing version instructions if available")


# ============================================================================
# Extraction Functions
# ============================================================================

def extract_recipe(
    description: str,
    ingredients: str,
    directions: str,
    addition: str = "",
    title: str = "",
    style: str = "",
    max_retries: int = 3
) -> dict:
    """Extract structured recipe data using OpenAI"""

    # Combine all text for extraction
    full_text = f"""
TITLE: {title}
STYLE: {style}

DESCRIPTION:
{description}

INGREDIENTS:
{ingredients}

DIRECTIONS:
{directions}

EXTRACT VERSION:
{addition if addition and addition != 'N/A' else 'Not provided'}
"""

    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": full_text}
                ],
                response_format=Recipe,
            )

            # Get the parsed recipe object
            recipe = response.choices[0].message.parsed
            return recipe.model_dump()

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return {"error": str(e), "title": title}

    return {"error": "Unknown error", "title": title}


def load_data(preprocess: bool = False):
    """Load recipe data from CSV files with optional preprocessing."""
    print("Loading recipe data...")
    df1 = pd.read_csv(DATASET_DIR / 'all_recipes_new.csv')
    df2 = pd.read_csv(DATASET_DIR / 'all_recipes_new1.csv')
    df_recipes = pd.concat([df1, df2]).reset_index(drop=True)

    # Load URL data for style and beer_name
    df_url = pd.read_csv(DATASET_DIR / 'all_urls.csv')
    df_url = df_url.drop(['Unnamed: 0'], axis=1)
    df_url = df_url.groupby('url').agg({
        'style': 'first',
        'beer_name': 'first',
        'final_results': 'first',
        'category': list
    }).reset_index()

    # Merge to get style and beer_name
    df_recipes = pd.merge(df_recipes, df_url[['beer_name', 'style', 'url', 'category']], how='left', on='url')
    df_recipes = df_recipes.fillna('')

    if preprocess:
        print("Preprocessing data...")

        # Extract medal information
        if 'src' in df_recipes.columns:
            df_recipes['medal'] = map_img_src_to_medal(df_recipes['src'])

        # Recheck and finalize categories
        df_recipes = recheck_and_finalize_category(df_recipes)

        # Extract year from description
        df_recipes = extract_year_from_text(df_recipes)

        print(f"Preprocessing complete. Final shape: {df_recipes.shape}")

    return df_recipes


def process_all_recipes(df, start_idx=0, save_every=50):
    """Process all recipes with checkpointing."""

    # Load existing results if resuming
    if start_idx > 0 and os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            results = json.load(f)
        print(f"Resuming from index {start_idx}, loaded {len(results)} existing results")
    else:
        results = []

    # Process remaining recipes
    for idx in tqdm(range(start_idx, len(df)), desc="Extracting recipes"):
        row = df.iloc[idx]

        try:
            result = extract_recipe(
                description=row['description'],
                ingredients=row['ingredients'],
                directions=row['directions'],
                addition=row.get('addition', ''),
                title=row['beer_name'],
                style=row['style']
            )

            # Add metadata
            result['url'] = row['url']
            result['original_category'] = row.get('category', [])
            result['index'] = idx

            # Add preprocessed data if available
            if 'medal' in row:
                result['medal'] = row['medal']
            if 'final_category' in row:
                result['final_category'] = row['final_category']

            results.append(result)

            # Save checkpoint periodically
            if (idx + 1) % save_every == 0:
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nCheckpoint saved at index {idx + 1}")

        except Exception as e:
            print(f"\nError at index {idx}: {e}")
            results.append({
                "error": str(e),
                "title": row['beer_name'],
                "url": row['url'],
                "index": idx
            })

        # Rate limiting - avoid hitting API limits
        time.sleep(0.5)

    # Save final results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nCompleted! Saved {len(results)} recipes to {OUTPUT_FILE}")
    return results


# ============================================================================
# Export Functions
# ============================================================================

def flatten_recipe(recipe: dict) -> dict:
    """Flatten nested recipe structure for DataFrame."""
    # Determine year: priority 1 = competition year, priority 2 = recipe year (from source/magazine)
    competition_year = recipe.get('competition', {}).get('year') if recipe.get('competition') else None
    recipe_year = recipe.get('year')
    final_year = competition_year if competition_year is not None else recipe_year

    flat = {
        'title': recipe.get('title'),
        'style': recipe.get('style'),
        'year': final_year,  # Competition year preferred, falls back to recipe/publication year
        'url': recipe.get('url'),
        'original_category': recipe.get('original_category'),
        'final_category': recipe.get('final_category'),
        'medal': recipe.get('medal'),

        # Specs
        'batch_size_gal': recipe.get('specs', {}).get('batch_size_gal'),
        'og': recipe.get('specs', {}).get('og'),
        'fg': recipe.get('specs', {}).get('fg'),
        'abv_pct': recipe.get('specs', {}).get('abv_pct'),
        'ibu': recipe.get('specs', {}).get('ibu'),
        'srm': recipe.get('specs', {}).get('srm'),
        'efficiency_pct': recipe.get('specs', {}).get('efficiency_pct'),

        # Ingredients as JSON strings for storage
        'malts_json': json.dumps(recipe.get('ingredients', {}).get('malts', [])),
        'hops_json': json.dumps(recipe.get('ingredients', {}).get('hops', [])),
        'yeast_json': json.dumps(recipe.get('ingredients', {}).get('yeast')),
        'adjuncts_json': json.dumps(recipe.get('ingredients', {}).get('adjuncts', [])),

        # Counts for quick analysis
        'num_malts': len(recipe.get('ingredients', {}).get('malts', [])),
        'num_hops': len(recipe.get('ingredients', {}).get('hops', [])),
        'num_adjuncts': len(recipe.get('ingredients', {}).get('adjuncts', [])),

        # Directions - Mash
        'mash_type': recipe.get('directions', {}).get('mash', {}).get('type') if recipe.get('directions', {}).get('mash') else None,
        'mash_steps_json': json.dumps(recipe.get('directions', {}).get('mash', {}).get('steps', []) if recipe.get('directions', {}).get('mash') else []),
        'num_mash_steps': len(recipe.get('directions', {}).get('mash', {}).get('steps', [])) if recipe.get('directions', {}).get('mash') else 0,

        # Directions - Boil
        'boil_time_min': recipe.get('directions', {}).get('boil', {}).get('time_min') if recipe.get('directions', {}).get('boil') else None,

        # Directions - Fermentation
        'fermentation_stages_json': json.dumps(recipe.get('directions', {}).get('fermentation', {}).get('stages', []) if recipe.get('directions', {}).get('fermentation') else []),
        'num_fermentation_stages': len(recipe.get('directions', {}).get('fermentation', {}).get('stages', [])) if recipe.get('directions', {}).get('fermentation') else 0,

        # Water Chemistry
        'water_description': recipe.get('ingredients', {}).get('water', {}).get('description') if recipe.get('ingredients', {}).get('water') else None,
        'water_Ca_ppm': recipe.get('ingredients', {}).get('water', {}).get('minerals_ppm', {}).get('Ca') if (recipe.get('ingredients', {}).get('water') and recipe.get('ingredients', {}).get('water', {}).get('minerals_ppm')) else None,
        'water_Mg_ppm': recipe.get('ingredients', {}).get('water', {}).get('minerals_ppm', {}).get('Mg') if (recipe.get('ingredients', {}).get('water') and recipe.get('ingredients', {}).get('water', {}).get('minerals_ppm')) else None,
        'water_Na_ppm': recipe.get('ingredients', {}).get('water', {}).get('minerals_ppm', {}).get('Na') if (recipe.get('ingredients', {}).get('water') and recipe.get('ingredients', {}).get('water', {}).get('minerals_ppm')) else None,
        'water_Cl_ppm': recipe.get('ingredients', {}).get('water', {}).get('minerals_ppm', {}).get('Cl') if (recipe.get('ingredients', {}).get('water') and recipe.get('ingredients', {}).get('water', {}).get('minerals_ppm')) else None,
        'water_SO4_ppm': recipe.get('ingredients', {}).get('water', {}).get('minerals_ppm', {}).get('SO4') if (recipe.get('ingredients', {}).get('water') and recipe.get('ingredients', {}).get('water', {}).get('minerals_ppm')) else None,
        'water_HCO3_ppm': recipe.get('ingredients', {}).get('water', {}).get('minerals_ppm', {}).get('HCO3') if (recipe.get('ingredients', {}).get('water') and recipe.get('ingredients', {}).get('water', {}).get('minerals_ppm')) else None,
        'water_salt_additions_json': json.dumps(recipe.get('ingredients', {}).get('water', {}).get('salt_additions', [])) if recipe.get('ingredients', {}).get('water') else None,
        'water_volume_gal': recipe.get('ingredients', {}).get('water', {}).get('total_water_volume_gal') if recipe.get('ingredients', {}).get('water') else None,

        # Extract version
        'extract_version': recipe.get('extract_version'),
    }
    return flat


def save_extracted_data(results: list, base_path: str = None):
    """Save extracted recipes to multiple formats."""

    if base_path is None:
        base_path = EXTRACTED_DIR / 'recipes_extracted'
    else:
        base_path = Path(base_path)

    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]

    print(f"\nValid recipes: {len(valid_results)}")
    print(f"Errors: {len(error_results)}")

    # 1. Save full JSON (nested structure)
    with open(f'{base_path}.json', 'w') as f:
        json.dump(valid_results, f, indent=2, default=str)
    print(f"Saved: {base_path}.json")

    # 2. Save flattened CSV
    df_flat = pd.DataFrame([flatten_recipe(r) for r in valid_results])
    df_flat.to_csv(f'{base_path}_flat.csv', index=False, quoting=1, escapechar='\\')  # quoting=csv.QUOTE_ALL
    print(f"Saved: {base_path}_flat.csv")

    # 3. Save normalized tables for database
    # Malts table
    malts_rows = []
    for i, r in enumerate(valid_results):
        for malt in r.get('ingredients', {}).get('malts', []):
            malts_rows.append({
                'recipe_id': i,
                'recipe_title': r.get('title'),
                **malt
            })
    if malts_rows:
        df_malts = pd.DataFrame(malts_rows)
        df_malts.to_csv(f'{base_path}_malts.csv', index=False, quoting=1, escapechar='\\')
        print(f"Saved: {base_path}_malts.csv ({len(df_malts)} rows)")

    # Hops table
    hops_rows = []
    for i, r in enumerate(valid_results):
        for hop in r.get('ingredients', {}).get('hops', []):
            hops_rows.append({
                'recipe_id': i,
                'recipe_title': r.get('title'),
                **hop
            })
    if hops_rows:
        df_hops = pd.DataFrame(hops_rows)
        df_hops.to_csv(f'{base_path}_hops.csv', index=False, quoting=1, escapechar='\\')
        print(f"Saved: {base_path}_hops.csv ({len(df_hops)} rows)")

    # Save errors for review
    if error_results:
        with open(f'{base_path}_errors.json', 'w') as f:
            json.dump(error_results, f, indent=2)
        print(f"Saved: {base_path}_errors.json")

    return df_flat


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Extract craft beer recipes using OpenAI')
    parser.add_argument('--sample', type=int, help='Process only N sample recipes for testing')
    parser.add_argument('--all', action='store_true', help='Process all recipes')
    parser.add_argument('--resume', type=int, help='Resume processing from index N')
    parser.add_argument('--export', type=str, help='Export existing results from JSON file')
    parser.add_argument('--preprocess', action='store_true', help='Enable preprocessing (medals, categories, years)')

    args = parser.parse_args()

    # Export mode
    if args.export:
        print(f"Loading results from {args.export}...")
        with open(args.export, 'r') as f:
            results = json.load(f)
        save_extracted_data(results)
        return

    # Load data with optional preprocessing
    df = load_data(preprocess=args.preprocess)
    print(f"Total recipes: {len(df)}")

    # Determine range
    if args.sample:
        start_idx = 0
        df = df.head(args.sample)
        print(f"\nProcessing {args.sample} sample recipes...")
    elif args.resume is not None:
        start_idx = args.resume
        print(f"\nResuming from index {start_idx}...")
    elif args.all:
        start_idx = 0
        print(f"\nProcessing all {len(df)} recipes...")
    else:
        print("\nNo mode specified. Use --sample N, --all, or --resume N")
        print("Example: python extract_recipes.py --sample 5")
        return

    # Process recipes
    results = process_all_recipes(df, start_idx=start_idx)

    # Save in multiple formats
    print("\nExporting results...")
    save_extracted_data(results)

    print("\n✓ Done!")


if __name__ == '__main__':
    main()
