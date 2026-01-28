"""
Recipe Data Preprocessing Pipeline

Converts raw extracted recipes into normalized tables optimized for dashboard analysis.
Handles:
1. JSON field explosion
2. Data quality validation
3. Style categorization
4. Ingredient normalization
5. Malt composition analysis
6. Hop schedule extraction
7. Water chemistry standardization
8. Yeast strain normalization
9. Competition/award analysis
10. Time-series preparation

Output: 7 normalized CSV tables ready for dashboard visualization
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from difflib import get_close_matches

# ============================================================================
# Configuration
# ============================================================================

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'
EXTRACTED_DIR = DATASET_DIR / 'extracted'
INPUT_FILE = EXTRACTED_DIR / 'recipes_extracted_flat.csv'
OUTPUT_DIR = DATASET_DIR / 'processed'
REFERENCE_DIR = DATASET_DIR / 'reference'
OUTPUT_DIR.mkdir(exist_ok=True)
REFERENCE_DIR.mkdir(exist_ok=True)

# Style categorization mapping (style_group - more granular)
STYLE_GROUPS = {
    'IPA': ['IPA', 'PALE ALE', 'AMERICAN IPA', 'IMPERIAL IPA', 'DOUBLE IPA', 'NEIPA', 'HAZY', 'ENGLISH IPA', 'WEST COAST'],
    'LAGER': ['LAGER', 'PILSNER', 'CZECH', 'GERMAN', 'PALE LAGER', 'LIGHT LAGER', 'AMBER LAGER', 'DUNKEL', 'BOCK', 'MÄRZEN'],
    'STOUT': ['STOUT', 'IMPERIAL STOUT', 'DRY STOUT', 'IRISH STOUT', 'OATMEAL STOUT', 'MILK STOUT', 'SWEET STOUT', 'FOREIGN STOUT', 'AMERICAN STOUT'],
    'PORTER': ['PORTER', 'ENGLISH PORTER', 'AMERICAN PORTER', 'BALTIC PORTER', 'ROBUST PORTER', 'BROWN PORTER'],
    'STRONG_ALE': ['BARLEYWINE', 'WEE HEAVY', 'BELGIAN DARK STRONG', 'IMPERIAL', 'STRONG ALE', 'ENGLISH STRONG'],
    'AMBER_ALE': ['AMBER', 'RED ALE', 'AMBER ALE'],
    'WHEAT_BEER': ['WHEAT', 'WEIZEN', 'HEFEWEIZEN', 'WITBIER', 'WITBEER', 'WHITE'],
    'BLONDE_PALE': ['BLONDE', 'PALE', 'LIGHT ALE', 'CREAM ALE'],
    'BELGIAN': ['BELGIAN', 'ABBEY', 'TRAPPIST', 'TRIPEL', 'DUBBEL', 'SAISON', 'FARMHOUSE'],
    'BROWN_ALE': ['BROWN', 'NEWCASTLE', 'ENGLISH BROWN'],
    'SOUR': ['SOUR', 'LAMBIC', 'GOSE', 'BERLINER'],
}

# Malt type classification
MALT_TYPES = {
    'base': ['pale', 'pilsner', 'maris otter', 'golden promise', 'american', '2-row', 'munich', 'vienna'],
    'crystal': ['crystal', 'caramel', 'cara'],
    'roast': ['chocolate', 'black', 'roast', 'patent'],
    'wheat': ['wheat', 'flaked wheat'],
    'adjunct': ['corn', 'rice', 'oats', 'flaked', 'dextrose', 'sugar', 'honey'],
}

# Yeast strain normalization
YEAST_NORMALIZATION = {
    '1056': ('Wyeast', 'Chico/US-05 (American Ale)'),
    'WLP001': ('White Labs', 'Chico/US-05 (American Ale)'),
    'US-05': ('Fermentis', 'Chico/US-05 (American Ale)'),
    '1728': ('Wyeast', 'Scottish Ale'),
    'WLP028': ('White Labs', 'Scottish Ale'),
    '34/70': ('Wyeast', '34/70 (Czech Lager)'),
}

# Salt to PPM conversion factors
SALT_TO_PPM = {
    'calcium chloride': {'Ca': 270, 'Cl': 482},
    'cacl2': {'Ca': 270, 'Cl': 482},
    'gypsum': {'Ca': 610, 'SO4': 1430},
    'casO4': {'Ca': 610, 'SO4': 1430},
    'salt': {'Na': 390, 'Cl': 630},
    'nacl': {'Na': 390, 'Cl': 630},
    'baking soda': {'Na': 380, 'HCO3': 1190},
    'nahco3': {'Na': 380, 'HCO3': 1190},
    'epsom salt': {'Mg': 104, 'SO4': 1120},
    'mgso4': {'Mg': 104, 'SO4': 1120},
}

# Parent style categorization (11 categories)
PARENT_STYLES = {
    'Ale': [
        'PALE ALE', 'IPA', 'AMERICAN IPA', 'IMPERIAL IPA', 'DOUBLE IPA', 'NEIPA', 'HAZY',
        'ENGLISH IPA', 'WEST COAST', 'BITTER', 'MILD', 'ESB', 'EXTRA SPECIAL',
        'AMBER ALE', 'BROWN ALE', 'BLONDE', 'GOLDEN ALE',
        'BELGIAN ALE', 'ABBEY', 'TRAPPIST', 'TRIPEL', 'DUBBEL', 'QUAD',
        'BELGIAN DARK STRONG', 'BELGIAN PALE', 'BELGIAN GOLDEN',
        'SAISON', 'FARMHOUSE', 'WITBIER', 'WITBEER',
        'BIÈRE DE GARDE', 'BIERE DE GARDE',
        'SCOTTISH', 'IRISH RED', 'RED ALE', 'OLD ALE', 'BARLEYWINE', 'WEE HEAVY',
        'STRONG ALE', 'ENGLISH STRONG', 'UK/US STRONG',
        'KOLSCH', 'KÖLSCH', 'ALTBIER', 'ALT',
        'AMERICAN WHEAT', 'SPECIALTY IPA',
    ],
    'Lager': [
        'LAGER', 'PILSNER', 'PILS', 'CZECH', 'BOHEMIAN', 'GERMAN PILSNER',
        'AMERICAN PILSNER', 'AMERICAN LAGER', 'LIGHT LAGER', 'PALE LAGER',
        'HELLES', 'MUNICH HELLES', 'VIENNA', 'VIENNA LAGER',
        'MÄRZEN', 'MARZEN', 'OKTOBERFEST', 'FESTBIER',
        'BOCK', 'DOPPELBOCK', 'EISBOCK', 'HELLES BOCK', 'MAIBOCK',
        'AMBER LAGER', 'DARK LAGER', 'DUNKEL', 'MUNICH DUNKEL',
        'SCHWARZBIER', 'RAUCHBIER', 'SMOKED LAGER',
    ],
    'Stout_Porter': [
        'STOUT', 'DRY STOUT', 'IRISH STOUT', 'SWEET STOUT', 'MILK STOUT',
        'OATMEAL STOUT', 'IMPERIAL STOUT', 'RUSSIAN IMPERIAL',
        'FOREIGN STOUT', 'TROPICAL STOUT', 'AMERICAN STOUT',
        'COFFEE STOUT', 'CHOCOLATE STOUT', 'EXPORT STOUT',
        'PORTER', 'ENGLISH PORTER', 'AMERICAN PORTER', 'BALTIC PORTER',
        'ROBUST PORTER', 'BROWN PORTER', 'SMOKE PORTER',
    ],
    'Wheat Beer': [
        'WHEAT', 'WEIZEN', 'HEFEWEIZEN', 'DUNKELWEIZEN', 'WEIZENBOCK',
        'ROGGENBIER', 'KRISTALLWEIZEN', 'WHEAT BEER', 'WHEAT & RYE',
        'BERLINER WEISSE', 'BERLINER',
    ],
    'Sour & Wild': [
        'SOUR', 'SOUR ALE', 'FRUITED SOUR', 'KETTLE SOUR',
        'GOSE', 'GUEUZE', 'LAMBIC', 'FLANDERS', 'FLANDERS RED', 'OUD BRUIN',
        'WILD ALE', 'MIXED FERMENTATION', 'BRETT', 'BRETTANOMYCES',
    ],
    'Specialty': [
        'FRUIT BEER', 'FRUIT', 'SPICE', 'HERB', 'SPICE & HERB', 'VEGETABLE',
        'PUMPKIN', 'AUTUMN', 'WINTER', 'CHRISTMAS', 'HOLIDAY',
        'WOOD', 'BARREL', 'BARREL-AGED', 'BOURBON BARREL',
        'SMOKE', 'SMOKED', 'RAUCH',
        'HISTORICAL', 'EXPERIMENTAL', 'EXP',
        'CHOCOLATE BEER', 'COFFEE BEER', 'HONEY BEER',
        'GLUTEN', 'GLUTEN-FREE', 'ALTERNATIVE',
    ],
    'Hybrid': [
        'HYBRID', 'HYBRID BEER',
        'CREAM ALE', 'STEAM', 'CALIFORNIA COMMON',
        'BRAGGOT',
    ],
    'Cider & Perry': [
        'CIDER', 'PERRY', 'APPLE', 'PEAR',
        'FRUIT CIDER', 'SPICED CIDER', 'ICE CIDER',
        'ENGLISH CIDER', 'FRENCH CIDER', 'NEW WORLD CIDER',
        'SPECIALTY CIDER',
    ],
    'Mead': [
        'MEAD', 'MELOMEL', 'PYMENT', 'CYSER', 'METHEGLIN',
        'SWEET MEAD', 'DRY MEAD', 'SEMI-SWEET MEAD',
        'FRUIT MEAD', 'BERRY MEAD', 'STONE FRUIT MEAD',
        'STANDARD MEAD', 'EXPERIMENTAL MEAD', 'HISTORICAL MEAD',
    ],
}

# ============================================================================
# Helper Functions
# ============================================================================

def categorize_style(style: str) -> str:
    """Categorize beer style into broader groups."""
    if pd.isna(style) or style == '':
        return 'UNKNOWN'

    style_upper = str(style).upper()
    for group, keywords in STYLE_GROUPS.items():
        for keyword in keywords:
            if keyword in style_upper:
                return group

    return 'OTHER'


def categorize_parent_style(style: str, original_category: str = None, final_category: str = None) -> str:
    """Categorize beer style into parent categories (11 categories).
    Uses original_category as fallback when style is unclear.
    """
    # First try to classify by style name
    if not pd.isna(style) and style != '':
        style_upper = str(style).upper()

        # Check each parent style category
        for parent, keywords in PARENT_STYLES.items():
            for keyword in keywords:
                if keyword in style_upper:
                    return parent

    # Fallback: use original_category if style didn't match
    if not pd.isna(original_category):
        cat_upper = str(original_category).upper()

        # Check for Mead
        if 'MEAD' in cat_upper:
            return 'Mead'
        # Check for Cider
        if 'CIDER' in cat_upper:
            return 'Cider & Perry'
        # Check for Lager (before Ale since some have both)
        if "'LAGER'" in cat_upper and "'ALE'" not in cat_upper:
            return 'Lager'
        # Check for Ale
        if "'ALE'" in cat_upper:
            return 'Ale'

    # Final fallback: use final_category
    if not pd.isna(final_category):
        if final_category == 'Mead':
            return 'Mead'
        elif final_category == 'Cider':
            return 'Cider & Perry'

    return 'Other'


def normalize_malt_type(malt_name: str) -> str:
    """Classify malt into type category."""
    malt_lower = malt_name.lower()

    for mtype, keywords in MALT_TYPES.items():
        for keyword in keywords:
            if keyword in malt_lower:
                return mtype

    return 'specialty'


def normalize_ingredient_name(name: str, ingredient_type: str = 'hop') -> str:
    """Normalize ingredient names by removing common suffixes."""
    if pd.isna(name):
        return None

    name = str(name).strip()

    if ingredient_type == 'hop':
        name = re.sub(r'\s*(hops?|hop pellets)$', '', name, flags=re.IGNORECASE)
    elif ingredient_type == 'malt':
        name = re.sub(r'\s*(malt|grain)$', '', name, flags=re.IGNORECASE)

    return name.strip()


def parse_json_safe(json_str, default=None):
    """Safely parse JSON string."""
    if pd.isna(json_str) or json_str == '' or json_str == '[]':
        return default if default is not None else []

    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else []


def estimate_ppm_from_salt(salt_type: str, amount: float, unit: str, water_gal: float) -> Dict[str, float]:
    """Estimate PPM from salt additions."""
    if pd.isna(water_gal) or water_gal == 0:
        return {}

    salt_type_lower = salt_type.lower()

    # Find matching salt type
    conversion = None
    for salt_key, ppm_dict in SALT_TO_PPM.items():
        if salt_key in salt_type_lower:
            conversion = ppm_dict
            break

    if not conversion:
        return {}

    # Convert amount to teaspoons
    if unit.lower() in ['tsp', 'teaspoon']:
        amount_tsp = amount
    elif unit.lower() in ['oz', 'ounce']:
        amount_tsp = amount * 6  # 1 oz ≈ 6 tsp
    elif unit.lower() in ['g', 'gram']:
        amount_tsp = amount / 5  # 1 tsp ≈ 5g
    else:
        return {}

    # Convert tsp to PPM (per 5 gallons)
    # 1 tsp per 5 gal = baseline
    ppm_per_5gal = amount_tsp / (water_gal / 5)

    result = {}
    for mineral, ppm_base in conversion.items():
        result[mineral] = ppm_base * ppm_per_5gal

    return result


# ============================================================================
# Step 1-2: Load and Validate Data
# ============================================================================

def load_and_validate_data(input_file):
    """Load CSV and validate data quality."""
    print("Step 1-2: Loading and validating data...")

    df = pd.read_csv(input_file)

    # Add recipe_id as first column
    df.insert(0, 'recipe_id', range(len(df)))

    # Convert original_category to uppercase
    if 'original_category' in df.columns:
        df['original_category'] = df['original_category'].apply(
            lambda x: x.upper() if pd.notna(x) and isinstance(x, str) else x
        )

    print(f"  ✓ Loaded {len(df)} recipes")

    # Check for required fields
    required_fields = ['title', 'style', 'year', 'og', 'fg', 'abv_pct', 'ibu', 'srm', 'malts_json', 'hops_json']
    missing_fields = [f for f in required_fields if f not in df.columns]

    if missing_fields:
        print(f"  ⚠ Missing fields: {missing_fields}")

    # Data quality report
    print("\n  Data Quality Report:")
    for field in ['year', 'og', 'fg', 'abv_pct', 'ibu', 'srm']:
        missing_count = df[field].isna().sum()
        pct = (missing_count / len(df)) * 100
        print(f"    {field}: {missing_count} missing ({pct:.1f}%)")

    return df


# ============================================================================
# Step 3: Style Categorization
# ============================================================================

def process_styles(recipes_df):
    """Add style group and parent style categorization."""
    print("\nStep 3: Categorizing styles...")

    recipes_df['style_group'] = recipes_df['style'].apply(categorize_style)

    # Use original_category and final_category as fallback for parent_style
    recipes_df['parent_style'] = recipes_df.apply(
        lambda row: categorize_parent_style(
            row['style'],
            row.get('original_category'),
            row.get('final_category')
        ),
        axis=1
    )

    print("  Style group distribution:")
    style_counts = recipes_df['style_group'].value_counts()
    for style, count in style_counts.items():
        print(f"    {style}: {count}")

    print("\n  Parent style distribution:")
    parent_counts = recipes_df['parent_style'].value_counts()
    for parent, count in parent_counts.items():
        print(f"    {parent}: {count}")

    return recipes_df


# ============================================================================
# Step 4-5: Malt Composition Analysis
# ============================================================================

def process_malts(recipes_df):
    """Explode and normalize malt data."""
    print("\nStep 4-5: Processing malts...")

    malts_rows = []

    for idx, row in recipes_df.iterrows():
        malts = parse_json_safe(row['malts_json'], [])

        if not malts:
            continue

        # Calculate total weight
        total_weight = 0
        for malt in malts:
            amount = malt.get('amount', 0)
            unit = malt.get('unit', 'lb')
            # Convert to lb
            if unit == 'oz':
                amount = amount / 16
            elif unit == 'kg':
                amount = amount * 2.20462
            elif unit == 'g':
                amount = amount / 453.592
            total_weight += amount

        # Process each malt
        for malt in malts:
            amount = malt.get('amount', 0)
            unit = malt.get('unit', 'lb')

            # Convert to lb
            if unit == 'oz':
                amount_lb = amount / 16
            elif unit == 'kg':
                amount_lb = amount * 2.20462
            elif unit == 'g':
                amount_lb = amount / 453.592
            else:
                amount_lb = amount

            pct_of_grist = (amount_lb / total_weight * 100) if total_weight > 0 else 0

            malts_rows.append({
                'recipe_id': idx,
                'recipe_title': row['title'],
                'style': row['style'],
                'style_group': row['style_group'],
                'year': row['year'],
                'malt_name': malt.get('name', ''),
                'malt_name_normalized': normalize_ingredient_name(malt.get('name', ''), 'malt'),
                'amount_lb': amount_lb,
                'color_L': malt.get('color_L'),
                'malt_type': normalize_malt_type(malt.get('name', '')),
                'pct_of_grist': pct_of_grist,
            })

    malts_df = pd.DataFrame(malts_rows)

    # Calculate grist composition per recipe
    grist_composition = malts_df.groupby('recipe_id').apply(
        lambda x: {
            'base_malt_pct': x[x['malt_type'] == 'base']['pct_of_grist'].sum(),
            'crystal_pct': x[x['malt_type'] == 'crystal']['pct_of_grist'].sum(),
            'roast_pct': x[x['malt_type'] == 'roast']['pct_of_grist'].sum(),
            'adjunct_pct': x[x['malt_type'] == 'adjunct']['pct_of_grist'].sum(),
        }
    ).to_dict()

    recipes_df['grist_composition'] = recipes_df.index.map(
        lambda idx: grist_composition.get(idx, {})
    )

    # Extract to separate columns
    recipes_df['base_malt_pct'] = recipes_df['grist_composition'].apply(lambda x: x.get('base_malt_pct', 0))
    recipes_df['crystal_pct'] = recipes_df['grist_composition'].apply(lambda x: x.get('crystal_pct', 0))
    recipes_df['roast_pct'] = recipes_df['grist_composition'].apply(lambda x: x.get('roast_pct', 0))
    recipes_df['adjunct_pct'] = recipes_df['grist_composition'].apply(lambda x: x.get('adjunct_pct', 0))

    print(f"  ✓ Processed {len(malts_df)} malt entries from {malts_df['recipe_id'].nunique()} recipes")

    return recipes_df, malts_df


# ============================================================================
# Step 6: Hop Schedule Extraction
# ============================================================================

def process_hops(recipes_df):
    """Explode and normalize hop data."""
    print("\nStep 6: Processing hops...")

    hops_rows = []

    for idx, row in recipes_df.iterrows():
        hops = parse_json_safe(row['hops_json'], [])

        if not hops:
            continue

        batch_size_gal = row.get('batch_size_gal', 5)
        if pd.isna(batch_size_gal) or batch_size_gal == 0:
            batch_size_gal = 5  # Default to 5 gal

        for hop in hops:
            amount = hop.get('amount', 0)
            unit = hop.get('unit', 'oz')

            # Convert to oz
            if unit == 'g':
                amount_oz = amount / 28.3495
            else:
                amount_oz = amount

            usage = hop.get('usage', '').lower() if hop.get('usage') else ''
            time_min = hop.get('time_min')

            # Classify hop type based on comprehensive keyword-based logic
            # 🟥 Bittering - Early hot-side additions whose primary purpose is IBUs
            # 🟧 Flavour - Mid–late boil additions (still hot-side, still some IBUs)
            # 🟩 Aroma - Late hot-side, post-boil, or cold-side additions

            hop_type = 'other'  # Default

            # 🟥 BITTERING keywords - Early hot-side additions
            bittering_keywords = [
                'first wort', 'fwh', 'mash', 'lauter',
                'start of boil', 'bittering'
            ]

            # 🟧 FLAVOUR keywords - Mid–late boil
            flavour_keywords = [
                'flavor', 'flavour', 'last 10 min', 'last minutes',
                'continuous hopping'
            ]

            # 🟩 AROMA keywords - Late hot-side, post-boil, cold-side
            aroma_keywords = [
                'flameout', 'whirlpool', 'hop stand', 'knockout',
                'hopback', 'steep', 'dry hop', 'dry-hop', 'dryhop',
                'fermentation', 'secondary', 'keg', 'hop oil',
                'aroma', 'post-boil', 'cold side'
            ]

            # Check bittering keywords first
            if any(keyword in usage for keyword in bittering_keywords):
                hop_type = 'bittering'
            # Check aroma keywords (before flavour, as they're more specific)
            elif any(keyword in usage for keyword in aroma_keywords):
                hop_type = 'aroma'
            # Check flavour keywords
            elif any(keyword in usage for keyword in flavour_keywords):
                hop_type = 'flavour'
            # Check boil time if usage is 'boil' or similar
            elif 'boil' in usage and time_min is not None:
                if time_min >= 45:
                    hop_type = 'bittering'
                elif time_min >= 15:
                    hop_type = 'flavour'
                else:
                    hop_type = 'aroma'

            # Calculate hop rates
            oz_per_gal = amount_oz / batch_size_gal
            oz_per_liter = oz_per_gal / 3.78541  # 1 gallon = 3.78541 liters

            hops_rows.append({
                'recipe_id': idx,
                'recipe_title': row['title'],
                'style': row['style'],
                'style_group': row['style_group'],
                'year': row['year'],
                'hop_name': hop.get('name', ''),
                'hop_name_normalized': normalize_ingredient_name(hop.get('name', ''), 'hop'),
                'amount_oz': amount_oz,
                'alpha_acid_pct': hop.get('alpha_acid_pct'),
                'usage': usage,
                'time_min': time_min,
                'hop_type': hop_type,
                'batch_size_gal': batch_size_gal,
                'oz_per_gal': oz_per_gal,
                'oz_per_liter': oz_per_liter,
                'biotransformation': hop.get('biotransformation'),
            })

    hops_df = pd.DataFrame(hops_rows)

    # Calculate hop schedule metrics per recipe
    # Note: 'flavour' uses British spelling to match the classification
    hop_schedule = hops_df.groupby('recipe_id').apply(
        lambda x: {
            'bittering_oz_gal': (x[x['hop_type'] == 'bittering']['oz_per_gal'].sum()) if len(x[x['hop_type'] == 'bittering']) > 0 else 0,
            'flavor_oz_gal': (x[x['hop_type'] == 'flavour']['oz_per_gal'].sum()) if len(x[x['hop_type'] == 'flavour']) > 0 else 0,
            'aroma_oz_gal': (x[x['hop_type'] == 'aroma']['oz_per_gal'].sum()) if len(x[x['hop_type'] == 'aroma']) > 0 else 0,
            'dry_hop_oz_gal': 0,  # Deprecated: dry hop is now part of aroma
        }
    ).to_dict()

    recipes_df['hop_schedule'] = recipes_df.index.map(
        lambda idx: hop_schedule.get(idx, {})
    )

    # Extract to separate columns
    recipes_df['bittering_oz_gal'] = recipes_df['hop_schedule'].apply(lambda x: x.get('bittering_oz_gal', 0))
    recipes_df['flavor_oz_gal'] = recipes_df['hop_schedule'].apply(lambda x: x.get('flavor_oz_gal', 0))
    recipes_df['aroma_oz_gal'] = recipes_df['hop_schedule'].apply(lambda x: x.get('aroma_oz_gal', 0))
    recipes_df['dry_hop_oz_gal'] = recipes_df['hop_schedule'].apply(lambda x: x.get('dry_hop_oz_gal', 0))

    print(f"  ✓ Processed {len(hops_df)} hop entries from {hops_df['recipe_id'].nunique()} recipes")

    return recipes_df, hops_df


# ============================================================================
# Step 7: Water Chemistry Standardization
# ============================================================================

def process_water(recipes_df):
    """Extract and standardize water chemistry."""
    print("\nStep 7: Processing water chemistry...")

    water_rows = []

    for idx, row in recipes_df.iterrows():
        water_data = {
            'recipe_id': idx,
            'recipe_title': row['title'],
            'style': row['style'],
            'year': row['year'],
            'Ca_ppm': row.get('water_Ca_ppm'),
            'Mg_ppm': row.get('water_Mg_ppm'),
            'Na_ppm': row.get('water_Na_ppm'),
            'Cl_ppm': row.get('water_Cl_ppm'),
            'SO4_ppm': row.get('water_SO4_ppm'),
            'HCO3_ppm': row.get('water_HCO3_ppm'),
            'water_description': row.get('water_description'),
        }

        # Try to estimate from salt additions
        water_vol = row.get('water_volume_gal')
        salt_additions = parse_json_safe(row.get('water_salt_additions_json'), [])

        if salt_additions and water_vol and water_vol > 0:
            estimated_ppm = {}
            for salt in salt_additions:
                salt_type = salt.get('salt_type', '')
                amount = salt.get('amount', 0)
                unit = salt.get('unit', 'tsp')

                ppm_dict = estimate_ppm_from_salt(salt_type, amount, unit, water_vol)
                for mineral, ppm in ppm_dict.items():
                    if mineral in estimated_ppm:
                        estimated_ppm[mineral] += ppm
                    else:
                        estimated_ppm[mineral] = ppm

            # Only fill if not already have explicit PPM
            for mineral in ['Ca', 'Mg', 'Na', 'Cl', 'SO4', 'HCO3']:
                key = f'{mineral}_ppm'
                if pd.isna(water_data[key]):
                    water_data[key] = estimated_ppm.get(mineral)

        # Calculate sulfate-chloride ratio
        if water_data['SO4_ppm'] and water_data['Cl_ppm']:
            water_data['SO4_Cl_ratio'] = water_data['SO4_ppm'] / water_data['Cl_ppm']
        else:
            water_data['SO4_Cl_ratio'] = None

        water_rows.append(water_data)

    water_df = pd.DataFrame(water_rows)

    # Map water profiles
    recipes_df = recipes_df.merge(
        water_df[['recipe_id', 'Ca_ppm', 'Mg_ppm', 'Na_ppm', 'Cl_ppm', 'SO4_ppm', 'HCO3_ppm', 'SO4_Cl_ratio']],
        on='recipe_id',
        how='left'
    )

    print(f"  ✓ Processed water chemistry for {len(water_df)} recipes")

    return recipes_df, water_df


# ============================================================================
# Step 8: Yeast Strain Normalization
# ============================================================================

def process_yeast(recipes_df):
    """Normalize and extract yeast data."""
    print("\nStep 8: Processing yeast...")

    yeast_rows = []

    for idx, row in recipes_df.iterrows():
        yeast = parse_json_safe(row['yeast_json'])
        fermentation_stages = parse_json_safe(row.get('fermentation_stages_json', '[]'), [])

        if not yeast:
            continue

        yeast_name = yeast.get('name', '')
        brand = yeast.get('brand')
        product_code = yeast.get('product_code')

        # Normalize yeast
        canonical_name = yeast_name
        if product_code in YEAST_NORMALIZATION:
            brand, canonical_name = YEAST_NORMALIZATION[product_code]

        # Extract fermentation temperatures
        primary_temp = None
        secondary_temp = None
        total_fermentation_days = 0

        for stage in fermentation_stages:
            stage_name = stage.get('stage', '').lower()
            start_temp = stage.get('start_temp_F')
            duration_days = stage.get('duration_days', 0)

            if 'primary' in stage_name and primary_temp is None:
                primary_temp = start_temp
            if 'secondary' in stage_name and secondary_temp is None:
                secondary_temp = start_temp

            if duration_days:
                total_fermentation_days += duration_days

        yeast_rows.append({
            'recipe_id': idx,
            'recipe_title': row['title'],
            'style': row['style'],
            'year': row['year'],
            'yeast_name': yeast_name,
            'yeast_canonical': canonical_name,
            'brand': brand,
            'product_code': product_code,
            'primary_temp_F': primary_temp,
            'secondary_temp_F': secondary_temp,
            'total_fermentation_days': total_fermentation_days,
        })

    yeast_df = pd.DataFrame(yeast_rows)

    print(f"  ✓ Processed yeast for {len(yeast_df)} recipes")
    print(f"    Unique yeast strains: {yeast_df['yeast_canonical'].nunique()}")

    return recipes_df, yeast_df


# ============================================================================
# Step 9: Mash Steps Normalization
# ============================================================================

def process_mash_steps(recipes_df):
    """Extract and normalize mash step data."""
    print("\nStep 9: Processing mash steps...")

    mash_rows = []

    for idx, row in recipes_df.iterrows():
        mash_steps = parse_json_safe(row.get('mash_steps_json', '[]'), [])

        if not mash_steps:
            continue

        for step_num, step in enumerate(mash_steps, start=1):
            mash_rows.append({
                'recipe_id': idx,
                'recipe_title': row['title'],
                'style': row['style'],
                'style_group': row.get('style_group'),
                'year': row['year'],
                'step_number': step_num,
                'step_name': step.get('name', ''),
                'temp_F': step.get('temp_F'),
                'time_min': step.get('time_min'),
            })

    mash_df = pd.DataFrame(mash_rows)

    print(f"  ✓ Processed {len(mash_df)} mash steps from {mash_df['recipe_id'].nunique()} recipes")

    return recipes_df, mash_df


# ============================================================================
# Step 10: Fermentation Stages Normalization
# ============================================================================

def process_fermentation_stages(recipes_df):
    """Extract and normalize fermentation stage data."""
    print("\nStep 10: Processing fermentation stages...")

    fermentation_rows = []

    for idx, row in recipes_df.iterrows():
        fermentation_stages = parse_json_safe(row.get('fermentation_stages_json', '[]'), [])

        if not fermentation_stages:
            continue

        for stage_num, stage in enumerate(fermentation_stages, start=1):
            fermentation_rows.append({
                'recipe_id': idx,
                'recipe_title': row['title'],
                'style': row['style'],
                'style_group': row.get('style_group'),
                'year': row['year'],
                'stage_number': stage_num,
                'stage': stage.get('stage', ''),
                'start_temp_F': stage.get('start_temp_F'),
                'end_temp_F': stage.get('end_temp_F'),
                'duration_days': stage.get('duration_days'),
                'ramp_per_day_F': stage.get('ramp_per_day_F'),
            })

    fermentation_df = pd.DataFrame(fermentation_rows)

    print(f"  ✓ Processed {len(fermentation_df)} fermentation stages from {fermentation_df['recipe_id'].nunique()} recipes")

    return recipes_df, fermentation_df


# ============================================================================
# Step 11: Competition/Award Analysis
# ============================================================================

def process_competition(recipes_df):
    """Extract competition metadata."""
    print("\nStep 9: Processing competition data...")

    # Medal mapping
    medal_map = {
        'NHC GOLD': 'Gold',
        'NHC SILVER': 'Silver',
        'NHC COPPER': 'Bronze',
        'NORMAL MEDAL': 'Medal',
        'CLONE': 'Clone',
        'PRO AM': 'Pro-Am',
        'No Medal': 'No Medal',
    }

    recipes_df['medal_category'] = recipes_df['medal'].map(medal_map).fillna('Unknown')

    medal_counts = recipes_df['medal_category'].value_counts()
    print("  Medal distribution:")
    for medal, count in medal_counts.items():
        pct = (count / len(recipes_df)) * 100
        print(f"    {medal}: {count} ({pct:.1f}%)")

    return recipes_df




# ============================================================================
# BJCP Style Guidelines Reference
# ============================================================================

def create_bjcp_reference():
    """Create BJCP style guidelines reference table."""
    print("\nCreating BJCP style guidelines reference...")

    # BJCP 2021 Guidelines - key styles mapped to our style groups
    bjcp_data = [
        # IPA styles
        {'style_name': 'American IPA', 'style_group': 'IPA', 'og_min': 1.056, 'og_max': 1.070, 'fg_min': 1.008, 'fg_max': 1.014, 'ibu_min': 40, 'ibu_max': 70, 'srm_min': 6, 'srm_max': 14, 'abv_min': 5.5, 'abv_max': 7.5},
        {'style_name': 'Double IPA', 'style_group': 'IPA', 'og_min': 1.065, 'og_max': 1.085, 'fg_min': 1.008, 'fg_max': 1.018, 'ibu_min': 60, 'ibu_max': 100, 'srm_min': 6, 'srm_max': 14, 'abv_min': 7.5, 'abv_max': 10.0},
        {'style_name': 'New England IPA', 'style_group': 'IPA', 'og_min': 1.060, 'og_max': 1.085, 'fg_min': 1.010, 'fg_max': 1.020, 'ibu_min': 25, 'ibu_max': 60, 'srm_min': 3, 'srm_max': 7, 'abv_min': 6.0, 'abv_max': 9.0},
        {'style_name': 'American Pale Ale', 'style_group': 'IPA', 'og_min': 1.045, 'og_max': 1.060, 'fg_min': 1.010, 'fg_max': 1.015, 'ibu_min': 30, 'ibu_max': 50, 'srm_min': 5, 'srm_max': 10, 'abv_min': 4.5, 'abv_max': 6.2},

        # Lager styles
        {'style_name': 'German Pilsner', 'style_group': 'LAGER', 'og_min': 1.044, 'og_max': 1.050, 'fg_min': 1.008, 'fg_max': 1.013, 'ibu_min': 22, 'ibu_max': 40, 'srm_min': 2, 'srm_max': 5, 'abv_min': 4.4, 'abv_max': 5.2},
        {'style_name': 'Czech Premium Pale Lager', 'style_group': 'LAGER', 'og_min': 1.044, 'og_max': 1.060, 'fg_min': 1.013, 'fg_max': 1.017, 'ibu_min': 30, 'ibu_max': 45, 'srm_min': 3.5, 'srm_max': 6, 'abv_min': 4.2, 'abv_max': 5.8},
        {'style_name': 'Munich Helles', 'style_group': 'LAGER', 'og_min': 1.044, 'og_max': 1.048, 'fg_min': 1.006, 'fg_max': 1.012, 'ibu_min': 16, 'ibu_max': 22, 'srm_min': 3, 'srm_max': 5, 'abv_min': 4.7, 'abv_max': 5.4},
        {'style_name': 'Märzen', 'style_group': 'LAGER', 'og_min': 1.054, 'og_max': 1.060, 'fg_min': 1.010, 'fg_max': 1.014, 'ibu_min': 18, 'ibu_max': 24, 'srm_min': 8, 'srm_max': 17, 'abv_min': 5.6, 'abv_max': 6.3},

        # Porter/Stout styles
        {'style_name': 'American Porter', 'style_group': 'PORTER_STOUT', 'og_min': 1.050, 'og_max': 1.070, 'fg_min': 1.012, 'fg_max': 1.018, 'ibu_min': 25, 'ibu_max': 50, 'srm_min': 22, 'srm_max': 40, 'abv_min': 4.8, 'abv_max': 6.5},
        {'style_name': 'Irish Stout', 'style_group': 'PORTER_STOUT', 'og_min': 1.036, 'og_max': 1.044, 'fg_min': 1.007, 'fg_max': 1.011, 'ibu_min': 25, 'ibu_max': 45, 'srm_min': 25, 'srm_max': 40, 'abv_min': 4.0, 'abv_max': 4.5},
        {'style_name': 'Imperial Stout', 'style_group': 'PORTER_STOUT', 'og_min': 1.075, 'og_max': 1.115, 'fg_min': 1.018, 'fg_max': 1.030, 'ibu_min': 50, 'ibu_max': 90, 'srm_min': 30, 'srm_max': 40, 'abv_min': 8.0, 'abv_max': 12.0},

        # Belgian styles
        {'style_name': 'Belgian Tripel', 'style_group': 'BELGIAN', 'og_min': 1.075, 'og_max': 1.085, 'fg_min': 1.008, 'fg_max': 1.014, 'ibu_min': 20, 'ibu_max': 40, 'srm_min': 4.5, 'srm_max': 7, 'abv_min': 7.5, 'abv_max': 9.5},
        {'style_name': 'Belgian Dubbel', 'style_group': 'BELGIAN', 'og_min': 1.062, 'og_max': 1.075, 'fg_min': 1.008, 'fg_max': 1.018, 'ibu_min': 15, 'ibu_max': 25, 'srm_min': 10, 'srm_max': 17, 'abv_min': 6.0, 'abv_max': 7.6},
        {'style_name': 'Saison', 'style_group': 'BELGIAN', 'og_min': 1.048, 'og_max': 1.065, 'fg_min': 1.002, 'fg_max': 1.008, 'ibu_min': 20, 'ibu_max': 35, 'srm_min': 5, 'srm_max': 14, 'abv_min': 5.0, 'abv_max': 7.0},
        {'style_name': 'Belgian Dark Strong Ale', 'style_group': 'BELGIAN', 'og_min': 1.075, 'og_max': 1.110, 'fg_min': 1.010, 'fg_max': 1.024, 'ibu_min': 20, 'ibu_max': 35, 'srm_min': 12, 'srm_max': 22, 'abv_min': 8.0, 'abv_max': 12.0},

        # Strong Ale styles
        {'style_name': 'English Barleywine', 'style_group': 'STRONG_ALE', 'og_min': 1.080, 'og_max': 1.120, 'fg_min': 1.018, 'fg_max': 1.030, 'ibu_min': 35, 'ibu_max': 70, 'srm_min': 8, 'srm_max': 22, 'abv_min': 8.0, 'abv_max': 12.0},
        {'style_name': 'American Barleywine', 'style_group': 'STRONG_ALE', 'og_min': 1.080, 'og_max': 1.120, 'fg_min': 1.016, 'fg_max': 1.030, 'ibu_min': 50, 'ibu_max': 100, 'srm_min': 9, 'srm_max': 18, 'abv_min': 8.0, 'abv_max': 12.0},
        {'style_name': 'Wee Heavy', 'style_group': 'STRONG_ALE', 'og_min': 1.070, 'og_max': 1.130, 'fg_min': 1.018, 'fg_max': 1.040, 'ibu_min': 17, 'ibu_max': 35, 'srm_min': 14, 'srm_max': 25, 'abv_min': 6.5, 'abv_max': 10.0},

        # Amber/Red Ale styles
        {'style_name': 'American Amber Ale', 'style_group': 'AMBER_ALE', 'og_min': 1.045, 'og_max': 1.060, 'fg_min': 1.010, 'fg_max': 1.015, 'ibu_min': 25, 'ibu_max': 40, 'srm_min': 10, 'srm_max': 17, 'abv_min': 4.5, 'abv_max': 6.2},
        {'style_name': 'Irish Red Ale', 'style_group': 'AMBER_ALE', 'og_min': 1.036, 'og_max': 1.046, 'fg_min': 1.010, 'fg_max': 1.014, 'ibu_min': 18, 'ibu_max': 28, 'srm_min': 9, 'srm_max': 14, 'abv_min': 3.8, 'abv_max': 5.0},

        # Wheat Beer styles
        {'style_name': 'Weissbier', 'style_group': 'WHEAT_BEER', 'og_min': 1.044, 'og_max': 1.053, 'fg_min': 1.008, 'fg_max': 1.014, 'ibu_min': 8, 'ibu_max': 15, 'srm_min': 2, 'srm_max': 6, 'abv_min': 4.3, 'abv_max': 5.6},
        {'style_name': 'Witbier', 'style_group': 'WHEAT_BEER', 'og_min': 1.044, 'og_max': 1.052, 'fg_min': 1.008, 'fg_max': 1.012, 'ibu_min': 8, 'ibu_max': 20, 'srm_min': 2, 'srm_max': 4, 'abv_min': 4.5, 'abv_max': 5.5},
        {'style_name': 'American Wheat Beer', 'style_group': 'WHEAT_BEER', 'og_min': 1.040, 'og_max': 1.055, 'fg_min': 1.008, 'fg_max': 1.013, 'ibu_min': 15, 'ibu_max': 30, 'srm_min': 3, 'srm_max': 6, 'abv_min': 4.0, 'abv_max': 5.5},

        # Brown Ale styles
        {'style_name': 'American Brown Ale', 'style_group': 'BROWN_ALE', 'og_min': 1.045, 'og_max': 1.060, 'fg_min': 1.010, 'fg_max': 1.016, 'ibu_min': 20, 'ibu_max': 30, 'srm_min': 18, 'srm_max': 35, 'abv_min': 4.3, 'abv_max': 6.2},
        {'style_name': 'English Brown Ale', 'style_group': 'BROWN_ALE', 'og_min': 1.040, 'og_max': 1.052, 'fg_min': 1.008, 'fg_max': 1.013, 'ibu_min': 20, 'ibu_max': 30, 'srm_min': 12, 'srm_max': 22, 'abv_min': 4.2, 'abv_max': 5.9},

        # Sour styles
        {'style_name': 'Berliner Weisse', 'style_group': 'SOUR', 'og_min': 1.028, 'og_max': 1.032, 'fg_min': 1.003, 'fg_max': 1.006, 'ibu_min': 3, 'ibu_max': 8, 'srm_min': 2, 'srm_max': 3, 'abv_min': 2.8, 'abv_max': 3.8},
        {'style_name': 'Gose', 'style_group': 'SOUR', 'og_min': 1.036, 'og_max': 1.056, 'fg_min': 1.006, 'fg_max': 1.010, 'ibu_min': 5, 'ibu_max': 12, 'srm_min': 3, 'srm_max': 4, 'abv_min': 4.2, 'abv_max': 4.8},
        {'style_name': 'Flanders Red Ale', 'style_group': 'SOUR', 'og_min': 1.048, 'og_max': 1.057, 'fg_min': 1.002, 'fg_max': 1.012, 'ibu_min': 10, 'ibu_max': 25, 'srm_min': 10, 'srm_max': 17, 'abv_min': 4.6, 'abv_max': 6.5},

        # Blonde/Pale styles
        {'style_name': 'Blonde Ale', 'style_group': 'BLONDE_PALE', 'og_min': 1.038, 'og_max': 1.054, 'fg_min': 1.008, 'fg_max': 1.013, 'ibu_min': 15, 'ibu_max': 28, 'srm_min': 3, 'srm_max': 6, 'abv_min': 3.8, 'abv_max': 5.5},
        {'style_name': 'Cream Ale', 'style_group': 'BLONDE_PALE', 'og_min': 1.042, 'og_max': 1.055, 'fg_min': 1.006, 'fg_max': 1.012, 'ibu_min': 8, 'ibu_max': 20, 'srm_min': 2, 'srm_max': 5, 'abv_min': 4.2, 'abv_max': 5.6},
    ]

    bjcp_df = pd.DataFrame(bjcp_data)
    print(f"  ✓ Created BJCP reference with {len(bjcp_df)} style guidelines")

    return bjcp_df




# ============================================================================
# Main Preprocessing Pipeline
# ============================================================================

def main():
    print("="*80)
    print("CRAFT BEER RECIPE DATA PREPROCESSING PIPELINE")
    print("="*80)

    # Step 1-2: Load and validate
    recipes_df = load_and_validate_data(INPUT_FILE)

    # Step 3: Style categorization
    recipes_df = process_styles(recipes_df)

    # Step 4-5: Malt composition
    recipes_df, malts_df = process_malts(recipes_df)

    # Step 6: Hop schedule
    recipes_df, hops_df = process_hops(recipes_df)

    # Step 7: Water chemistry
    recipes_df, water_df = process_water(recipes_df)

    # Step 8: Yeast normalization
    recipes_df, yeast_df = process_yeast(recipes_df)

    # Step 9: Mash steps normalization
    recipes_df, mash_steps_df = process_mash_steps(recipes_df)

    # Step 10: Fermentation stages normalization
    recipes_df, fermentation_df = process_fermentation_stages(recipes_df)

    # Step 11: Competition/awards
    recipes_df = process_competition(recipes_df)

    # BJCP Reference data
    bjcp_df = create_bjcp_reference()

    # Create recipe lookup table (recipe_id -> url as PK)
    print("\nCreating recipe lookup table...")
    recipe_lookup_df = recipes_df[['recipe_id', 'title', 'url']].copy()
    print(f"  ✓ Created recipe lookup with {len(recipe_lookup_df)} entries")

    # =====================================================================
    # Save all normalized tables
    # =====================================================================
    print("\n" + "="*80)
    print("SAVING NORMALIZED TABLES")
    print("="*80)

    # Main processed tables
    tables = [
        ('recipes_normalized.csv', recipes_df),
        ('malts_normalized.csv', malts_df),
        ('hops_normalized.csv', hops_df),
        ('water_normalized.csv', water_df),
        ('yeast_normalized.csv', yeast_df),
        ('mash_steps_normalized.csv', mash_steps_df),
        ('fermentation_stages_normalized.csv', fermentation_df),
    ]

    for filename, df in tables:
        output_path = OUTPUT_DIR / filename
        df.to_csv(output_path, index=False)
        print(f"  ✓ Saved {filename} ({len(df)} rows)")

    # Save reference data
    bjcp_path = REFERENCE_DIR / 'bjcp_guidelines.csv'
    bjcp_df.to_csv(bjcp_path, index=False)
    print(f"  ✓ Saved reference/bjcp_guidelines.csv ({len(bjcp_df)} rows)")

    # Save recipe lookup table
    lookup_path = REFERENCE_DIR / 'recipe_lookup.csv'
    recipe_lookup_df.to_csv(lookup_path, index=False)
    print(f"  ✓ Saved reference/recipe_lookup.csv ({len(recipe_lookup_df)} rows)")

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Reference directory: {REFERENCE_DIR}")
    print("\nGenerated tables:")
    print("  • recipes_normalized.csv - Main normalized recipe data")
    print("  • malts_normalized.csv - Malt ingredient details")
    print("  • hops_normalized.csv - Hop ingredient details (with hop_type classification)")
    print("  • water_normalized.csv - Water chemistry profiles")
    print("  • yeast_normalized.csv - Yeast strain details")
    print("  • mash_steps_normalized.csv - Mash step details (temperature profiles)")
    print("  • fermentation_stages_normalized.csv - Fermentation stage details (temp schedules)")
    print("\nReference tables:")
    print("  • bjcp_guidelines.csv - BJCP style guidelines")
    print("  • recipe_lookup.csv - Recipe ID to URL mapping (PK)")


if __name__ == '__main__':
    main()
