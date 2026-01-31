import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from scipy import stats

# Import reusable visualization functions
from visualizations import (
    plot_bell_curve,
    plot_scatter_trend,
    plot_donut,
    plot_malt_kde,
    plot_hop_rate_kde,
    plot_late_hop_usage,
    plot_violin,
    plot_box_ions,
    get_stats_summary
)

# Page configuration
st.set_page_config(
    page_title="Craft Beer Recipe Dashboard",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #D4A574;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Smaller sidebar filter text */
    section[data-testid="stSidebar"] * {
        font-size: 12px !important;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 18px !important;
    }
    section[data-testid="stSidebar"] .stSlider label {
        font-size: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    """Load all normalized datasets"""
    base_path = Path("dataset/processed")

    recipes = pd.read_csv(base_path / "recipes_normalized.csv")
    malts = pd.read_csv(base_path / "malts_normalized.csv")
    hops = pd.read_csv(base_path / "hops_normalized.csv")
    water = pd.read_csv(base_path / "water_normalized.csv")
    yeast = pd.read_csv(base_path / "yeast_normalized.csv")

    return recipes, malts, hops, water, yeast

# Load data
try:
    recipes_df, malts_df, hops_df, water_df, yeast_df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Header
st.markdown('<div class="main-header">🍺 Craft Beer Recipe Dashboard</div>', unsafe_allow_html=True)
st.markdown("### Analyzing 1,400+ Award-Winning Homebrew Recipes")
st.markdown("---")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Start with full dataset for cascading filters
temp_df = recipes_df.copy()

# Helper function for case-insensitive sorting
def case_insensitive_sort(items):
    return sorted(items, key=str.lower)

# 1. Type of Drinks (Multi-select with All option)
st.sidebar.markdown("<small><b>Type of Drinks</b></small>", unsafe_allow_html=True)
final_category_options = case_insensitive_sort(temp_df['final_category'].dropna().unique().tolist())
final_category_with_all = ["All"] + final_category_options
selected_type_drink = st.sidebar.multiselect(
    label="",
    options=final_category_with_all,
    default=["All"],
    key="type_drinks_filter"
)
# Handle "All" selection
if "All" not in selected_type_drink:
    temp_df = temp_df[temp_df['final_category'].isin(selected_type_drink)]

# 2. Medal (Multi-select with All option) - cascaded
st.sidebar.markdown("<small><b>Medal Category</b></small>", unsafe_allow_html=True)
medal_options = case_insensitive_sort(temp_df['medal'].dropna().unique().tolist())
medal_with_all = ["All"] + medal_options
selected_medal = st.sidebar.multiselect(
    label="",
    options=medal_with_all,
    default=["All"],
    key="medal_filter"
)
# Handle "All" selection
if "All" not in selected_medal:
    temp_df = temp_df[temp_df['medal'].isin(selected_medal)]

# 3. Parent Style (Multi-select with All option) - cascaded
st.sidebar.markdown("<small><b>Parent Style Group</b></small>", unsafe_allow_html=True)
parent_style_options = case_insensitive_sort(temp_df['parent_style'].dropna().unique().tolist())
parent_style_with_all = ["All"] + parent_style_options
selected_parent = st.sidebar.multiselect(
    label="",
    options=parent_style_with_all,
    default=["All"],
    key="parent_style_filter"
)
# Handle "All" selection
if "All" not in selected_parent:
    temp_df = temp_df[temp_df['parent_style'].isin(selected_parent)]

# 4. Style Group (Multi-select with All option) - cascaded
st.sidebar.markdown("<small><b>Beer Style Group</b></small>", unsafe_allow_html=True)
style_group_options = case_insensitive_sort(temp_df['style_group'].dropna().unique().tolist())
style_group_with_all = ["All"] + style_group_options
selected_group = st.sidebar.multiselect(
    label="",
    options=style_group_with_all,
    default=["All"],
    key="style_group_filter"
)
# Handle "All" selection
if "All" not in selected_group:
    temp_df = temp_df[temp_df['style_group'].isin(selected_group)]

# 5. Specific Style (Multi-select with All option) - cascaded
st.sidebar.markdown("<small><b>Beer Style</b></small>", unsafe_allow_html=True)
style_options = case_insensitive_sort(temp_df['style'].dropna().unique().tolist())
style_with_all = ["All"] + style_options
selected_style = st.sidebar.multiselect(
    label="",
    options=style_with_all,
    default=["All"],
    key="style_filter"
)
# Handle "All" selection
if "All" not in selected_style:
    temp_df = temp_df[temp_df['style'].isin(selected_style)]

# ABV range filter
abv_min, abv_max = st.sidebar.slider(
    "ABV Range (%)",
    min_value=0.0,
    max_value=20.0,
    value=(0.0, 20.0),
    step=0.5
)

# Year range filter
year_min = int(recipes_df['year'].min()) if not recipes_df['year'].isna().all() else 1974
year_max = int(recipes_df['year'].max()) if not recipes_df['year'].isna().all() else 2024
selected_years = st.sidebar.slider(
    "Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

# Apply final filters (ABV and Year) - handle NaN values
filtered_df = temp_df[
    (temp_df['abv_pct'].fillna(0).between(abv_min, abv_max)) &
    (temp_df['year'].fillna(year_min).between(selected_years[0], selected_years[1]))
]

# Key metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Recipes", f"{len(filtered_df):,}")

with col2:
    avg_abv = filtered_df['abv_pct'].mean() if len(filtered_df) > 0 else 0
    st.metric("Avg ABV", f"{avg_abv:.1f}%" if pd.notna(avg_abv) else "N/A")

with col3:
    avg_ibu = filtered_df['ibu'].mean() if len(filtered_df) > 0 else 0
    st.metric("Avg IBU", f"{avg_ibu:.0f}" if pd.notna(avg_ibu) else "N/A")

with col4:
    avg_og = filtered_df['og'].mean() if len(filtered_df) > 0 else 0
    st.metric("Avg OG", f"{avg_og:.3f}" if pd.notna(avg_og) else "N/A")

with col5:
    gold_medals = len(filtered_df[filtered_df['medal'] == 'Gold']) if len(filtered_df) > 0 else 0
    st.metric("Gold Medals", f"{gold_medals:,}")

st.markdown("---")

# Tab navigation
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Recipe Distribution",
    "📊 Profile Overview",
    "🌾 Malts & Grains",
    "🌿 Hops",
    "🧬 Yeast",
    "🌊 Water",
    "🔍 Mash & Boil & Ferment",
])

# TAB 1: Overview
with tab0:
    st.subheader("📊 Recipe Distribution Insights")

    if len(filtered_df) == 0:
        st.warning("No recipes match the current filters. Please adjust your filter settings.")
    else:
        # Row 1: Medal Distribution
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Distribution by Medal")
            medal_counts = filtered_df['medal'].value_counts()

            if len(medal_counts) > 0:
                # Define custom colors for medals
                medal_colors = {
                    'NHC GOLD': '#FFD700',       # Gold
                    'NHC SILVER': '#C0C0C0',     # Silver
                    'NHC COPPER': '#CD7F32',     # Bronze/Copper
                    'No Medal': "#2C3E61",       # Cornflower blue
                    'CLONE': "#483669",          # Medium purple
                    'PRO AM': "#105854",         # Light sea green
                    'NORMAL MEDAL': "#5B2319"    # Tomato red
                }

                # Create color list based on medal names
                colors = [medal_colors.get(medal, '#A9A9A9') for medal in medal_counts.index]

                # Use go.Pie for better color control
                fig = go.Figure(data=[go.Pie(
                    labels=medal_counts.index,
                    values=medal_counts.values,
                    hole=0.3,
                    marker=dict(colors=colors),
                    textposition='inside',
                    textinfo='percent+label+value'
                )])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No medal data available.")

        with col2:
            # Specific Style (Top 15)
            st.markdown("#### Distribution by Specific Beer Style (Top 15)")
            style_counts = filtered_df['style'].value_counts().head(15)
            if len(style_counts) > 0:
                fig = px.bar(
                    x=style_counts.values,
                    y=style_counts.index,
                    orientation='h',
                    labels={'x': 'Number of Recipes', 'y': 'Beer Style'},
                    color=style_counts.values,
                    color_continuous_scale='YlOrBr',
                    text=style_counts.values
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No style data available.")

        # Time series
        st.subheader("Recipe Trends Over Time")
        yearly_data = filtered_df.dropna(subset=['year'])
        if len(yearly_data) > 0:
            yearly_counts = yearly_data.groupby('year').size().reset_index(name='count')
            fig = px.line(
                yearly_counts,
                x='year',
                y='count',
                markers=True,
                labels={'year': 'Year', 'count': 'Number of Recipes'}
            )
            fig.update_traces(line_color="#D10000", marker=dict(size=8))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No year data available for trend analysis.")

        st.markdown("---")


with tab1:
    # Bell Curve Distributions Section
    st.subheader("Beer Characteristics Distributions")
    st.markdown("*Statistical distributions of key beer parameters with mean, median, and standard deviation*")

    # Define all parameters to display
    parameters = [
        ("abv_pct", "ABV (%)"),
        ("og", "Original Gravity (OG)"),
        ("fg", "Final Gravity (FG)"),
        ("ibu", "IBU (Bitterness)"),
        ("srm", "SRM (Color)")
    ]

    # Display parameters in rows of 2 (distribution + trend side by side)
    for param_col, param_label in parameters:
        if param_col in filtered_df.columns:
            st.markdown(f"#### {param_label}")
            col_dist, col_trend = st.columns(2)

            with col_dist:
                # Bell curve distribution
                bell_data = filtered_df[param_col].dropna().values
                if len(bell_data) > 0:
                    fig_bell = plot_bell_curve(
                        bell_data,
                        title=f"{param_label} Distribution",
                        height=350
                    )
                    st.plotly_chart(fig_bell, use_container_width=True)
                else:
                    st.info(f"No {param_label} data available.")

            with col_trend:
                # Scatter trend over time
                trend_data = filtered_df.dropna(subset=['year', param_col])
                if len(trend_data) > 2:
                    fig_scatter, scatter_stats = plot_scatter_trend(
                        trend_data,
                        x_col='year',
                        y_col=param_col,
                        title=f"{param_label} Trend Over Time",
                        x_label="Year",
                        y_label=param_label,
                        height=350
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Not enough data for trend analysis.")

            st.markdown("---")

# TAB 2: Malts & Grains
with tab2:
    st.subheader("Malt Analysis")

    # Filter malts by selected recipes
    filtered_recipe_ids = filtered_df['recipe_id'].unique()
    filtered_malts = malts_df[malts_df['recipe_id'].isin(filtered_recipe_ids)]

    # Calculate malt type aggregations
    if len(filtered_malts) > 0 and 'malt_type' in filtered_malts.columns and 'pct_of_grist' in filtered_malts.columns:
        # Sum by recipe and malt type first
        df_sum_malt_type = filtered_malts.groupby(['recipe_id', 'malt_type'], as_index=False)['pct_of_grist'].sum()

        # Average across recipes
        avg_malt_pct = df_sum_malt_type.groupby('malt_type', as_index=False)['pct_of_grist'].mean()
        avg_malt_pct['pct'] = avg_malt_pct['pct_of_grist'] / avg_malt_pct['pct_of_grist'].sum() * 100

        # Malt type selector
        malt_types = ['All Types'] + sorted(filtered_malts['malt_type'].dropna().unique().tolist())
        selected_malt_type = st.selectbox(
            "Select Malt Type to Analyze:",
            malt_types,
            key="malt_type_selector"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Average Grist Composition")
            if selected_malt_type == 'All Types':
                # Overall malt type distribution
                fig_donut = plot_donut(
                    data=avg_malt_pct,
                    names_col='malt_type',
                    values_col='pct_of_grist',
                    title='Malt Types Distribution',
                    height=400
                )
            else:
                # Specific malt names within selected type
                malt_names = filtered_malts[filtered_malts['malt_type'] == selected_malt_type]
                if 'malt_name_normalized' in malt_names.columns:
                    name_col = 'malt_name_normalized'
                elif 'malt_name' in malt_names.columns:
                    name_col = 'malt_name'
                else:
                    name_col = 'malt'

                malt_name_avg = malt_names.groupby(name_col, as_index=False)['pct_of_grist'].mean()
                malt_name_avg = malt_name_avg.nlargest(10, 'pct_of_grist')  # Top 10

                fig_donut = plot_donut(
                    data=malt_name_avg,
                    names_col=name_col,
                    values_col='pct_of_grist',
                    title=f'Top {selected_malt_type.title()} Malts',
                    height=400
                )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col2:
            st.markdown("#### % of Grist Distribution (KDE)")
            if selected_malt_type == 'All Types':
                # KDE by malt type
                fig_kde = plot_malt_kde(
                    df_sum_malt_type,
                    value_col='pct_of_grist',
                    type_col='malt_type',
                    title='Malt Type Usage Distribution',
                    height=400
                )
            else:
                # KDE for specific malt type
                type_data = df_sum_malt_type[df_sum_malt_type['malt_type'] == selected_malt_type]
                if len(type_data) >= 10:
                    fig_kde = plot_malt_kde(
                        type_data,
                        value_col='pct_of_grist',
                        type_col='malt_type',
                        title=f'{selected_malt_type.title()} Usage Distribution',
                        min_points=5,
                        height=400
                    )
                else:
                    fig_kde = go.Figure()
                    fig_kde.add_annotation(
                        text="Not enough data for KDE",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    fig_kde.update_layout(height=400)
            st.plotly_chart(fig_kde, use_container_width=True)

        with col3:
            st.markdown("#### Trend Over Time")
            if selected_malt_type == 'All Types':
                # Select a malt type for trend analysis
                trend_type = st.selectbox(
                    "Malt type for trend:",
                    sorted(avg_malt_pct['malt_type'].tolist()),
                    key="malt_trend_type"
                )
            else:
                trend_type = selected_malt_type

            # Calculate yearly average for selected malt type
            yearly_malt = df_sum_malt_type[df_sum_malt_type['malt_type'] == trend_type]
            if len(yearly_malt) > 0:
                yearly_malt = yearly_malt.merge(
                    filtered_df[['recipe_id', 'year']].drop_duplicates(),
                    on='recipe_id',
                    how='left'
                )
                # Drop rows with missing year
                yearly_malt = yearly_malt.dropna(subset=['year'])

                if len(yearly_malt) > 0:
                    yearly_avg = yearly_malt.groupby('year', as_index=False)['pct_of_grist'].mean()

                    if len(yearly_avg) > 2:
                        fig_scatter, stats = plot_scatter_trend(
                            yearly_avg,
                            x_col='year',
                            y_col='pct_of_grist',
                            title=f'{trend_type.title()} Malt % Over Time',
                            x_label='Year',
                            y_label='% of Grist',
                            height=400
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.info("Not enough data points for trend analysis.")
                else:
                    st.info("No year data available for trend analysis.")
            else:
                st.info("No data available for this malt type.")
    else:
        st.info("No malt data available for the selected filters.")


# TAB 3: Hops
with tab3:
    st.subheader("Hop Analysis")

    # Filter hops by selected recipes
    filtered_recipe_ids = filtered_df['recipe_id'].unique()
    filtered_hops = hops_df[hops_df['recipe_id'].isin(filtered_recipe_ids)]

    if len(filtered_hops) > 0:
        # Determine column names
        hop_name_col = 'hop_name_normalized' if 'hop_name_normalized' in filtered_hops.columns else 'hop_name' if 'hop_name' in filtered_hops.columns else 'hop'
        hop_type_col = 'hop_type' if 'hop_type' in filtered_hops.columns else 'usage'
        rate_col = 'oz_per_gal' if 'oz_per_gal' in filtered_hops.columns else None

        # Get available hop types
        hop_types = ['All Types'] + sorted(filtered_hops[hop_type_col].dropna().unique().tolist())
        selected_hop_type = st.selectbox(
            "Select Hop Usage Type:",
            hop_types,
            key="hop_type_selector"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Top Hops Used")
            if selected_hop_type == 'All Types':
                # Overall hop usage by type
                hop_type_counts = filtered_hops[hop_type_col].value_counts().reset_index()
                hop_type_counts.columns = [hop_type_col, 'count']
                fig_donut = plot_donut(
                    data=hop_type_counts,
                    names_col=hop_type_col,
                    values_col='count',
                    title='Hop Usage by Type',
                    height=400
                )
            else:
                # Top hops for selected type
                type_hops = filtered_hops[filtered_hops[hop_type_col] == selected_hop_type]
                hop_counts = type_hops[hop_name_col].value_counts().head(10).reset_index()
                hop_counts.columns = [hop_name_col, 'count']
                fig_donut = plot_donut(
                    data=hop_counts,
                    names_col=hop_name_col,
                    values_col='count',
                    title=f'Top {selected_hop_type.title()} Hops',
                    height=400
                )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col2:
            st.markdown("#### Hop Rate Distribution (KDE)")
            if rate_col and rate_col in filtered_hops.columns:
                if selected_hop_type == 'All Types':
                    # KDE for all hop types
                    stages = filtered_hops[hop_type_col].dropna().unique().tolist()
                    fig_kde = plot_hop_rate_kde(
                        filtered_hops,
                        rate_col=rate_col,
                        stage_col=hop_type_col,
                        stages=stages[:4],  # Limit to 4 stages for readability
                        title='Hop Rate by Usage Type',
                        height=400
                    )
                else:
                    # KDE for selected hop type only
                    fig_kde = plot_hop_rate_kde(
                        filtered_hops,
                        rate_col=rate_col,
                        stage_col=hop_type_col,
                        stages=[selected_hop_type],
                        title=f'{selected_hop_type.title()} Hop Rate Distribution',
                        height=400
                    )
                st.plotly_chart(fig_kde, use_container_width=True)
            else:
                st.info("Hop rate data not available.")

        with col3:
            st.markdown("#### Usage Trend Over Time")
            # Calculate yearly hop usage percentage
            hops_with_year = filtered_hops.merge(
                filtered_df[['recipe_id', 'year']].drop_duplicates(),
                on='recipe_id',
                how='left'
            )

            # Check if merge was successful and year column exists
            if 'year' not in hops_with_year.columns or hops_with_year['year'].isna().all():
                st.info("No year data available for trend analysis.")
            elif selected_hop_type == 'All Types':
                # Drop rows with missing year
                hops_with_year = hops_with_year.dropna(subset=['year'])

                if len(hops_with_year) > 0:
                    # Calculate usage by hop type over time
                    used = hops_with_year.groupby(['year', 'recipe_id', hop_type_col]).size().reset_index(name='n')
                    total_recipes = hops_with_year.groupby('year')['recipe_id'].nunique().rename('total')
                    stage_counts = used.groupby(['year', hop_type_col])['recipe_id'].nunique().rename('count').reset_index()
                    stage_counts = stage_counts.merge(total_recipes, on='year')
                    stage_counts['pct'] = stage_counts['count'] / stage_counts['total'] * 100

                    # Plot late hop usage trends
                    late_stages = [s for s in ['flavour', 'aroma', 'dry_hop'] if s in stage_counts[hop_type_col].values]
                    if late_stages:
                        fig_trend = plot_late_hop_usage(
                            stage_counts,
                            stages=late_stages,
                            show_loess=False,
                            title='Late Hop Usage Over Time',
                            height=400
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.info("No late hop data available for trend analysis.")
                else:
                    st.info("No hop data with year information available.")
            else:
                # Trend for specific hop type
                hops_with_year = hops_with_year.dropna(subset=['year'])

                if rate_col and rate_col in hops_with_year.columns and len(hops_with_year) > 0:
                    yearly_rate = hops_with_year[hops_with_year[hop_type_col] == selected_hop_type].groupby('year', as_index=False)[rate_col].mean()
                    if len(yearly_rate) > 2:
                        fig_scatter, stats = plot_scatter_trend(
                            yearly_rate,
                            x_col='year',
                            y_col=rate_col,
                            title=f'{selected_hop_type.title()} Rate Over Time',
                            x_label='Year',
                            y_label='oz/gallon',
                            height=400
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.info("Not enough data points for trend analysis.")
                else:
                    st.info("Hop rate trend data not available.")
    else:
        st.info("No hop data available for the selected filters.")

# TAB 4: Yeast
with tab4:
    st.subheader("Yeast Analysis")

    # Filter yeast by selected recipes
    filtered_recipe_ids = filtered_df['recipe_id'].unique()
    filtered_yeast = yeast_df[yeast_df['recipe_id'].isin(filtered_recipe_ids)]

    if len(filtered_yeast) > 0:
        # Determine yeast column name
        yeast_col = 'yeast_canonical' if 'yeast_canonical' in filtered_yeast.columns else 'yeast_name'

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top Yeast Strains")
            yeast_counts = filtered_yeast[yeast_col].value_counts().head(15).reset_index()
            yeast_counts.columns = [yeast_col, 'count']

            fig_donut = plot_donut(
                data=yeast_counts,
                names_col=yeast_col,
                values_col='count',
                title='Yeast Strain Distribution',
                height=450
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col2:
            st.markdown("#### Yeast Usage Trend Over Time")
            # Merge with recipe year data
            yeast_with_year = filtered_yeast.merge(
                filtered_df[['recipe_id', 'year']].drop_duplicates(),
                on='recipe_id',
                how='left'
            )

            # Get top 5 yeasts for trend analysis
            top_yeasts = filtered_yeast[yeast_col].value_counts().head(5).index.tolist()

            # Check if year column exists in merged data
            if 'year' not in yeast_with_year.columns or yeast_with_year['year'].isna().all():
                st.info("No year data available for trend analysis.")
            elif len(top_yeasts) > 0:
                selected_yeast = st.selectbox(
                    "Select yeast strain for trend:",
                    top_yeasts,
                    key="yeast_trend_selector"
                )

                # Calculate yearly usage count for selected yeast
                yeast_yearly = yeast_with_year[yeast_with_year[yeast_col] == selected_yeast]
                if 'year' in yeast_yearly.columns:
                    yeast_yearly = yeast_yearly.dropna(subset=['year'])
                else:
                    yeast_yearly = pd.DataFrame()

                if len(yeast_yearly) > 0:
                    yearly_counts = yeast_yearly.groupby('year').size().reset_index(name='count')

                    if len(yearly_counts) > 3:
                        fig_scatter, stats = plot_scatter_trend(
                            yearly_counts,
                            x_col='year',
                            y_col='count',
                            title=f'{selected_yeast} Usage Over Time',
                            x_label='Year',
                            y_label='Number of Recipes',
                            height=400
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.info("Not enough data points for trend analysis.")
                else:
                    st.info("No year data available for trend analysis.")
            else:
                st.info("No yeast strains available for trend analysis.")

        # Fermentation temperature distribution
        st.markdown("---")
        st.markdown("#### Fermentation Temperature Distribution")

        temp_col = 'fermentation_temp_f' if 'fermentation_temp_f' in filtered_yeast.columns else 'primary_temp_F' if 'primary_temp_F' in filtered_yeast.columns else None

        if temp_col and temp_col in filtered_yeast.columns:
            temp_data = filtered_yeast.dropna(subset=[temp_col])

            if len(temp_data) > 0:
                # Bell curve for fermentation temperature
                fig_bell = plot_bell_curve(
                    temp_data[temp_col].values,
                    title='Fermentation Temperature Distribution',
                    height=400
                )
                st.plotly_chart(fig_bell, use_container_width=True)

                # Stats
                stat_cols = st.columns(4)
                stat_cols[0].metric("Mean Temp", f"{temp_data[temp_col].mean():.1f}°F")
                stat_cols[1].metric("Median Temp", f"{temp_data[temp_col].median():.1f}°F")
                stat_cols[2].metric("Min Temp", f"{temp_data[temp_col].min():.1f}°F")
                stat_cols[3].metric("Max Temp", f"{temp_data[temp_col].max():.1f}°F")
            else:
                st.info("No fermentation temperature data available.")
        else:
            st.info("Fermentation temperature data not available in dataset.")
    else:
        st.info("No yeast data available for the selected filters.")

# TAB 5: Water Chemistry
with tab5:
    st.subheader("Water Chemistry Analysis")

    if len(filtered_df) == 0:
        st.warning("No recipes match the current filters. Please adjust your filter settings.")
    else:
        # Define water ion columns
        water_ions = ['Ca_ppm', 'Mg_ppm', 'Na_ppm', 'Cl_ppm', 'SO4_ppm', 'HCO3_ppm']
        available_ions = [col for col in water_ions if col in filtered_df.columns]

        if available_ions:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Ion Concentration Distribution")
                # Box plot for water ions
                fig_box = plot_box_ions(
                    filtered_df,
                    ions=available_ions,
                    title='Water Ion Concentrations',
                    height=450
                )
                st.plotly_chart(fig_box, use_container_width=True)

            with col2:
                st.markdown("#### Sulfate-to-Chloride Ratio")
                if 'SO4_ppm' in available_ions and 'Cl_ppm' in available_ions:
                    # Calculate SO4:Cl ratio
                    ratio_data = filtered_df[['style_group', 'SO4_ppm', 'Cl_ppm']].dropna()
                    if len(ratio_data) > 0:
                        ratio_data = ratio_data.copy()
                        ratio_data['SO4_Cl_ratio'] = ratio_data['SO4_ppm'] / ratio_data['Cl_ppm'].replace(0, np.nan)
                        ratio_avg = ratio_data.groupby('style_group')['SO4_Cl_ratio'].mean().sort_values(ascending=False).head(15)

                        if len(ratio_avg) > 0:
                            fig_ratio = go.Figure(data=[
                                go.Bar(
                                    y=ratio_avg.index,
                                    x=ratio_avg.values,
                                    orientation='h',
                                    marker_color=ratio_avg.values,
                                    marker_colorscale='RdYlBu',
                                    text=[f'{v:.2f}' for v in ratio_avg.values],
                                    textposition='outside'
                                )
                            ])

                            fig_ratio.add_vline(x=1, line_dash="dash", line_color="gray",
                                               annotation_text="1:1 (Balanced)")

                            fig_ratio.update_layout(
                                title="Sulfate:Chloride Ratio by Style",
                                xaxis_title="SO4:Cl Ratio",
                                yaxis_title="",
                                height=450,
                                showlegend=False
                            )
                            st.plotly_chart(fig_ratio, use_container_width=True)

                            st.info("**Interpretation**: Ratio > 1 emphasizes hop bitterness (IPAs), Ratio < 1 emphasizes malt sweetness (stouts)")
                        else:
                            st.info("No sulfate-to-chloride ratio data available.")
                    else:
                        st.info("No water chemistry data available for ratio calculation.")
                else:
                    st.info("Sulfate or Chloride data not available.")

            # Water profile by style
            st.markdown("---")
            st.markdown("#### Average Water Profile by Style Group")

            water_by_style = filtered_df.groupby('style_group')[available_ions].mean().head(10)

            if len(water_by_style) > 0:
                fig_water = go.Figure()
                ion_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
                ion_names = ['Calcium', 'Magnesium', 'Sodium', 'Chloride', 'Sulfate', 'Bicarbonate']

                for i, col in enumerate(available_ions):
                    fig_water.add_trace(go.Bar(
                        name=ion_names[i] if i < len(ion_names) else col,
                        x=water_by_style.index,
                        y=water_by_style[col],
                        marker_color=ion_colors[i % len(ion_colors)]
                    ))

                fig_water.update_layout(
                    title="Water Ion Profile by Style Group",
                    xaxis_title="Style Group",
                    yaxis_title="Concentration (ppm)",
                    barmode='group',
                    height=450,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig_water, use_container_width=True)
        else:
            st.info("Water chemistry data not available in the current dataset.")

# TAB 6: Mash, Boil & Fermentation
with tab6:
    st.subheader("Mash, Boil & Fermentation Analysis")

    if len(filtered_df) == 0:
        st.warning("No recipes match the current filters. Please adjust your filter settings.")
    else:
        # Load additional data files
        @st.cache_data
        def load_process_data():
            base_path = Path("dataset/processed")
            mash_steps = pd.read_csv(base_path / "mash_steps_normalized.csv")
            ferm_stages = pd.read_csv(base_path / "fermentation_stages_normalized.csv")
            return mash_steps, ferm_stages

        try:
            mash_steps_df, ferm_stages_df = load_process_data()
        except Exception:
            mash_steps_df = pd.DataFrame()
            ferm_stages_df = pd.DataFrame()

        filtered_recipe_ids = filtered_df['recipe_id'].unique()

        # -----------------------------------------------------------
        # 1. MASH METHODS
        # -----------------------------------------------------------
        st.markdown("### 1. Mash Methods & Temperature")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Mash Type Distribution")
            if 'mash_type' in filtered_df.columns and len(filtered_df['mash_type'].dropna()) > 0:
                mash_counts = filtered_df['mash_type'].value_counts().reset_index()
                mash_counts.columns = ['mash_type', 'count']

                fig_mash = plot_donut(
                    data=mash_counts,
                    names_col='mash_type',
                    values_col='count',
                    title='Mash Type Distribution',
                    height=400
                )
                st.plotly_chart(fig_mash, use_container_width=True)
            else:
                st.info("Mash type data not available.")

        with col2:
            st.markdown("#### Mash Temperature Distribution")
            if len(mash_steps_df) > 0:
                filtered_mash = mash_steps_df[mash_steps_df['recipe_id'].isin(filtered_recipe_ids)]

                if 'temp_F' in filtered_mash.columns and len(filtered_mash) > 0:
                    mash_temp_data = filtered_mash['temp_F'].dropna().values

                    if len(mash_temp_data) > 3:
                        fig_mash_temp = plot_bell_curve(
                            mash_temp_data,
                            title='Mash Temperature Distribution',
                            height=400
                        )
                        st.plotly_chart(fig_mash_temp, use_container_width=True)
                    else:
                        st.info("Not enough mash temperature data.")
                else:
                    st.info("Mash temperature data not available.")
            else:
                st.info("Mash steps data not loaded.")

        # Mash step violin plot
        if len(mash_steps_df) > 0:
            st.markdown("#### Mash Step Temperature by Step Name")
            filtered_mash = mash_steps_df[mash_steps_df['recipe_id'].isin(filtered_recipe_ids)]

            if 'step_name' in filtered_mash.columns and 'temp_F' in filtered_mash.columns:
                mash_violin_data = filtered_mash.dropna(subset=['step_name', 'temp_F'])

                if len(mash_violin_data) > 0:
                    # Clean step names
                    mash_violin_data = mash_violin_data.copy()
                    mash_violin_data['step_name_clean'] = mash_violin_data['step_name'].str.lower().str.strip()

                    fig_violin = plot_violin(
                        mash_violin_data,
                        x_col='step_name_clean',
                        y_col='temp_F',
                        title='Mash Step Temperature Distribution',
                        x_label='Mash Step',
                        y_label='Temperature (°F)',
                        height=400
                    )
                    st.plotly_chart(fig_violin, use_container_width=True)

        st.markdown("---")

        # -----------------------------------------------------------
        # 2. BOIL DURATION
        # -----------------------------------------------------------
        st.markdown("### 2. Boil Duration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Boil Time Distribution")
            if 'boil_time_min' in filtered_df.columns:
                boil_data = filtered_df['boil_time_min'].dropna().values

                if len(boil_data) > 3:
                    fig_boil = plot_bell_curve(
                        boil_data,
                        title='Boil Duration Distribution',
                        height=400
                    )
                    st.plotly_chart(fig_boil, use_container_width=True)
                else:
                    st.info("Not enough boil time data.")
            else:
                st.info("Boil time data not available.")

        with col2:
            st.markdown("#### Boil Duration Trend Over Time")
            if 'boil_time_min' in filtered_df.columns:
                boil_trend_data = filtered_df.dropna(subset=['year', 'boil_time_min'])
                if len(boil_trend_data) > 2:
                    fig_boil_trend, stats = plot_scatter_trend(
                        boil_trend_data,
                        x_col='year',
                        y_col='boil_time_min',
                        title='Boil Duration Over Time',
                        x_label='Year',
                        y_label='Boil Time (min)',
                        height=400
                    )
                    st.plotly_chart(fig_boil_trend, use_container_width=True)

                    if stats:
                        stat_cols = st.columns(3)
                        stat_cols[0].metric("Pearson's r", f"{stats.get('pearson_r', 0):.3f}")
                        stat_cols[1].metric("Trend", stats.get('trend', 'N/A'))
                        stat_cols[2].metric("Avg Boil", f"{filtered_df['boil_time_min'].mean():.0f} min")
                else:
                    st.info("Not enough data for trend analysis.")

        st.markdown("---")

        # -----------------------------------------------------------
        # 3. FERMENTATION
        # -----------------------------------------------------------
        st.markdown("### 3. Fermentation Temperature by Stage")

        if len(ferm_stages_df) > 0:
            filtered_ferm = ferm_stages_df[ferm_stages_df['recipe_id'].isin(filtered_recipe_ids)]

            if len(filtered_ferm) > 0:
                # Calculate midpoint temperature if not present
                if 'temp_F' not in filtered_ferm.columns:
                    if 'start_temp_F' in filtered_ferm.columns and 'end_temp_F' in filtered_ferm.columns:
                        filtered_ferm = filtered_ferm.copy()
                        filtered_ferm['temp_F'] = filtered_ferm[['start_temp_F', 'end_temp_F']].mean(axis=1)

                if 'stage' in filtered_ferm.columns and 'temp_F' in filtered_ferm.columns:
                    ferm_data = filtered_ferm.dropna(subset=['stage', 'temp_F'])

                    if len(ferm_data) > 0:
                        # Clean stage names
                        ferm_data = ferm_data.copy()
                        ferm_data['stage_clean'] = ferm_data['stage'].str.lower().str.strip()

                        fig_ferm_violin = plot_violin(
                            ferm_data,
                            x_col='stage_clean',
                            y_col='temp_F',
                            title='Fermentation Temperature by Stage',
                            x_label='Fermentation Stage',
                            y_label='Temperature (°F)',
                            height=450
                        )
                        st.plotly_chart(fig_ferm_violin, use_container_width=True)

                        # Summary stats
                        stage_stats = ferm_data.groupby('stage_clean')['temp_F'].agg(['mean', 'std', 'count']).round(1)
                        st.dataframe(stage_stats.rename(columns={'mean': 'Avg Temp (°F)', 'std': 'Std Dev', 'count': 'Count'}))
                    else:
                        st.info("No fermentation temperature data after filtering.")
                else:
                    st.info("Fermentation stage or temperature columns not available.")
            else:
                st.info("No fermentation data for selected recipes.")
        else:
            st.info("Fermentation stages data not loaded.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Data source: 1,395 award-winning homebrew recipes from the Homebrewers Association</p>
        <p>Dashboard created with Streamlit | Data processing: Python + Pandas</p>
    </div>
""", unsafe_allow_html=True)
