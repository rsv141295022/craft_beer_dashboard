import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from scipy import stats

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

# Apply final filters (ABV and Year)
filtered_df = temp_df[
    (temp_df['abv_pct'].between(abv_min, abv_max)) &
    (temp_df['year'].between(selected_years[0], selected_years[1]))
]

# Key metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Recipes", f"{len(filtered_df):,}")

with col2:
    avg_abv = filtered_df['abv_pct'].mean()
    st.metric("Avg ABV", f"{avg_abv:.1f}%")

with col3:
    avg_ibu = filtered_df['ibu'].mean()
    st.metric("Avg IBU", f"{avg_ibu:.0f}")

with col4:
    avg_og = filtered_df['og'].mean()
    st.metric("Avg OG", f"{avg_og:.3f}")

with col5:
    gold_medals = len(filtered_df[filtered_df['medal'] == 'Gold'])
    st.metric("Gold Medals", f"{gold_medals:,}")

st.markdown("---")

# Tab navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🌾 Malts & Grains",
    "🌿 Hops",
    "🧬 Yeast",
    "🔍 Recipe Explorer",
    "📈 Advanced Analytics"
])

# TAB 1: Overview
with tab1:
    st.subheader("📊 Recipe Distribution Insights")

    # Row 1: Medal Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribution by Medal")
        medal_counts = filtered_df['medal'].value_counts()

        # Define custom colors for medals
        medal_colors = {
            'NHC GOLD': '#FFD700',       # Gold
            'NHC SILVER': '#C0C0C0',     # Silver
            'NHC COPPER': '#CD7F32',     # Bronze/Copper
            'No Medal': '#6495ED',       # Cornflower blue
            'CLONE': '#9370DB',          # Medium purple
            'PRO AM': '#20B2AA',         # Light sea green
            'NORMAL MEDAL': '#FF6347'    # Tomato red
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

        # Display count
        st.info(f"**Total Medal Types:** {len(medal_counts)} | **Total Recipes:** {medal_counts.sum()}")

    with col2:
        # Specific Style (Top 15)
        st.markdown("#### Distribution by Specific Beer Style (Top 15)")
        style_counts = filtered_df['style'].value_counts().head(15)
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

        total_styles = len(filtered_df['style'].value_counts())
        st.info(f"**Total Specific Styles:** {total_styles} | **Showing Top:** 15")

    st.markdown("---")

    # Time series
    st.subheader("Recipe Trends Over Time")
    yearly_counts = filtered_df.groupby('year').size().reset_index(name='count')
    fig = px.line(
        yearly_counts,
        x='year',
        y='count',
        markers=True,
        labels={'year': 'Year', 'count': 'Number of Recipes'}
    )
    fig.update_traces(line_color='#D4A574', marker=dict(size=8))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Specifications scatter
    st.subheader("ABV vs IBU by Style Group")
    scatter_df = filtered_df.dropna(subset=['abv_pct', 'ibu'])
    fig = px.scatter(
        scatter_df,
        x='abv_pct',
        y='ibu',
        color='style_group',
        hover_data=['title', 'style', 'og', 'fg'],
        labels={'abv_pct': 'ABV (%)', 'ibu': 'IBU', 'style_group': 'Style Group'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: Malts & Grains
with tab2:
    st.subheader("Malt Analysis")

    # Filter malts by selected recipes
    filtered_recipe_ids = filtered_df['recipe_id'].unique()
    filtered_malts = malts_df[malts_df['recipe_id'].isin(filtered_recipe_ids)]

    col1, col2 = st.columns(2)

    with col1:
        # Top malts
        st.markdown("#### Top 15 Malts Used")
        malt_col = 'malt_name_normalized' if 'malt_name_normalized' in filtered_malts.columns else 'malt_name' if 'malt_name' in filtered_malts.columns else 'malt'
        top_malts = filtered_malts[malt_col].value_counts().head(15)
        fig = px.bar(
            x=top_malts.values,
            y=top_malts.index,
            orientation='h',
            labels={'x': 'Usage Count', 'y': 'Malt'},
            color=top_malts.values,
            color_continuous_scale='ylorrd'
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Malt type distribution
        st.markdown("#### Malt Type Distribution")
        malt_type_counts = filtered_malts['malt_type'].value_counts()
        fig = px.pie(
            values=malt_type_counts.values,
            names=malt_type_counts.index,
            color_discrete_sequence=px.colors.sequential.Oranges
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Grist composition
    st.markdown("#### Average Grist Composition by Style")
    grist_cols = ['base_malt_pct', 'crystal_pct', 'roast_pct', 'adjunct_pct']
    available_grist_cols = [col for col in grist_cols if col in filtered_df.columns]

    if available_grist_cols:
        grist_by_style = filtered_df.groupby('style_group')[available_grist_cols].mean().head(10)

        fig = go.Figure()
        for col in available_grist_cols:
            fig.add_trace(go.Bar(
                name=col.replace('_pct', '').replace('_', ' ').title(),
                x=grist_by_style.index,
                y=grist_by_style[col]
            ))

        fig.update_layout(
            barmode='stack',
            xaxis_title='Beer Style',
            yaxis_title='Percentage of Grist (%)',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: Hops
with tab3:
    st.subheader("Hop Analysis")

    # Filter hops by selected recipes
    filtered_recipe_ids = filtered_df['recipe_id'].unique()
    filtered_hops = hops_df[hops_df['recipe_id'].isin(filtered_recipe_ids)]

    col1, col2 = st.columns(2)

    with col1:
        # Top hops
        st.markdown("#### Top 15 Hops Used")
        hop_col = 'hop_name_normalized' if 'hop_name_normalized' in filtered_hops.columns else 'hop_name' if 'hop_name' in filtered_hops.columns else 'hop'
        top_hops = filtered_hops[hop_col].value_counts().head(15)
        fig = px.bar(
            x=top_hops.values,
            y=top_hops.index,
            orientation='h',
            labels={'x': 'Usage Count', 'y': 'Hop'},
            color=top_hops.values,
            color_continuous_scale='Greens'
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Hop usage type
        st.markdown("#### Hop Usage Type")
        usage_col = 'usage' if 'usage' in filtered_hops.columns else 'hop_usage'
        usage_counts = filtered_hops[usage_col].value_counts()
        fig = px.pie(
            values=usage_counts.values,
            names=usage_counts.index,
            color_discrete_sequence=px.colors.sequential.Greens
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Hopping rates by style
    st.markdown("#### Hopping Rates by Style (oz/gallon)")
    hop_cols = ['bittering_oz_gal', 'flavor_oz_gal', 'aroma_oz_gal', 'dry_hop_oz_gal']
    available_hop_cols = [col for col in hop_cols if col in filtered_df.columns]

    if available_hop_cols:
        hop_by_style = filtered_df.groupby('style_group')[available_hop_cols].mean().head(10)

        fig = go.Figure()
        for col in available_hop_cols:
            fig.add_trace(go.Bar(
                name=col.replace('_oz_gal', '').replace('_', ' ').title(),
                x=hop_by_style.index,
                y=hop_by_style[col]
            ))

        fig.update_layout(
            barmode='group',
            xaxis_title='Beer Style',
            yaxis_title='Hops (oz/gallon)',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

# TAB 4: Yeast
with tab4:
    st.subheader("Yeast Analysis")

    # Filter yeast by selected recipes
    filtered_recipe_ids = filtered_df['recipe_id'].unique()
    filtered_yeast = yeast_df[yeast_df['recipe_id'].isin(filtered_recipe_ids)]

    col1, col2 = st.columns(2)

    with col1:
        # Top yeast strains
        st.markdown("#### Top 15 Yeast Strains")
        top_yeast = filtered_yeast['yeast_canonical'].value_counts().head(15)
        fig = px.bar(
            x=top_yeast.values,
            y=top_yeast.index,
            orientation='h',
            labels={'x': 'Usage Count', 'y': 'Yeast Strain'},
            color=top_yeast.values,
            color_continuous_scale='Purples'
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Fermentation temperature distribution
        st.markdown("#### Fermentation Temperature Distribution")
        if 'fermentation_temp_f' in filtered_yeast.columns:
            temp_data = filtered_yeast.dropna(subset=['fermentation_temp_f'])
            if len(temp_data) > 0:
                fig = px.histogram(
                    temp_data,
                    x='fermentation_temp_f',
                    nbins=30,
                    labels={'fermentation_temp_f': 'Temperature (°F)', 'count': 'Frequency'},
                    color_discrete_sequence=['#9467bd']
                )
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No fermentation temperature data available for selected filters.")
        else:
            st.info("Fermentation temperature data not available.")

# TAB 5: Recipe Explorer
with tab5:
    st.subheader("Recipe Explorer")

    # Display dataframe with selection
    display_cols = [
        'title', 'style', 'abv_pct', 'ibu', 'og', 'fg', 'srm',
        'medal', 'year', 'num_malts', 'num_hops'
    ]
    available_display_cols = [col for col in display_cols if col in filtered_df.columns]

    st.dataframe(
        filtered_df[available_display_cols].sort_values('year', ascending=False),
        use_container_width=True,
        height=400
    )

    # Recipe detail view
    st.markdown("---")
    st.markdown("#### Recipe Details")

    recipe_titles = filtered_df['title'].unique()
    selected_recipe = st.selectbox("Select a recipe to view details:", recipe_titles)

    if selected_recipe:
        recipe = filtered_df[filtered_df['title'] == selected_recipe].iloc[0]
        recipe_id = recipe['recipe_id']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Specifications**")
            st.write(f"**Style:** {recipe['style']}")
            st.write(f"**ABV:** {recipe['abv_pct']:.1f}%")
            st.write(f"**IBU:** {recipe['ibu']:.0f}" if pd.notna(recipe['ibu']) else "**IBU:** N/A")
            st.write(f"**OG:** {recipe['og']:.3f}" if pd.notna(recipe['og']) else "**OG:** N/A")
            st.write(f"**FG:** {recipe['fg']:.3f}" if pd.notna(recipe['fg']) else "**FG:** N/A")
            st.write(f"**SRM:** {recipe['srm']:.0f}" if pd.notna(recipe['srm']) else "**SRM:** N/A")
            st.write(f"**Batch Size:** {recipe['batch_size_gal']:.1f} gal" if pd.notna(recipe['batch_size_gal']) else "**Batch Size:** N/A")

        with col2:
            st.markdown("**Ingredients**")
            recipe_malts = malts_df[malts_df['recipe_id'] == recipe_id]
            recipe_hops = hops_df[hops_df['recipe_id'] == recipe_id]
            recipe_yeast = yeast_df[yeast_df['recipe_id'] == recipe_id]

            st.write(f"**Malts ({len(recipe_malts)}):**")
            for _, malt in recipe_malts.iterrows():
                malt_name = malt.get('malt_name_normalized', malt.get('malt_name', malt.get('malt', 'Unknown')))
                malt_type = malt.get('malt_type', 'Unknown')
                st.write(f"- {malt_name} ({malt_type})")

            st.write(f"\n**Hops ({len(recipe_hops)}):**")
            for _, hop in recipe_hops.iterrows():
                hop_name = hop.get('hop_name_normalized', hop.get('hop_name', hop.get('hop', 'Unknown')))
                hop_usage = hop.get('usage', hop.get('hop_usage', 'Unknown'))
                st.write(f"- {hop_name} ({hop_usage})")

        with col3:
            st.markdown("**Awards & Process**")
            st.write(f"**Medal:** {recipe['medal']}")
            st.write(f"**Final Category:** {recipe['final_category']}" if pd.notna(recipe['final_category']) else "**Final Category:** N/A")
            st.write(f"**Year:** {int(recipe['year'])}" if pd.notna(recipe['year']) else "**Year:** N/A")
            st.write(f"**Efficiency:** {recipe['efficiency_pct']:.0f}%" if pd.notna(recipe['efficiency_pct']) else "**Efficiency:** N/A")
            st.write(f"**Boil Time:** {recipe['boil_time_min']:.0f} min" if pd.notna(recipe['boil_time_min']) else "**Boil Time:** N/A")

            if len(recipe_yeast) > 0:
                st.write(f"\n**Yeast:**")
                for _, y in recipe_yeast.iterrows():
                    st.write(f"- {y['yeast_canonical']}")

# TAB 6: Advanced Analytics
with tab6:
    st.subheader("📈 Advanced Analytics")

    # Filter data for analytics
    filtered_recipe_ids = filtered_df['recipe_id'].unique()
    filtered_hops_analytics = hops_df[hops_df['recipe_id'].isin(filtered_recipe_ids)]
    filtered_malts_analytics = malts_df[malts_df['recipe_id'].isin(filtered_recipe_ids)]
    filtered_yeast_analytics = yeast_df[yeast_df['recipe_id'].isin(filtered_recipe_ids)]
    filtered_water_analytics = water_df[water_df['recipe_id'].isin(filtered_recipe_ids)]

    # Style selector for some analytics
    available_styles = case_insensitive_sort(filtered_df['style'].dropna().unique().tolist())

    # -----------------------------------------------------------
    # 1. STYLE EVOLUTION (TREND ANALYSIS)
    # -----------------------------------------------------------
    st.markdown("### 1. Style Evolution - Trend Analysis")
    st.markdown("*Scatter plot with linear regression showing how beer characteristics change over time*")

    col1, col2 = st.columns([1, 3])

    with col1:
        # Variable selector
        trend_variable = st.selectbox(
            "Select Variable to Analyze:",
            ["og", "fg", "abv_pct", "ibu", "srm"],
            format_func=lambda x: {
                "og": "Original Gravity (OG)",
                "fg": "Final Gravity (FG)",
                "abv_pct": "ABV (%)",
                "ibu": "IBU (Bitterness)",
                "srm": "SRM (Color)"
            }.get(x, x)
        )

        # Style selector for trend
        trend_style = st.selectbox(
            "Select Style (optional):",
            ["All Styles"] + available_styles,
            key="trend_style"
        )

    with col2:
        # Filter data for trend analysis
        trend_data = filtered_df.dropna(subset=['year', trend_variable]).copy()
        if trend_style != "All Styles":
            trend_data = trend_data[trend_data['style'] == trend_style]

        if len(trend_data) > 5:
            # Calculate Pearson correlation
            x = trend_data['year'].values
            y = trend_data[trend_variable].values

            # Remove any infinities or NaN
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            if len(x) > 2:
                correlation, p_value = stats.pearsonr(x, y)
                slope, intercept = np.polyfit(x, y, 1)

                # Create scatter plot with trend line
                fig = go.Figure()

                # Scatter points
                fig.add_trace(go.Scatter(
                    x=trend_data['year'],
                    y=trend_data[trend_variable],
                    mode='markers',
                    name='Recipes',
                    marker=dict(
                        size=8,
                        color=trend_data[trend_variable],
                        colorscale='Viridis',
                        opacity=0.6
                    ),
                    text=trend_data['title'],
                    hovertemplate='<b>%{text}</b><br>Year: %{x}<br>Value: %{y:.2f}<extra></extra>'
                ))

                # Trend line
                x_line = np.array([trend_data['year'].min(), trend_data['year'].max()])
                y_line = slope * x_line + intercept

                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    name=f'Trend (r={correlation:.3f})',
                    line=dict(color='red', width=3, dash='dash')
                ))

                var_label = {
                    "og": "Original Gravity",
                    "fg": "Final Gravity",
                    "abv_pct": "ABV (%)",
                    "ibu": "IBU",
                    "srm": "SRM"
                }.get(trend_variable, trend_variable)

                fig.update_layout(
                    title=f"{var_label} Trend Over Time {'(' + trend_style + ')' if trend_style != 'All Styles' else ''}",
                    xaxis_title="Year",
                    yaxis_title=var_label,
                    height=450,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display statistics
                trend_dir = "increasing" if slope > 0 else "decreasing"
                strength = "strong" if abs(correlation) > 0.5 else "moderate" if abs(correlation) > 0.3 else "weak"

                col_stat1, col_stat2, col_stat3 = st.columns(3)
                col_stat1.metric("Pearson's r", f"{correlation:.3f}")
                col_stat2.metric("P-value", f"{p_value:.4f}")
                col_stat3.metric("Trend", f"{strength.title()} {trend_dir}")
            else:
                st.warning("Not enough data points for trend analysis.")
        else:
            st.warning("Not enough data to display trend analysis. Try adjusting your filters.")

    st.markdown("---")

    # -----------------------------------------------------------
    # 2. TECHNICAL DISTRIBUTIONS (THE "SWEET SPOT")
    # -----------------------------------------------------------
    st.markdown("### 2. Technical Distributions - The 'Sweet Spot'")
    st.markdown("*Histogram showing where winning recipes fall compared to typical ranges*")

    col1, col2 = st.columns([1, 3])

    with col1:
        dist_variable = st.selectbox(
            "Select Variable:",
            ["og", "fg", "abv_pct", "ibu", "srm"],
            format_func=lambda x: {
                "og": "Original Gravity (OG)",
                "fg": "Final Gravity (FG)",
                "abv_pct": "ABV (%)",
                "ibu": "IBU (Bitterness)",
                "srm": "SRM (Color)"
            }.get(x, x),
            key="dist_var"
        )

        dist_style = st.selectbox(
            "Select Style (optional):",
            ["All Styles"] + available_styles,
            key="dist_style"
        )

        show_box = st.checkbox("Show Box Plot", value=False)

    with col2:
        dist_data = filtered_df.dropna(subset=[dist_variable]).copy()
        if dist_style != "All Styles":
            dist_data = dist_data[dist_data['style'] == dist_style]

        if len(dist_data) > 0:
            var_label = {
                "og": "Original Gravity",
                "fg": "Final Gravity",
                "abv_pct": "ABV (%)",
                "ibu": "IBU",
                "srm": "SRM"
            }.get(dist_variable, dist_variable)

            if show_box:
                # Box plot
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=dist_data[dist_variable],
                    name=var_label,
                    boxpoints='outliers',
                    marker_color='#D4A574',
                    line_color='#8B4513'
                ))
                fig.update_layout(
                    title=f"{var_label} Distribution {'(' + dist_style + ')' if dist_style != 'All Styles' else ''}",
                    yaxis_title=var_label,
                    height=450,
                    showlegend=False
                )
            else:
                # Histogram
                fig = px.histogram(
                    dist_data,
                    x=dist_variable,
                    nbins=30,
                    color_discrete_sequence=['#D4A574'],
                    labels={dist_variable: var_label}
                )

                # Add mean and median lines
                mean_val = dist_data[dist_variable].mean()
                median_val = dist_data[dist_variable].median()

                fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                             annotation_text=f"Mean: {mean_val:.2f}")
                fig.add_vline(x=median_val, line_dash="dot", line_color="blue",
                             annotation_text=f"Median: {median_val:.2f}")

                fig.update_layout(
                    title=f"{var_label} Distribution {'(' + dist_style + ')' if dist_style != 'All Styles' else ''}",
                    xaxis_title=var_label,
                    yaxis_title="Number of Recipes",
                    height=450,
                    showlegend=False
                )

            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Mean", f"{dist_data[dist_variable].mean():.2f}")
            col_s2.metric("Median", f"{dist_data[dist_variable].median():.2f}")
            col_s3.metric("Std Dev", f"{dist_data[dist_variable].std():.2f}")
            col_s4.metric("Recipes", f"{len(dist_data):,}")
        else:
            st.warning("No data available for selected filters.")

    st.markdown("---")

    # -----------------------------------------------------------
    # 3. MALT GRIST COMPOSITION
    # -----------------------------------------------------------
    st.markdown("### 3. Malt Grist Composition")
    st.markdown("*How grain bill compositions have evolved and typical style profiles*")

    col1, col2 = st.columns(2)

    with col1:
        # Stacked Area Chart - Grist trends over time
        st.markdown("#### Grist Composition Trends Over Time")

        grist_cols = ['base_malt_pct', 'crystal_pct', 'roast_pct', 'adjunct_pct']
        available_grist = [col for col in grist_cols if col in filtered_df.columns]

        if available_grist:
            yearly_grist = filtered_df.groupby('year')[available_grist].mean().reset_index()
            yearly_grist = yearly_grist.dropna()

            if len(yearly_grist) > 0:
                fig = go.Figure()

                colors = ['#8B4513', '#DAA520', '#2F4F4F', '#D2B48C']
                names = ['Base Malt', 'Crystal', 'Roast', 'Adjunct']

                for i, col in enumerate(available_grist):
                    fig.add_trace(go.Scatter(
                        x=yearly_grist['year'],
                        y=yearly_grist[col],
                        mode='lines',
                        name=names[i] if i < len(names) else col,
                        fill='tonexty' if i > 0 else 'tozeroy',
                        line=dict(color=colors[i % len(colors)])
                    ))

                fig.update_layout(
                    xaxis_title="Year",
                    yaxis_title="Percentage of Grist (%)",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough yearly data for trend visualization.")
        else:
            st.info("Grist composition data not available.")

    with col2:
        # Pie/Donut Chart - Average composition by style
        st.markdown("#### Average Grist Composition")

        grist_style = st.selectbox(
            "Select Style:",
            ["All Styles"] + available_styles,
            key="grist_style"
        )

        if available_grist:
            if grist_style == "All Styles":
                grist_avg = filtered_df[available_grist].mean()
            else:
                style_data = filtered_df[filtered_df['style'] == grist_style]
                grist_avg = style_data[available_grist].mean()

            grist_avg = grist_avg.dropna()

            if len(grist_avg) > 0:
                labels = ['Base Malt', 'Crystal', 'Roast', 'Adjunct'][:len(grist_avg)]
                colors = ['#8B4513', '#DAA520', '#2F4F4F', '#D2B48C'][:len(grist_avg)]

                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=grist_avg.values,
                    hole=0.4,
                    marker=dict(colors=colors),
                    textposition='inside',
                    textinfo='percent+label'
                )])

                fig.update_layout(
                    title=f"{'All Styles' if grist_style == 'All Styles' else grist_style}",
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No grist data available.")

    st.markdown("---")

    # -----------------------------------------------------------
    # 4. HOP SCHEDULE AND "LUPULIN SHIFT"
    # -----------------------------------------------------------
    st.markdown("### 4. Hop Schedule - The 'Lupulin Shift'")
    st.markdown("*How hop usage patterns have evolved: from bittering to late/dry hopping*")

    hop_rate_cols = ['bittering_oz_gal', 'flavor_oz_gal', 'aroma_oz_gal', 'dry_hop_oz_gal']
    available_hop_rates = [col for col in hop_rate_cols if col in filtered_df.columns]

    if available_hop_rates:
        # Calculate yearly averages
        yearly_hops = filtered_df.groupby('year')[available_hop_rates].mean().reset_index()
        yearly_hops = yearly_hops.dropna()

        if len(yearly_hops) > 0:
            fig = go.Figure()

            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            names = ['Bittering', 'Flavor', 'Aroma/Whirlpool', 'Dry Hop']

            for i, col in enumerate(available_hop_rates):
                fig.add_trace(go.Scatter(
                    x=yearly_hops['year'],
                    y=yearly_hops[col],
                    mode='lines+markers',
                    name=names[i] if i < len(names) else col,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=8)
                ))

            fig.update_layout(
                title="Hop Usage Rates Over Time (oz/gallon)",
                xaxis_title="Year",
                yaxis_title="Hop Rate (oz/gallon)",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info("📊 **Lupulin Shift**: Notice how late hopping (aroma, dry hop) rates have increased over time while early bittering additions may have decreased in some styles.")
        else:
            st.info("Not enough yearly data for hop schedule visualization.")
    else:
        st.info("Hop schedule data not available in the current dataset.")

    st.markdown("---")

    # -----------------------------------------------------------
    # 5. HOP TIMING DISTRIBUTION
    # -----------------------------------------------------------
    st.markdown("### 5. Hop Timing Distribution")
    st.markdown("*When are hops added during the brewing process?*")

    if 'time_min' in filtered_hops_analytics.columns:
        # Create bins for hop timing
        hop_timing = filtered_hops_analytics.copy()
        hop_timing = hop_timing[hop_timing['time_min'].notna()]

        if len(hop_timing) > 0:
            # Define time bins
            def categorize_time(t):
                if t >= 60:
                    return '60+ min (Bittering)'
                elif t >= 30:
                    return '30-59 min'
                elif t >= 15:
                    return '15-29 min'
                elif t >= 5:
                    return '5-14 min'
                elif t > 0:
                    return '1-4 min (Flame Out)'
                else:
                    return '0 min (Whirlpool/DH)'

            hop_timing['time_category'] = hop_timing['time_min'].apply(categorize_time)

            # Order categories
            category_order = ['60+ min (Bittering)', '30-59 min', '15-29 min',
                            '5-14 min', '1-4 min (Flame Out)', '0 min (Whirlpool/DH)']

            timing_counts = hop_timing['time_category'].value_counts()
            timing_counts = timing_counts.reindex(category_order).dropna()

            # Calculate percentages
            total = timing_counts.sum()
            timing_pct = (timing_counts / total * 100).round(1)

            fig = go.Figure(data=[
                go.Bar(
                    x=timing_counts.index,
                    y=timing_pct.values,
                    marker_color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#95C623'],
                    text=[f'{v:.1f}%' for v in timing_pct.values],
                    textposition='outside'
                )
            ])

            fig.update_layout(
                title="Hop Addition Timing Distribution",
                xaxis_title="Addition Time",
                yaxis_title="Percentage of Additions (%)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info(f"📊 Total hop additions analyzed: {total:,.0f}")
        else:
            st.info("No hop timing data available for selected filters.")
    else:
        st.info("Hop timing data not available.")

    st.markdown("---")

    # -----------------------------------------------------------
    # 6. YEAST STRAIN PERFORMANCE
    # -----------------------------------------------------------
    st.markdown("### 6. Yeast Strain Performance")
    st.markdown("*Popular strains and their fermentation temperature ranges*")

    if len(filtered_yeast_analytics) > 0:
        yeast_col = 'yeast_canonical' if 'yeast_canonical' in filtered_yeast_analytics.columns else 'yeast_name'
        temp_col = 'primary_temp_F' if 'primary_temp_F' in filtered_yeast_analytics.columns else 'fermentation_temp_f'

        if temp_col in filtered_yeast_analytics.columns:
            # Get top 15 yeast strains
            top_strains = filtered_yeast_analytics[yeast_col].value_counts().head(15).index.tolist()
            yeast_subset = filtered_yeast_analytics[filtered_yeast_analytics[yeast_col].isin(top_strains)]

            # Calculate statistics per strain
            yeast_stats = yeast_subset.groupby(yeast_col)[temp_col].agg(['count', 'mean', 'min', 'max', 'std']).reset_index()
            yeast_stats = yeast_stats.dropna(subset=['mean'])
            yeast_stats = yeast_stats.sort_values('count', ascending=True)

            if len(yeast_stats) > 0:
                fig = go.Figure()

                # Bar for count (popularity)
                fig.add_trace(go.Bar(
                    y=yeast_stats[yeast_col],
                    x=yeast_stats['count'],
                    name='Usage Count',
                    orientation='h',
                    marker_color='rgba(148, 103, 189, 0.7)',
                    text=yeast_stats['count'].astype(int),
                    textposition='outside'
                ))

                fig.update_layout(
                    title="Top Yeast Strains by Usage",
                    xaxis_title="Number of Recipes",
                    yaxis_title="Yeast Strain",
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # Temperature range chart
                st.markdown("#### Fermentation Temperature Ranges by Strain")

                fig2 = go.Figure()

                for i, row in yeast_stats.iterrows():
                    if pd.notna(row['min']) and pd.notna(row['max']):
                        fig2.add_trace(go.Scatter(
                            x=[row['min'], row['mean'], row['max']],
                            y=[row[yeast_col]] * 3,
                            mode='lines+markers',
                            name=row[yeast_col],
                            line=dict(width=8),
                            marker=dict(size=12),
                            showlegend=False,
                            hovertemplate=f"<b>{row[yeast_col]}</b><br>Min: {row['min']:.1f}°F<br>Mean: {row['mean']:.1f}°F<br>Max: {row['max']:.1f}°F<extra></extra>"
                        ))

                fig2.update_layout(
                    title="Fermentation Temperature Ranges",
                    xaxis_title="Temperature (°F)",
                    yaxis_title="Yeast Strain",
                    height=500
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No temperature data available for yeast strains.")
        else:
            st.info("Fermentation temperature data not available.")
    else:
        st.info("No yeast data available for selected filters.")

    st.markdown("---")

    # -----------------------------------------------------------
    # 7. WATER CHEMISTRY PROFILE
    # -----------------------------------------------------------
    st.markdown("### 7. Water Chemistry Profile")
    st.markdown("*Ion concentrations and the critical Sulfate-to-Chloride ratio*")

    water_ions = ['Ca_ppm', 'Mg_ppm', 'Na_ppm', 'Cl_ppm', 'SO4_ppm']

    # Check which columns exist in filtered_df
    available_water = [col for col in water_ions if col in filtered_df.columns]

    if available_water:
        col1, col2 = st.columns([1, 3])

        with col1:
            water_style = st.selectbox(
                "Select Style:",
                ["All Styles"] + available_styles,
                key="water_style"
            )

        with col2:
            if water_style == "All Styles":
                water_data = filtered_df[available_water + ['style_group']].dropna()
            else:
                water_data = filtered_df[filtered_df['style'] == water_style][available_water].dropna()

            if len(water_data) > 0:
                # Calculate averages
                if water_style == "All Styles":
                    # Show by style group
                    water_avg = water_data.groupby('style_group')[available_water].mean()
                    water_avg = water_avg.head(10)  # Top 10 style groups

                    fig = go.Figure()

                    ion_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
                    ion_names = ['Calcium', 'Magnesium', 'Sodium', 'Chloride', 'Sulfate']

                    for i, col in enumerate(available_water):
                        fig.add_trace(go.Bar(
                            name=ion_names[i] if i < len(ion_names) else col,
                            x=water_avg.index,
                            y=water_avg[col],
                            marker_color=ion_colors[i % len(ion_colors)]
                        ))

                    fig.update_layout(
                        title="Water Ion Concentrations by Style Group",
                        xaxis_title="Style Group",
                        yaxis_title="Concentration (ppm)",
                        barmode='group',
                        height=450,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                else:
                    # Single style comparison
                    avg_values = water_data[available_water].mean()

                    ion_names = ['Calcium', 'Magnesium', 'Sodium', 'Chloride', 'Sulfate'][:len(available_water)]
                    ion_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6'][:len(available_water)]

                    fig = go.Figure(data=[
                        go.Bar(
                            x=ion_names,
                            y=avg_values.values,
                            marker_color=ion_colors,
                            text=[f'{v:.0f}' for v in avg_values.values],
                            textposition='outside'
                        )
                    ])

                    fig.update_layout(
                        title=f"Average Water Profile: {water_style}",
                        xaxis_title="Ion",
                        yaxis_title="Concentration (ppm)",
                        height=400,
                        showlegend=False
                    )

                st.plotly_chart(fig, use_container_width=True)

                # Sulfate to Chloride Ratio
                if 'SO4_ppm' in available_water and 'Cl_ppm' in available_water:
                    st.markdown("#### Sulfate-to-Chloride Ratio")

                    if water_style == "All Styles":
                        ratio_data = filtered_df[['style_group', 'SO4_ppm', 'Cl_ppm']].dropna()
                        ratio_data['SO4_Cl_ratio'] = ratio_data['SO4_ppm'] / ratio_data['Cl_ppm'].replace(0, np.nan)
                        ratio_avg = ratio_data.groupby('style_group')['SO4_Cl_ratio'].mean().sort_values(ascending=False).head(15)
                    else:
                        style_water = filtered_df[filtered_df['style'] == water_style][['SO4_ppm', 'Cl_ppm']].dropna()
                        if len(style_water) > 0:
                            ratio_avg = pd.Series({water_style: (style_water['SO4_ppm'] / style_water['Cl_ppm'].replace(0, np.nan)).mean()})
                        else:
                            ratio_avg = pd.Series()

                    if len(ratio_avg) > 0:
                        fig2 = go.Figure(data=[
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

                        # Add reference line at 1:1
                        fig2.add_vline(x=1, line_dash="dash", line_color="gray",
                                      annotation_text="1:1 (Balanced)")

                        fig2.update_layout(
                            title="Sulfate:Chloride Ratio (>1 = Hoppy, <1 = Malty)",
                            xaxis_title="SO4:Cl Ratio",
                            yaxis_title="",
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                        st.info("📊 **Interpretation**: Ratio > 1 emphasizes hop bitterness (good for IPAs), Ratio < 1 emphasizes malt sweetness (good for stouts, Oktoberfests)")
            else:
                st.info("No water chemistry data available for selected filters.")
    else:
        st.info("Water chemistry data not available in the current dataset.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Data source: 1,395 award-winning homebrew recipes from the Homebrewers Association</p>
        <p>Dashboard created with Streamlit | Data processing: Python + Pandas</p>
    </div>
""", unsafe_allow_html=True)
