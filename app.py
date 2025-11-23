# Name and Surname : Burak Tülüceoğlu
# Student Number : 2021555064
# Name and Surname : Bilge Yılmaz
# Student Number : 2021555070
# Name and Surname : Halit Şen
# Student Number : 2021555060

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import IsolationForest, RandomForestRegressor # Added RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    ml_available = True
except ImportError:
    st.error("Please install 'scikit-learn' library: pip install scikit-learn")
    ml_available = False

# --- 0. PAGE SETTINGS ---
st.set_page_config(
    page_title="Global Emissions Analysis - Group Project",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. DATA LOADING AND PROCESSING ---
@st.cache_data
def load_data():
    # Read CSV file
    df = pd.read_csv("owid-co2-data.csv")
    
    # Get continent data from Plotly
    gapminder = px.data.gapminder()[['country', 'continent']].drop_duplicates()
    df = pd.merge(df, gapminder, on='country', how='left')
    
    continent_corrections = {
        # --- AMERICAS ---
        'Argentina': 'Americas', 'Bolivia': 'Americas', 'Brazil': 'Americas', 
        'Chile': 'Americas', 'Colombia': 'Americas', 'Ecuador': 'Americas',
        'Guyana': 'Americas', 'Paraguay': 'Americas', 'Peru': 'Americas',
        'Suriname': 'Americas', 'Uruguay': 'Americas', 'Venezuela': 'Americas',
        'French Guiana': 'Americas', 'Falkland Islands': 'Americas',
        'Bahamas': 'Americas', 'Barbados': 'Americas', 'Belize': 'Americas',
        'Canada': 'Americas', 'Costa Rica': 'Americas', 'Cuba': 'Americas',
        'Dominican Republic': 'Americas', 'El Salvador': 'Americas', 'Guatemala': 'Americas',
        'Haiti': 'Americas', 'Honduras': 'Americas', 'Jamaica': 'Americas',
        'Mexico': 'Americas', 'Nicaragua': 'Americas', 'Panama': 'Americas',
        'Trinidad and Tobago': 'Americas', 'United States': 'Americas',
        'Antigua and Barbuda': 'Americas', 'Saint Kitts and Nevis': 'Americas',
        'Saint Lucia': 'Americas', 'Saint Vincent and the Grenadines': 'Americas',
        'Grenada': 'Americas', 'Dominica': 'Americas', 'Greenland': 'Americas',
        'Bermuda': 'Americas', 'Aruba': 'Americas', 'Curacao': 'Americas',
        'Turks and Caicos Islands': 'Americas', 'British Virgin Islands': 'Americas',
        'Anguilla': 'Americas', 'Bonaire Sint Eustatius and Saba': 'Americas',
        'Montserrat': 'Americas', 'Sint Maarten (Dutch part)': 'Americas', 
        'Saint Pierre and Miquelon': 'Americas',

        # --- ASIA ---
        'Russia': 'Asia', 'South Korea': 'Asia', 'North Korea': 'Asia', 'Taiwan': 'Asia',
        'Iran': 'Asia', 'Syria': 'Asia', 'Saudi Arabia': 'Asia', 'United Arab Emirates': 'Asia',
        'Vietnam': 'Asia', 'Thailand': 'Asia', 'Hong Kong': 'Asia', 'Macao': 'Asia',
        'Palestine': 'Asia', 'Yemen': 'Asia', 'Uzbekistan': 'Asia', 'Kazakhstan': 'Asia',
        'Turkmenistan': 'Asia', 'Tajikistan': 'Asia', 'Kyrgyzstan': 'Asia', 'Azerbaijan': 'Asia',
        'Armenia': 'Asia', 'Georgia': 'Asia', 'Qatar': 'Asia', 'Bhutan': 'Asia',
        'Brunei': 'Asia', 'Laos': 'Asia', 'Maldives': 'Asia', 'East Timor': 'Asia', 'Timor': 'Asia',
        
        # --- EUROPE ---
        'Ukraine': 'Europe', 'Czechia': 'Europe', 'Slovakia': 'Europe', 'Belarus': 'Europe',
        'North Macedonia': 'Europe', 'Bosnia and Herzegovina': 'Europe', 'Moldova': 'Europe',
        'Cyprus': 'Europe', 'Estonia': 'Europe', 'Latvia': 'Europe', 'Lithuania': 'Europe',
        'Luxembourg': 'Europe', 'Malta': 'Europe', 'Andorra': 'Europe', 'San Marino': 'Europe',
        'Liechtenstein': 'Europe', 'Vatican': 'Europe', 'Monaco': 'Europe', 'Faroe Islands': 'Europe',
        'Kosovo': 'Europe',
        
        # --- AFRICA ---
        'Democratic Republic of Congo': 'Africa', 'Congo': 'Africa', 'Egypt': 'Africa',
        'South Sudan': 'Africa', 'Eswatini': 'Africa', 'Cape Verde': 'Africa', 'Seychelles': 'Africa',
        'Saint Helena': 'Africa',
        
        # --- OCEANIA ---
        'Papua New Guinea': 'Oceania', 'Fiji': 'Oceania', 'Solomon Islands': 'Oceania',
        'Micronesia (country)': 'Oceania', 'Vanuatu': 'Oceania', 'Samoa': 'Oceania',
        'Kiribati': 'Oceania', 'Tonga': 'Oceania', 'Marshall Islands': 'Oceania',
        'Palau': 'Oceania', 'Nauru': 'Oceania', 'Tuvalu': 'Oceania', 'New Caledonia': 'Oceania',
        'French Polynesia': 'Oceania', 'Cook Islands': 'Oceania', 'Wallis and Futuna': 'Oceania',
        'Niue': 'Oceania', 'Christmas Island': 'Oceania'
    }
    
    for country, continent in continent_corrections.items():
        df.loc[df['country'] == country, 'continent'] = continent
        
    df['continent'] = df['continent'].fillna('Other')
    
    # Filter valid countries
    df_countries = df[df['iso_code'].notna()]
    df_countries = df_countries[~df_countries['iso_code'].str.contains('OWID', na=False) | (df_countries['iso_code'] == 'OWID_KOS')]
    
    # Clean numeric columns
    emission_cols = [c for c in df.columns if 'co2' in c or 'methane' in c or 'nitrous' in c]
    df_countries[emission_cols] = df_countries[emission_cols].fillna(0)
    
    return df_countries

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- HELPER FUNCTION: HANDLE MISSING GDP ---
def get_data_with_fallback(df_full, year, country_list=None):
    """Fetches data for selected year. Fills missing GDP/Pop with previous year if needed."""
    df_current = df_full[df_full['year'] == year].copy()
    
    if country_list:
        df_current = df_current[df_current['country'].isin(country_list)]
        
    if df_current['gdp'].isnull().mean() > 0.9:
        fallback_year = 2022
        if year > fallback_year:
            df_fallback = df_full[df_full['year'] == fallback_year][['country', 'gdp', 'population']]
            df_current = df_current.drop(columns=['gdp', 'population'], errors='ignore')
            df_current = pd.merge(df_current, df_fallback, on='country', how='left')
            return df_current, True
    return df_current, False

# --- 2. SIDEBAR ---
st.sidebar.title("Control Panel")

min_year = 1750
max_year = int(df['year'].max())

year_range = st.sidebar.slider(
    "Select Analysis Period", 
    min_year, 
    max_year, 
    (1900, 2022),
    help="Use the sliders to define the start and end years for trend analysis."
)
start_year, selected_year = year_range

all_countries = sorted(df['country'].unique())
default_countries = ["Turkey", "United States", "China", "Germany", "India", "Russia", "Brazil"]
default_countries = [c for c in default_countries if c in all_countries]

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Country Filter")

select_all = st.sidebar.checkbox("Select All Countries")

if select_all:
    selected_countries = all_countries
    st.sidebar.info(f"✅ All {len(all_countries)} countries selected.")
else:
    selected_countries = st.sidebar.multiselect(
        "Compare Countries (Manual Selection)", 
        all_countries, 
        default=default_countries
    )

# Download Button
df_year_download, _ = get_data_with_fallback(df, selected_year)
csv = df_year_download.to_csv(index=False).encode('utf-8')
st.sidebar.markdown("---")
st.sidebar.download_button(
    label=f"📥 Download Data ({selected_year})",
    data=csv,
    file_name=f"co2_data_{selected_year}.csv",
    mime="text/csv",
)

# --- 3. MAIN PAGE & KPI ---
df_year, used_fallback = get_data_with_fallback(df, selected_year)

st.title(f"🌍 Global Emissions Dashboard ({start_year} - {selected_year})")
st.markdown(f"This dashboard analyzes global CO2 emissions from **{start_year} to {selected_year}**, exploring Geographic, Temporal, and Relational dimensions.")

if used_fallback:
    st.warning(f"⚠️ Note: Since economic data for {selected_year} is missing, 2022 GDP data is used for snapshot visualizations.")

st.markdown(f"### 📊 Summary of {selected_year}")
col1, col2, col3, col4 = st.columns(4)

total_co2 = df_year['co2'].sum() / 1000
avg_co2 = df_year['co2_per_capita'].mean()
max_co2_country = df_year.loc[df_year['co2'].idxmax()]
tr_data = df_year[df_year['country'] == 'Turkey']

col1.metric("Global Emissions", f"{total_co2:.1f} Bn Tons")
col2.metric("Avg Per Capita", f"{avg_co2:.2f} Tons")
col3.metric("Top Emitter", max_co2_country['country'], f"{max_co2_country['co2']/1000:.1f} Bn Tons")
if not tr_data.empty:
    col4.metric("Turkey", f"{tr_data.iloc[0]['co2']:.1f} Mn Tons")
else:
    col4.metric("Turkey", "No Data")

st.markdown("---")

# --- 4. TAB STRUCTURE ---
tab1, tab2, tab3 = st.tabs([
    "Global & Structural", 
    "Time & Trends", 
    "Relational & Analytical"
])
# ==============================================================================
# PART 1: GLOBAL AND STRUCTURAL ANALYSIS 
# ==============================================================================
with tab1:    
    col1a, col1b = st.columns([2, 1])
    
    with col1a:
        # --- CHART 1: 3D MAP ---
        st.subheader("1. Global Emission Map")
        
        map_mode = st.radio("Map Mode:", ["Raw CO2 (Animated Period)", "ML: Similarity Groups (K-Means)"], horizontal=True)
        
        if "ML" in map_mode and ml_available:
            st.info(f"🤖 **AI Analysis ({selected_year}):** Countries grouped by CO2, GDP, and Population similarity.")
            
            ml_data = df_year[['iso_code', 'country', 'co2', 'gdp', 'population']].copy()
            
            for col in ['co2', 'gdp', 'population']:
                ml_data[col] = ml_data[col].fillna(ml_data[col].median())
            
            ml_features = np.log1p(ml_data[['co2', 'gdp', 'population']])
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(ml_features)
            
            kmeans = KMeans(n_clusters=4, random_state=42)
            ml_data['Cluster'] = kmeans.fit_predict(scaled_features)
            ml_data['Cluster'] = "Group " + (ml_data['Cluster'] + 1).astype(str)
            
            fig_globe = px.choropleth(
                ml_data,
                locations="iso_code",
                color="Cluster", 
                hover_name="country",
                hover_data=["co2", "gdp", "population"],
                projection="orthographic",
                title=f"Country Clusters (AI - K-Means)",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
        else:
            # ANIMATED MAP
            anim_df = df[(df["year"] >= 1900) & (df["year"] <= 2022) & (df["co2"] > 0)].sort_values("year")
            max_co2 = anim_df["co2"].max()

            fig_globe = px.choropleth(
                anim_df,
                locations="iso_code",
                color="co2",
                hover_name="country",
                projection="orthographic",
                animation_frame="year",
                animation_group="iso_code",
                color_continuous_scale="Reds",
                range_color=[0, max_co2],
                labels={"co2": "CO₂ (Mt)"}
            )

        fig_globe.update_geos(
            showcoastlines=True, coastlinecolor="#333333",
            showcountries=True, countrycolor="Black",
            showocean=True, oceancolor="#eefaff"
        )
        fig_globe.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_globe, use_container_width=True)

    with col1b:
        # --- CHART 2: SUNBURST/TREEMAP ---
        st.subheader("2. Regional Hierarchy")
        chart_type = st.radio("View:", ["Sunburst (Circular)", "Treemap (Rectangular)"], horizontal=True)
        
        df_hier = df_year[df_year['co2'] > 0]
        if not df_hier.empty:
            if "Sunburst" in chart_type:
                fig_hier = px.sunburst(
                    df_hier, path=['continent', 'country'], values='co2',
                    color='continent', color_discrete_sequence=px.colors.qualitative.Set2
                )
            else:
                fig_hier = px.treemap(
                    df_hier, path=['continent', 'country'], values='co2',
                    color='continent', color_discrete_sequence=px.colors.qualitative.Set2
                )
            st.plotly_chart(fig_hier, use_container_width=True)

    # --- CHART 3: SANKEY ---
    st.subheader("3. Energy Source Flows (Sankey)")
    st.caption("Distribution of fossil fuels (Coal, Oil, Gas) by continent.")
    
    source_cols = ['coal_co2', 'oil_co2', 'gas_co2', 'cement_co2', 'flaring_co2']
    df_sankey = df_year.groupby('continent')[source_cols].sum().reset_index()
    
    fuel_sources = ['Coal', 'Oil', 'Gas', 'Cement', 'Flaring']
    continents = df_sankey['continent'].tolist()
    all_nodes = fuel_sources + continents
    
    sources_idx, targets_idx, values_list = [], [], []
    node_colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"] + px.colors.qualitative.Set2
    
    for i, fuel in enumerate(source_cols):
        for j, cont in enumerate(continents):
            val = df_sankey.loc[df_sankey['continent'] == cont, fuel].values[0]
            if val > 1: 
                sources_idx.append(i)
                targets_idx.append(len(fuel_sources) + j)
                values_list.append(val)

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes, color=node_colors[:len(all_nodes)]),
        link=dict(source=sources_idx, target=targets_idx, value=values_list)
    )])
    fig_sankey.update_layout(height=400, font_size=12)
    st.plotly_chart(fig_sankey, use_container_width=True)
