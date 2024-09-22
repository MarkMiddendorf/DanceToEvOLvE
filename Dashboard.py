import pandas as pd # type: ignore
import plotly.express as px # type: ignore
import streamlit as st # type: ignore

st.set_page_config(page_title = "DanceToEvOLvE Dashboard", layout = "wide")
st.header("💃 DanceToEvOLvE Dashboard")

df = pd.read_excel(
    io = '/Users/markmiddendorf/Desktop/INTL BUS/DanceToEvOLve/combined_output9.20.24.xlsx',
    engine = 'openpyxl',
    sheet_name = 'Combined',
    skiprows=0,
    usecols='A:L',
    nrows=1922,
)

# Example: Assuming 'Retention_Rate' values are strings with '%' at the end
#df['Retention_Rate'] = df['Retention_Rate'].str.replace('%', '')  # Remove the '%' sign
#df['Retention_Rate'] = pd.to_numeric(df['Retention_Rate'], errors='coerce')  # Convert to numeric

# --- SideBar ----
st.sidebar.header("Please Filter Here: ")
grouptype = st.sidebar.multiselect(
    "Select the group:",
    options=df["Group"].dropna().unique(),
    default=["Overall"]
)

# Unique cities excluding 'Overall'
uniqueCities = df["City"].dropna().unique().tolist()
citiesNoOverall = [city for city in uniqueCities if city != "Overall"]

citySelect = st.sidebar.multiselect(
    "Select the City:",
    options=uniqueCities,  # Use the filtered cities
    default=["Overall"] # Set the default based on the condition
)

locationSelect = st.sidebar.multiselect(
    "Select the location:",
    options=df["Location"].dropna().unique(),
    default=["Overall"]
)

# Unique values for Calculation_Type
unique_calc_types = df["Calculation_Type"].dropna().unique()
default_calc_type = ['School Year over School Year Retention'] if 'School Year over School Year Retention' in unique_calc_types else []

calcType = st.sidebar.multiselect(
    "Select the Calculation Type:",
    options=unique_calc_types,
    default=default_calc_type
)

# Unique values for Categories
unique_categories = df["Category"].dropna().unique().tolist()
default_categories = ['Chicago', 'Cleveland', 'San Diego']
default_categories = [cat for cat in default_categories if cat in unique_categories]  # Only include if they exist

category = st.sidebar.multiselect(
    "Select the Category:",
    options=unique_categories,
    default=default_categories
)

# Unique values for Year_Start
unique_start_years = df["Year_Start"].dropna().unique()
default_start_year = ['2022-23 School Year'] if '2022-23 School Year' in unique_start_years else []

startYear = st.sidebar.multiselect(
    "Select the Start Year:",
    options=unique_start_years,
    default=default_start_year
)

# Unique values for Year_End
unique_end_years = df["Year_End"].dropna().unique()
default_end_year = ['2023-24 School Year'] if '2023-24 School Year' in unique_end_years else []

endYear = st.sidebar.multiselect(
    "Select the End Year:",
    options=unique_end_years,
    default=default_end_year
)

# Unique values for Season_Start
unique_start_seasons = df["Season_Start"].dropna().unique()
default_start_season = ['SchoolYear/SchoolYear'] if 'SchoolYear/SchoolYear' in unique_start_seasons else []

startSeason = st.sidebar.multiselect(
    "Select the Start Season:",
    options=unique_start_seasons,
    default=default_start_season
)

# Unique values for Season_End
unique_end_seasons = df["Season_End"].dropna().unique()
default_end_season = ['SchoolYear/SchoolYear'] if 'SchoolYear/SchoolYear' in unique_end_seasons else []

endSeason = st.sidebar.multiselect(
    "Select the End Season:",
    options=unique_end_seasons,
    default=default_end_season
)

# Unique values for Session_Start
unique_start_sessions = df["Session_Start"].dropna().unique()
default_start_session = ['SchoolYear/SchoolYear'] if 'SchoolYear/SchoolYear' in unique_start_sessions else []

startSession = st.sidebar.multiselect(
    "Select the Start Session:",
    options=unique_start_sessions,
    default=default_start_session
)

# Unique values for Session_End
unique_end_sessions = df["Session_End"].dropna().unique()
default_end_session = ['SchoolYear/SchoolYear'] if 'SchoolYear/SchoolYear' in unique_end_sessions else []

endSession = st.sidebar.multiselect(
    "Select the End Session:",
    options=unique_end_sessions,
    default=default_end_session
)

# Selection based on sidebar filters
df_selection = df.query(
    "City in @citySelect & Category in @category & Calculation_Type in @calcType & Group in @grouptype & "
    "Year_Start in @startYear & Year_End in @endYear & "
    "Season_Start in @startSeason & Season_End in @endSeason & "
    "Session_Start in @startSession & Session_End in @endSession"
)

# Define the fixed query parameters for each city
group_value = 'Overall'
city_value = 'Overall'
location_value = 'Overall'
calculation_type_value = 'School Year over School Year Retention'
year_start_value = '2022-23 School Year'
year_end_value = '2023-24 School Year'
season_start_value = 'SchoolYear/SchoolYear'
season_end_value = 'SchoolYear/SchoolYear'
session_start_value = 'SchoolYear/SchoolYear'
session_end_value = 'SchoolYear/SchoolYear'
categories = ['Chicago', 'Cleveland', 'San Diego']

# Define the fixed query parameters for each city reg / nonreg
group_value1 = 'City'
city_value1 = ['Chicago', 'Cleveland', 'San Diego']
location_value1 = 'GroupedByCity (use Category)'
calculation_type_value1 = 'School Year over School Year Retention'
year_start_value1 = '2022-23 School Year'
year_end_value1 = '2023-24 School Year'
season_start_value1 = 'SchoolYear/SchoolYear'
season_end_value1 = 'SchoolYear/SchoolYear'
session_start_value1 = 'SchoolYear/SchoolYear'
session_end_value1 = 'SchoolYear/SchoolYear'
categories1 = ['Reg', 'Non Reg']

# Initialize city_dfs and city_category_dfs
city_dfs = {}
city_category_dfs = {category: {} for category in categories1}

# Query for each category (Overall)
for category in categories:
    query_string = (
        f"Group == '{group_value}' & "
        f"City == '{city_value}' & "
        f"Location == '{location_value}' & "
        f"Category == '{category}' & "
        f"Calculation_Type == '{calculation_type_value}' & "
        f"Year_Start == '{year_start_value}' & "
        f"Year_End == '{year_end_value}' & "
        f"Season_Start == '{season_start_value}' & "
        f"Season_End == '{season_end_value}' & "
        f"Session_Start == '{session_start_value}' & "
        f"Session_End == '{session_end_value}'"
    )
    
    # Query the DataFrame and store the result
    city_dfs[category] = df.query(query_string)[['Year_Start', 'Year_End', 'Retention_Rate', 'Location', 'Category']]

# Query for Reg/Non Reg
for category in categories1:
    for city in city_value1:
        # Construct the query string
        query_string1 = (
            f"Group == '{group_value1}' & "
            f"City == '{city}' & "
            f"Location == '{location_value1}' & "
            f"Calculation_Type == '{calculation_type_value1}' & "
            f"Year_Start == '{year_start_value1}' & "
            f"Year_End == '{year_end_value1}' & "
            f"Season_Start == '{season_start_value1}' & "
            f"Season_End == '{season_end_value1}' & "
            f"Session_Start == '{session_start_value1}' & "
            f"Session_End == '{session_end_value1}' & "
            f"Category == '{category}'"
        )

        # Query the DataFrame and store the result
        city_category_dfs[category][city] = df.query(query_string1)[['Year_Start', 'Year_End', 'Retention_Rate', 'Location', 'Category']]

# Convert Retention_Rate to percentage format for all DataFrames
def format_rate(value):
    value = value * 100
    return f"{value:.2f}%" if pd.notnull(value) else "N/A"

def format_retention_rate(df):
    if not df.empty:
        df['Retention_Rate'] = df['Retention_Rate'].apply(format_rate)
    else:
        df = pd.DataFrame({'Retention_Rate': ['N/A']})
    return df

# Apply the formatting to each DataFrame
for category in categories:
    city_dfs[category] = format_retention_rate(city_dfs[category])

for category in categories1:
    for city in city_value1:
        city_category_dfs[category][city] = format_retention_rate(city_category_dfs[category][city])

# Displaying KPIs with st.columns
st.header("Retention Overview")

# Set up columns for each city
cols = st.columns(len(categories))

for i, category in enumerate(categories):
    with cols[i]:
        st.subheader(category)
        st.write(f"**Base Year:** {city_dfs[category].iloc[0]['Year_Start'] if not city_dfs[category].empty else 'N/A'}")
        st.write(f"**Retention Year:** {city_dfs[category].iloc[0]['Year_End'] if not city_dfs[category].empty else 'N/A'}")
        st.write(f"**Retention Rate:** {city_dfs[category].iloc[0]['Retention_Rate'] if not city_dfs[category].empty else 'N/A'}")
        
        # Display Reg and Non Reg retention rates if available
        st.write(f"**Retention Rate Reg:** {city_category_dfs['Reg'][category].iloc[0]['Retention_Rate'] if not city_category_dfs['Reg'][category].empty else 'N/A'}")
        st.write(f"**Retention Rate Non Reg:** {city_category_dfs['Non Reg'][category].iloc[0]['Retention_Rate'] if not city_category_dfs['Non Reg'][category].empty else 'N/A'}")

st.markdown("---")

# --- Dynamic Graph ---
st.header(":bar_chart: Dynamic Retention Rate Graph")

if df_selection.empty:
    st.warning("No data available for the selected filters. Please adjust your filters.")
else:
    df_selection["Retention_Rate"] = pd.to_numeric(df_selection["Retention_Rate"], errors='coerce')
    df_selection["Year_Session"] = df_selection["Year_End"].astype(str) + " - " + df_selection["Session_End"]
    df_selection["Year_Session"] = pd.Categorical(df_selection["Year_Session"], ordered=True, 
                                                  categories=sorted(df_selection["Year_Session"].unique()))
    df_selection = df_selection.sort_values("Year_Session")

    # --- Dynamic Graph with Markers ---
    fig = px.line(
        df_selection,
        x="Year_Session",
        y="Retention_Rate",
        color="Category",
        markers=True,
        title="Retention Rate Over Time",
        labels={
            "Year_Session": "Year and Session",
            "Retention_Rate": "Retention Rate (%)",
            "Category": "Category"
        }
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title="Year and Session",
        yaxis_title="Retention Rate (%)",
        legend_title="Category",
        template="plotly_white",
        hovermode="x"
    )

    # Show the figure
    st.plotly_chart(fig, use_container_width=True)
