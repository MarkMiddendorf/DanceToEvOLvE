import pandas as pd # type: ignore
import plotly.express as px # type: ignore
import streamlit as st # type: ignore

st.set_page_config(page_title = "DanceToEvOLvE Dashboard", layout = "wide")
st.header("💃 DanceToEvOLvE Dashboard")

df = pd.read_excel(
    io = '/',
    engine = 'openpyxl',
    sheet_name = 'Combined',
    skiprows=0,
    usecols='A:L',
    nrows=1922,
)

# --- SideBar ----
st.sidebar.header("Please Filter Here: ")
grouptype = st.sidebar.multiselect(
    "Select the group:",
    options=df["City"].dropna().unique(),
    default=["Overall"]
)

citySelect = st.sidebar.multiselect(
    "Select the City:",
    options=df["City"].dropna().unique(),
    default=["Overall"]
)

locationSelect = st.sidebar.multiselect(
    "Select the location:",
    options=df["Location"].dropna().unique(),
    default=["Overall"]
)

calcType = st.sidebar.multiselect(
    "Select the Calculation Type:",
    options=df["Calculation_Type"].dropna().unique(),
    default=df["Calculation_Type"].dropna().unique()
)

category = st.sidebar.multiselect(
    "Select the Category:",
    options=df["Category"].dropna().unique(),
    default=["Reg", "Non Reg"]
)

startYear = st.sidebar.multiselect(
    "Select the Start Year:",
    options=df["Year_Start"].dropna().unique(),
    default=[df["Year_Start"].dropna().unique()[0]]  # Use the first available year
)

endYear = st.sidebar.multiselect(
    "Select the End Year:",
    options=df["Year_End"].dropna().unique(),
    default=[df["Year_End"].dropna().unique()[-1]]  # Use the last available year
)

startSeason = st.sidebar.multiselect(
    "Select the Start Season:",
    options=df["Season_Start"].dropna().unique(),
    default=df["Season_Start"].dropna().unique()
)

endSeason = st.sidebar.multiselect(
    "Select the End Season:",
    options=df["Season_End"].dropna().unique(),
    default=df["Season_End"].dropna().unique()
)

startSession = st.sidebar.multiselect(
    "Select the Start Session:",
    options=df["Session_Start"].dropna().unique(),
    default=df["Session_Start"].dropna().unique()
)

endSession = st.sidebar.multiselect(
    "Select the End Session:",
    options=df["Session_End"].dropna().unique(),
    default=df["Session_End"].dropna().unique()
)

# Selection based on sidebar filters
df_selection = df.query(
    "City in @citySelect & Calculation_Type in @calcType & Group in @grouptype & "
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

# Initialize city_dfs
city_dfs = {}

# Query for each category
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
    
    city_dfs[category] = df.query(query_string)[['Session_Start', 'Retention_Rate', 'Location', 'Category']]

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

# Displaying KPIs with st.columns
st.header("Retention Overview")

# Set up columns for each city
cols = st.columns(len(categories))

for i, category in enumerate(categories):
    with cols[i]:
        st.subheader(category)
        st.write(f"**Session Start:** {city_dfs[category].iloc[0]['Session_Start'] if not city_dfs[category].empty else 'N/A'}")
        st.write(f"**Retention Rate:** {city_dfs[category].iloc[0]['Retention_Rate'] if not city_dfs[category].empty else 'N/A'}")

st.markdown("---")

# --- Dynamic Graph ---
st.header(":bar_chart: Dynamic Retention Rate Graph")

if df_selection.empty:
    st.warning("No data available for the selected filters. Please adjust your filters.")
else:
    df_selection["Retention_Rate"] = pd.to_numeric(df_selection["Retention_Rate"], errors='coerce')
    df_selection["Year_Session"] = df_selection["Year_Start"].astype(str) + " - " + df_selection["Session_Start"]
    df_selection["Year_Session"] = pd.Categorical(df_selection["Year_Session"], ordered=True, 
                                                  categories=sorted(df_selection["Year_Session"].unique()))
    df_selection = df_selection.sort_values("Year_Session")

    # --- Dynamic Graph with Markers ---
    fig = px.line(
        df_selection,
        x="Year_Session",
        y="Retention_Rate",
        color="Group",
        markers=True,
        title="Retention Rate Over Time",
        labels={
            "Year_Session": "Year and Session",
            "Retention_Rate": "Retention Rate (%)",
            "Group": "Group"
        }
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title="Year and Session",
        yaxis_title="Retention Rate (%)",
        legend_title="Group",
        template="plotly_white",
        hovermode="x"
    )

    # Show the figure
    st.plotly_chart(fig, use_container_width=True)
