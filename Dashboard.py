import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import streamlit as st  # type: ignore
from streamlit_gsheets import GSheetsConnection

# Set page configuration
st.set_page_config(page_title="DanceToEvOLvE Dashboard", layout="wide")
st.header("💃 DanceToEvOLvE Dashboard")

# Establish connection to Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

# Read data from the Google Sheets connection
data = conn.read(worksheet="Sheet1", usecols=list(range(12)))
df = data

# --- SideBar ----
st.sidebar.header("Please Filter Here:")

# Sidebar filters
grouptype = st.sidebar.multiselect(
    "Select the group:", options=df["Group"].dropna().unique(), default=["Overall"]
)

# Unique cities excluding 'Overall'
uniqueCities = df["City"].dropna().unique().tolist()
citiesNoOverall = [city for city in uniqueCities if city != "Overall"]

citySelect = st.sidebar.multiselect(
    "Select the City:", options=uniqueCities, default=["Overall"]
)

locationSelect = st.sidebar.multiselect(
    "Select the location:", options=df["Location"].dropna().unique(), default=["Overall"]
)

# Unique values for Calculation_Type
unique_calc_types = df["Calculation_Type"].dropna().unique()
default_calc_type = (
    ["School Year over School Year Retention"]
    if "School Year over School Year Retention" in unique_calc_types
    else []
)

calcType = st.sidebar.multiselect(
    "Select the Calculation Type:", options=unique_calc_types, default=default_calc_type
)

# Unique values for Categories
unique_categories = df["Category"].dropna().unique().tolist()
default_categories = [
    cat for cat in ["Chicago", "Cleveland", "San Diego"] if cat in unique_categories
]

category = st.sidebar.multiselect(
    "Select the Category:", options=unique_categories, default=default_categories
)

# Unique values for Year_Start and Year_End
startYear = st.sidebar.multiselect(
    "Select the Start Year:",
    options=df["Year_Start"].dropna().unique(),
    default=["2022-23 School Year"],
)

endYear = st.sidebar.multiselect(
    "Select the End Year:",
    options=df["Year_End"].dropna().unique(),
    default=["2023-24 School Year"],
)

# Unique values for Season and Session
startSeason = st.sidebar.multiselect(
    "Select the Start Season:",
    options=df["Season_Start"].dropna().unique(),
    default=["SchoolYear/SchoolYear"],
)

endSeason = st.sidebar.multiselect(
    "Select the End Season:",
    options=df["Season_End"].dropna().unique(),
    default=["SchoolYear/SchoolYear"],
)

startSession = st.sidebar.multiselect(
    "Select the Start Session:",
    options=df["Session_Start"].dropna().unique(),
    default=["SchoolYear/SchoolYear"],
)

endSession = st.sidebar.multiselect(
    "Select the End Session:",
    options=df["Session_End"].dropna().unique(),
    default=["SchoolYear/SchoolYear"],
)

# Selection based on sidebar filters
df_selection = df.query(
    "City in @citySelect & Category in @category & Calculation_Type in @calcType & Group in @grouptype & "
    "Year_Start in @startYear & Year_End in @endYear & "
    "Season_Start in @startSeason & Season_End in @endSeason & "
    "Session_Start in @startSession & Session_End in @endSession"
)

# Define the fixed query parameters
group_value = 'Overall'
categories = ['Chicago', 'Cleveland', 'San Diego']
group_value1 = 'City'
categories1 = ['Reg', 'Non Reg']
city_value1 = ['Chicago', 'Cleveland', 'San Diego']

# Initialize city_dfs and city_category_dfs
city_dfs = {}
city_category_dfs = {category: {} for category in categories1}

# Query for each category (Overall)
for category in categories:
    query_string = (
        f"Group == '{group_value}' & "
        f"City == 'Overall' & "
        f"Location == 'Overall' & "
        f"Category == '{category}' & "
        f"Calculation_Type == 'School Year over School Year Retention' & "
        f"Year_Start == '2022-23 School Year' & "
        f"Year_End == '2023-24 School Year' & "
        f"Season_Start == 'SchoolYear/SchoolYear' & "
        f"Season_End == 'SchoolYear/SchoolYear' & "
        f"Session_Start == 'SchoolYear/SchoolYear' & "
        f"Session_End == 'SchoolYear/SchoolYear'"
    )
    city_dfs[category] = df.query(query_string)[
        ["Year_Start", "Year_End", "Retention_Rate", "Location", "Category"]
    ]

# Query for Reg/Non Reg
for category in categories1:
    for city in city_value1:
        query_string1 = (
            f"Group == '{group_value1}' & "
            f"City == '{city}' & "
            f"Location == 'GroupedByCity (use Category)' & "
            f"Calculation_Type == 'School Year over School Year Retention' & "
            f"Year_Start == '2022-23 School Year' & "
            f"Year_End == '2023-24 School Year' & "
            f"Season_Start == 'SchoolYear/SchoolYear' & "
            f"Season_End == 'SchoolYear/SchoolYear' & "
            f"Session_Start == 'SchoolYear/SchoolYear' & "
            f"Session_End == 'SchoolYear/SchoolYear' & "
            f"Category == '{category}'"
        )
        city_category_dfs[category][city] = df.query(query_string1)[
            ["Year_Start", "Year_End", "Retention_Rate", "Location", "Category"]
        ]

# Convert Retention_Rate to percentage format for all DataFrames
def format_rate(value):
    try:
        value = float(value) * 100
        return f"{value:.2f}%" if pd.notnull(value) else "N/A"
    except (ValueError, TypeError):
        return "N/A"

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
cols = st.columns(len(categories))

for i, category in enumerate(categories):
    with cols[i]:
        st.subheader(category)
        st.write(f"**Base Year:** {city_dfs[category].iloc[0]['Year_Start'] if not city_dfs[category].empty else 'N/A'}")
        st.write(f"**Retention Year:** {city_dfs[category].iloc[0]['Year_End'] if not city_dfs[category].empty else 'N/A'}")
        st.write(f"**Retention Rate:** {city_dfs[category].iloc[0]['Retention_Rate'] if not city_dfs[category].empty else 'N/A'}")
        st.write(f"**Retention Rate Reg:** {city_category_dfs['Reg'][category].iloc[0]['Retention_Rate'] if not city_category_dfs['Reg'][category].empty else 'N/A'}")
        st.write(f"**Retention Rate Non Reg:** {city_category_dfs['Non Reg'][category].iloc[0]['Retention_Rate'] if not city_category_dfs['Non Reg'][category].empty else 'N/A'}")

st.markdown("---")

# --- Dynamic Graph ---
st.header(":bar_chart: Dynamic Retention Rate Graph")

# Check if the filtered DataFrame is empty
if df_selection.empty:
    st.warning("No data available for the selected filters. Please adjust your filters.")
else:
    # Convert Retention_Rate to numeric, handling errors by coercing invalid values to NaN
    df_selection["Retention_Rate"] = pd.to_numeric(df_selection["Retention_Rate"], errors='coerce')

    # Combine Year_End and Session_End to create Year_Session for the x-axis
    df_selection["Year_Session"] = df_selection["Year_End"].astype(str) + " - " + df_selection["Session_End"]

    # Ensure Year_Session is sorted without converting it to a categorical variable
    unique_sessions = sorted(df_selection["Year_Session"].dropna().unique())
    df_selection["Year_Session"] = pd.Series(pd.Categorical(df_selection["Year_Session"], categories=unique_sessions, ordered=True)).astype(str)


    # Sort the DataFrame by the Year_Session column to maintain chronological order
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

    # Customize the layout of the plot for improved readability and visual appeal
    fig.update_layout(
        xaxis_title="Year and Session",
        yaxis_title="Retention Rate (%)",
        legend_title="Category",
        template="plotly_white",
        hovermode="x"
    )

    # Display the figure in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)

