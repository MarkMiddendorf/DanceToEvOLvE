pip install plotly-express

import pandas as pd # type: ignore
import streamlit as st # type: ignore
import plotly.express as px # type: ignore

st.set_page_config(page_title = "💃 DanceToEvOLvE Dashboard", layout = "wide")
st.header("💃 DanceToEvOLvE Dashboard")

df = pd.read_excel(
    io = '/Users/markmiddendorf/Desktop/INTL BUS/DanceToEvOLve/combined_output2.xlsx',
    engine = 'openpyxl',
    sheet_name = 'Combined',
    skiprows=0,
    usecols='A:H',
    nrows=2952,
)

# --- SideBar ----
st.sidebar.header("Please Filter Here: ")
city = st.sidebar.multiselect(
    "Select the City:",
    options = df["City"].dropna().unique(),
    default = ["Overall"]
)

calcType = st.sidebar.multiselect(
    "Select the Calculation Type:",
    options = df["Calculation_Type"].dropna().unique(),
    default = df["Calculation_Type"].dropna().unique()
)

grouptype = st.sidebar.multiselect(
    "Select the Group:",
    options = df["Group"].dropna().unique(),
    default = ["Chicago", "Cleveland", "San Diego"]
)

startYear = st.sidebar.multiselect(
    "Select the Start Year:",
    options = df["Year_Start"].dropna().unique(),
    default = df["Year_Start"].dropna().unique()
)

endYear = st.sidebar.multiselect(
    "Select the End Year:",
    options = df["Year_End"].dropna().unique(),
    default = df["Year_End"].dropna().unique()
)

startSession = st.sidebar.multiselect(
    "Select the Start Session:",
    options = df["Session_Start"].dropna().unique(),
    default = df["Session_Start"].dropna().unique()
)

endSession = st.sidebar.multiselect(
    "Select the End Session:",
    options = df["Session_End"].dropna().unique(),
    default = df["Session_End"].dropna().unique()
)



df_selection = df.query(
    "City == @city & Calculation_Type == @calcType & Group == @grouptype & Year_Start == @startYear & Year_End == @endYear & Session_Start == @startSession & Session_End == @endSession"
)


# Queries for each city
df_chicago = df.query(
    "City == 'Overall' & Calculation_Type == 'School Year over School Year Retention' & Group == 'Chicago' & Year_Start == 2022 & Year_End == 2023 & Session_Start == '2022-2023 School Year' & Session_End == '2022-2023 School Year'"
)[["Session_Start", "Retention_Rate"]]

df_sandiego = df.query(
    "City == 'Overall' & Calculation_Type == 'School Year over School Year Retention' & Group == 'San Diego' & Year_Start == 2022 & Year_End == 2023 & Session_Start == '2022-2023 School Year' & Session_End == '2022-2023 School Year'"
)[["Session_Start", "Retention_Rate"]]

df_cleveland = df.query(
    "City == 'Overall' & Calculation_Type == 'School Year over School Year Retention' & Group == 'Cleveland' & Year_Start == 2022 & Year_End == 2023 & Session_Start == '2022-2023 School Year' & Session_End == '2022-2023 School Year'"
)[["Session_Start", "Retention_Rate"]]

df_chicagoReg = df.query(
    "City == 'Chicago' & Calculation_Type == 'School Year over School Year Retention' & Group == 'Chicago_Reg' & Year_Start == 2022 & Year_End == 2023 & Session_Start == '2022-2023 School Year' & Session_End == '2022-2023 School Year'"
)[["Session_Start", "Retention_Rate"]]

df_chicagoNonReg = df.query(
    "City == 'Chicago' & Calculation_Type == 'School Year over School Year Retention' & Group == 'Chicago_Non Reg' & Year_Start == 2022 & Year_End == 2023 & Session_Start == '2022-2023 School Year' & Session_End == '2022-2023 School Year'"
)[["Session_Start", "Retention_Rate"]]

df_clevelandReg = df.query(
    "City == 'Cleveland' & Calculation_Type == 'School Year over School Year Retention' & Group == 'Cleveland_Reg' & Year_Start == 2022 & Year_End == 2023 & Session_Start == '2022-2023 School Year' & Session_End == '2022-2023 School Year'"
)[["Session_Start", "Retention_Rate"]]

df_clevelandNonReg = df.query(
    "City == 'Cleveland' & Calculation_Type == 'School Year over School Year Retention' & Group == 'Cleveland_Non Reg' & Year_Start == 2022 & Year_End == 2023 & Session_Start == '2022-2023 School Year' & Session_End == '2022-2023 School Year'"
)[["Session_Start", "Retention_Rate"]]

df_sanDiegoReg = df.query(
    "City == 'San Diego' & Calculation_Type == 'School Year over School Year Retention' & Group == 'San Diego_Reg' & Year_Start == 2022 & Year_End == 2023 & Session_Start == '2022-2023 School Year' & Session_End == '2022-2023 School Year'"
)[["Session_Start", "Retention_Rate"]]

df_sanDiegoNonReg = df.query(
    "City == 'San Diego' & Calculation_Type == 'School Year over School Year Retention' & Group == 'San Diego_Non Reg' & Year_Start == 2022 & Year_End == 2023 & Session_Start == '2022-2023 School Year' & Session_End == '2022-2023 School Year'"
)[["Session_Start", "Retention_Rate"]]

# Convert Retention_Rate to percentage format for all DataFrames
def format_rate(value):
    value = value * 100
    return f"{value:.2f}%" if pd.notnull(value) else "N/A"

def format_retention_rate(df):
    if not df.empty:
        df['Retention_Rate'] = df['Retention_Rate'].apply(format_rate)
    else:
        df = pd.DataFrame({'Retention_Rate': ['N/A']})
    return df  # Return the DataFrame after modifications

# Apply the formatting to each DataFrame
df_chicago = format_retention_rate(df_chicago)
df_sandiego = format_retention_rate(df_sandiego)
df_cleveland = format_retention_rate(df_cleveland)
df_chicagoReg = format_retention_rate(df_chicagoReg)
df_sanDiegoReg = format_retention_rate(df_sanDiegoReg)
df_clevelandReg = format_retention_rate(df_clevelandReg)
df_chicagoNonReg = format_retention_rate(df_chicagoNonReg)
df_sanDiegoNonReg = format_retention_rate(df_sanDiegoNonReg)
df_clevelandNonReg = format_retention_rate(df_clevelandNonReg)


# Displaying KPIs with st.columns
st.header("Retention Overview")

# Set up columns for each city
col1, col2, col3 = st.columns(3)

# Chicago KPI
with col1:
    st.subheader("Chicago")
    st.write(f"**Session Start:** {df_chicago.iloc[0]['Session_Start']}")
    st.write(f"**Retention Rate:** {df_chicago.iloc[0]['Retention_Rate']}")
    st.write(f"**Retention Rate Reg:** {df_chicagoReg.iloc[0]['Retention_Rate']}")
    st.write(f"**Retention Rate Non Reg:** {df_chicagoNonReg.iloc[0]['Retention_Rate']}")

# San Diego KPI
with col2:
    st.subheader("San Diego")
    st.write(f"**Session Start:** {df_sandiego.iloc[0]['Session_Start']}")
    st.write(f"**Retention Rate:** {df_sandiego.iloc[0]['Retention_Rate']}")
    st.write(f"**Retention Rate Reg:** {df_sanDiegoReg.iloc[0]['Retention_Rate']}")
    st.write(f"**Retention Rate Non Reg:** {df_sanDiegoNonReg.iloc[0]['Retention_Rate']}")

# Cleveland KPI
with col3:
    st.subheader("Cleveland")
    st.write(f"**Session Start:** {df_cleveland.iloc[0]['Session_Start']}")
    st.write(f"**Retention Rate:** {df_cleveland.iloc[0]['Retention_Rate']}")
    st.write(f"**Retention Rate Reg:** {df_clevelandReg.iloc[0]['Retention_Rate']}")
    st.write(f"**Retention Rate Non Reg:** {df_clevelandNonReg.iloc[0]['Retention_Rate']}")

st.markdown("---")

# --- Dynamic Graph ---
st.header(":bar_chart: Dynamic Retention Rate Graph")

# --- Check for empty dataframe ---
if df_selection.empty:
    st.warning("No data available for the selected filters. Please adjust your filters.")
else:
    # Convert Retention_Rate to numeric if it's not already
    df_selection["Retention_Rate"] = pd.to_numeric(df_selection["Retention_Rate"], errors='coerce')

    # Create a combined column of Year and Session for the x-axis
    df_selection["Year_Session"] = df_selection["Year_Start"].astype(str) + " - " + df_selection["Session_Start"]

    # Ensure the Year_Session is treated as a categorical type to preserve order
    df_selection["Year_Session"] = pd.Categorical(df_selection["Year_Session"], ordered=True, 
                                                  categories=sorted(df_selection["Year_Session"].unique()))

    # Sort data by Year_Session to ensure proper line plotting
    df_selection = df_selection.sort_values("Year_Session")

    # --- Dynamic Graph with Markers ---
    fig = px.line(
        df_selection,
        x="Year_Session",
        y="Retention_Rate",
        color="Group",
        markers=True,  # Add markers to each data point
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
        hovermode="x"  # Better hover interaction for line charts
    )

    # Show the figure
    st.plotly_chart(fig, use_container_width=True)
