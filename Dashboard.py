import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px
import plotly.graph_objects as go  # Import Plotly graph objects

# Set up Streamlit page config
st.set_page_config(layout="wide")

# Define the scope for accessing Google Sheets
scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

# Load credentials from Streamlit secrets
creds_dict = {
    "type": st.secrets["type"],
    "project_id": st.secrets["project_id"],
    "private_key_id": st.secrets["private_key_id"],
    "private_key": st.secrets["private_key"].replace("\\n", "\n"),  # replace escaped newlines
    "client_email": st.secrets["client_email"],
    "client_id": st.secrets["client_id"],
    "auth_uri": st.secrets["auth_uri"],
    "token_uri": st.secrets["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["client_x509_cert_url"]
}

# Authorize the client using credentials
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)

# Open Google Sheet by name or URL
sheet = client.open("DanceToEvolve_Data").worksheet("Data")

# Get data from Google Sheet
data = sheet.get_all_records()

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# select all for filters
def select_all_option_expander(label, options, sort_order='alphabetical'):
    if sort_order == 'alphabetical':
        options = sorted(options)
    elif sort_order == 'numerical':
        try:
            options = sorted(options, key=lambda x: float(x))
        except ValueError:
            options = sorted(options)

    with st.expander(f"Filter {label}", expanded=False):
        all_selected = st.checkbox(f"Select All {label}", value=True, key=f"{label}-select-all")
        if all_selected:
            selected = st.multiselect(f"Select {label}:", options, default=options, key=f"{label}-multiselect")
        else:
            selected = st.multiselect(f"Select {label}:", options, key=f"{label}-multiselect")
    return selected

st.markdown(
    """
    <style>
    .stMultiSelect [role="listbox"] {
        max-height: 50;  /* Adjust the height of the filters */
        overflow-y: auto;  /* Enable vertical scrolling */
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def addSchoolYear(df):
    # School year definition:
    # Summer 1 -> Summer 2 -> Fall 1 -> Fall 2 -> Winter 1 -> Winter 2 -> Spring 1 -> Spring 2
    school_year_order = [
        ('Summer', 1), ('Summer', 2),('Fall', 1), ('Fall', 2), ('Winter', 1), ('Winter', 2),
        ('Spring', 1), ('Spring', 2)
    ]
    
    # Create a mapping of periods to school years
    school_year_mapping = {}
    
    years = sorted(df['Year'].unique())  # Loop through years
    
    for year_start in years:
        # Define start periods for the current school year
        start_periods = [
            (year_start, 'Summer', 1), (year_start, 'Summer', 2),  # Summer in year_start
            (year_start, 'Fall', 1), (year_start, 'Fall', 2),  # Fall in year_start
            (year_start + 1, 'Winter', 1), (year_start + 1, 'Winter', 2),  # Winter in year_start + 1
            (year_start + 1, 'Spring', 1), (year_start + 1, 'Spring', 2)   # Spring in year_start + 1
        ]
        
        # Map start periods to school year
        school_year_mapping.update({
            (year_start, 'Summer', 1): f"{year_start}",
            (year_start, 'Summer', 2): f"{year_start}",
            (year_start, 'Fall', 1): f"{year_start}",
            (year_start, 'Fall', 2): f"{year_start}",
            (year_start + 1, 'Winter', 1): f"{year_start}",
            (year_start + 1, 'Winter', 2): f"{year_start}",
            (year_start + 1, 'Spring', 1): f"{year_start}",
            (year_start + 1, 'Spring', 2): f"{year_start}"
        })

    # Create a new column in the DataFrame for the school year
    df['School Year'] = df.apply(lambda row: school_year_mapping.get((row['Year'], row['Season'], row['Session'])), axis=1)

    return df

# Add new school year 
df = addSchoolYear(df)

# Convert School Year and Session to integers (if they are not strings)
df['School Year'] = df['School Year'].astype(int)
df['Session'] = df['Session'].astype(int)

# Convert Season to string (if needed)
df['Season'] = df['Season'].astype(str)

# Order Seasons
season_order = {'Summer': 1, 'Fall': 2, 'Winter': 3, 'Spring': 4}
df['Season_Order'] = df['Season'].map(season_order)

# Create a sorting key
df['Sort_Key'] = df['School Year'] * 100 + df['Season_Order'] * 10 + df['Session']

# Create X-axis labels BUT CHANGED TO SCHOOL YEAR
df['Year_Season_Session'] = df['School Year'].astype(str) + ' ' + df['Season'] + ' ' + df['Session'].astype(str)

# Sort the DataFrame
df = df.sort_values('Sort_Key')

# School Year, Season, and Session filters
col_school_year, col_season, col_session = st.columns(3)

with col_school_year:
    selected_school_years = st.multiselect('School Year', df['School Year'].unique(), df['School Year'].unique(), format_func=str)

with col_season:
    selected_seasons = st.multiselect('Season', df['Season'].unique(), df['Season'].unique(), format_func=str)

with col_session:
    selected_sessions = st.multiselect('Session', df['Session'].unique(), df['Session'].unique(), format_func=str)

# Class, Location, Teacher, Age, and Reg/NonReg Filters
st.markdown("<h5 style='text-align: left;'>Additional Filters</h5>", unsafe_allow_html=True)

# Filters formatting
col_city, col_class, col_location, col_teacher, col_age, col_reg_nonreg = st.columns(6)

with col_city:
    selected_cities = select_all_option_expander('City', df['City'].unique(), sort_order='alphabetical')

# Filter the DataFrame to show only locations tied to the selected cities
filtered_locations = df[df['City'].isin(selected_cities)]['Location'].unique()

# Now, display only the locations tied to the selected cities
with col_location:
    selected_locations = select_all_option_expander('Location', filtered_locations, sort_order='alphabetical')

with col_class:
    selected_classes = select_all_option_expander('Class', df['Class'].unique(), sort_order='alphabetical')

with col_teacher:
    selected_teachers = select_all_option_expander('Teacher', df['Teacher'].unique(), sort_order='alphabetical')

with col_age:
    selected_ages = select_all_option_expander('Age', df['Age'].unique(), sort_order='numerical')

with col_reg_nonreg:
    selected_reg_nonreg = select_all_option_expander('Reg/NonReg', df['Reg/NonReg'].unique(), sort_order='alphabetical')

# Apply filters
filtered_df = df[(df['Class'].isin(selected_classes)) &
                 (df['Location'].isin(selected_locations)) &
                 (df['Teacher'].isin(selected_teachers)) &
                 (df['Age'].isin(selected_ages)) &
                 (df['Reg/NonReg'].isin(selected_reg_nonreg)) &
                 (df['School Year'].isin(selected_school_years)) &
                 (df['Season'].isin(selected_seasons)) &
                 (df['Session'].isin(selected_sessions))]

# Initialize total_dancers
total_dancers = 0 

# DANCER ENROLLMENT GRAPH
st.markdown("<h5 style='text-align: left;'></h5>", unsafe_allow_html=True)
if not filtered_df.empty:
    grouped_df = filtered_df.groupby('Year_Season_Session').agg({'DancerID': 'count'}).reset_index()
    grouped_df.rename(columns={'DancerID': 'Number of Dancers'}, inplace=True)
    grouped_df = grouped_df.merge(df[['Year_Season_Session', 'Sort_Key']].drop_duplicates(), on='Year_Season_Session')
    grouped_df = grouped_df.sort_values('Sort_Key')
    total_dancers = grouped_df['Number of Dancers'].sum()

    # Plotting  dynamic graph
    fig = px.line(grouped_df, x='Year_Season_Session', y='Number of Dancers', markers=True, 
                  title='Dancer Enrollment')
    
    fig.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                      line=dict(width=3, color='steelblue'))

    fig.update_layout(
        xaxis_title="Year-Season-Session",
        yaxis_title="Dancers Enrolled",
        xaxis_tickangle=-45,
        template='plotly_white',
        font=dict(size=14, color='black'),
        title_font=dict(size=24, color='white'),
        title_x=0.5,
        xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
        yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )

    for i, row in grouped_df.iterrows():
        fig.add_annotation(
            x=row['Year_Season_Session'],
            y=row['Number of Dancers'],
            text=str(row['Number of Dancers']),
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            xshift=10,
            font=dict(color='white', weight='bold')
        )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Total Enrollment: {total_dancers}</h5>", unsafe_allow_html=True)
else:
    st.warning("No data available for the selected filters.")

# DANCER ACQUISITION GRAPH
st.markdown("<h5 style='text-align: left;'></h5>", unsafe_allow_html=True)
if not filtered_df.empty:
    df_sorted = df.sort_values('Sort_Key')
    seen_dancers = set()
    newly_acquired = []
    for _, row in df_sorted.iterrows():
        session_id = row['Year_Season_Session']
        dancer_id = row['DancerID']
        if dancer_id not in seen_dancers:
            newly_acquired.append({'Year_Season_Session': session_id, 'DancerID': dancer_id, 
                                   'Class': row['Class'], 'Location': row['Location'], 
                                   'Teacher': row['Teacher'], 'Age': row['Age'], 
                                   'Reg/NonReg': row['Reg/NonReg']})
            seen_dancers.add(dancer_id)

    acquired_df = pd.DataFrame(newly_acquired)
    acquired_filtered_df = acquired_df[
        (acquired_df['Class'].isin(selected_classes)) &
        (acquired_df['Location'].isin(selected_locations)) &
        (acquired_df['Teacher'].isin(selected_teachers)) &
        (acquired_df['Age'].isin(selected_ages)) &
        (acquired_df['Reg/NonReg'].isin(selected_reg_nonreg)) &
        (acquired_df['Year_Season_Session'].isin(filtered_df['Year_Season_Session']))]

    acquired_grouped_filtered_df = acquired_filtered_df.groupby('Year_Season_Session').agg({'DancerID': 'count'}).reset_index()
    acquired_grouped_filtered_df.rename(columns={'DancerID': 'Newly Acquired Students'}, inplace=True)
    acquired_grouped_filtered_df = acquired_grouped_filtered_df.merge(df[['Year_Season_Session', 'Sort_Key']].drop_duplicates(), on='Year_Season_Session')
    acquired_grouped_filtered_df = acquired_grouped_filtered_df.sort_values('Sort_Key')

    # Plot graph
    fig_acquired_filtered = px.line(acquired_grouped_filtered_df, x='Year_Season_Session', y='Newly Acquired Students', markers=True, 
                                    title='Dancer Acquisition')
    
    fig_acquired_filtered.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                                        line=dict(width=3, color='mediumseagreen'))

    fig_acquired_filtered.update_layout(
        xaxis_title="Year-Season-Session",
        yaxis_title="Newly Acquired Students",
        xaxis_tickangle=-45,
        template='plotly_white',
        font=dict(size=14, color='black'),
        title_font=dict(size=24, color='white'),
        title_x=0.5,
        xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
        yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )

    for i, row in acquired_grouped_filtered_df.iterrows():
        fig_acquired_filtered.add_annotation(
            x=row['Year_Season_Session'],
            y=row['Newly Acquired Students'],
            text=str(row['Newly Acquired Students']),
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            xshift=10,
            font=dict(color='white', weight='bold')
        )

    st.plotly_chart(fig_acquired_filtered, use_container_width=True)
else:
    st.warning("No data available for newly acquired students based on the selected filters.")

if 'acquired_grouped_filtered_df' in globals() and not acquired_grouped_filtered_df.empty:
    # Calculate total acquired students
    total_acquired_students = acquired_grouped_filtered_df['Newly Acquired Students'].sum()
else:
    total_acquired_students = 0

# Calculate the Acquisition Ratio
if total_dancers > 0:
    acquisition_ratio = (total_acquired_students / total_dancers) * 100
else:
    acquisition_ratio = 0

# Display the total acquired students and Acquisition Ratio side by side
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Total Acquisition: {total_acquired_students}</h5>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<h5 style='text-align: left; margin-left: 0px;'>Acquisition Ratio: {acquisition_ratio:.2f}%</h5>", unsafe_allow_html=True)

### RETENTION

#Retention Calculations
# Function to classify students based on class attendance for all years
def classify_students_by_attendance(df):

    # Set custom thresholds for each type
    thresholds = {
        'Type_8': 8,
        'Type_7': 7,
        'Type_6': 6, 
        'Type_5': 5,  
        'Type_4': 4,
        'Type_3': 3,
        'Type_2': 2,
        'Type_1': 1,
        'Type_0': 0,
    }

    # Create a DataFrame to hold the attendance counts
    attendance_counts = df.groupby(['School Year', 'Season', 'Session', 'DancerID']).size().reset_index(name='Count')
    attendance_counts['School Year'] = attendance_counts['School Year'].astype(int)


    # Loop through each school year in the DataFrame
    years = sorted(df['School Year'].unique())
    years = [int(year) for year in years]

    for year in years:
        # Get the previous year's data
        previous_year = year - 1
        previous_year = int(previous_year)  # Convert to integer
        
        # If there is a previous year, then
        if previous_year in years:
            previous_year_attendance = attendance_counts[attendance_counts['School Year'] == previous_year].copy()

            # Count occurrences of each DancerID
            dancer_counts = previous_year_attendance.groupby('DancerID')['Count'].sum().reset_index(name='Attendance Count')

            # Classify students based on attendance count
            dancer_counts.loc[:, 'Type'] = dancer_counts['Attendance Count'].apply(lambda x: classify_student(x, thresholds))

            # Optionally, you can merge this back to your original DataFrame if needed
            df = df.merge(dancer_counts[['DancerID', 'Type']], on='DancerID', how='left')

            # Rename the 'Type' column after merging
            df.rename(columns={'Type': f'Type_{previous_year}'}, inplace=True)

    return df  # Return the DataFrame with classifications

# Function to classify based on attendance thresholds
def classify_student(attendance, thresholds):
    if attendance >= thresholds['Type_8']:
        return 'Type 8'
    elif attendance == thresholds['Type_7']:
        return 'Type 7'
    elif attendance == thresholds['Type_6']:
        return 'Type 6'
    elif attendance == thresholds['Type_5']:
        return 'Type 5'
    elif attendance == thresholds['Type_4']:
        return 'Type 4'
    elif attendance == thresholds['Type_3']:
        return 'Type 3'
    elif attendance == thresholds['Type_2']:
        return 'Type 2'
    elif attendance == thresholds['Type_1']:
        return 'Type 1'
    else:
        return 'Type 0'

def calculate_retention(df, group_by_cols, start_periods, end_periods):
    # Filter DataFrame for start and end periods based on School Year
    group_start = df[df['School Year'].isin(start_periods)].groupby(group_by_cols, observed=True)['DancerID'].apply(set)
    group_end = df[df['School Year'].isin(end_periods)].groupby(group_by_cols, observed=True)['DancerID'].apply(set)

    retention_results = {}
    
    for group in group_start.index:
        try:
            if group in group_end:
                start_count = len(group_start[group])  # Total unique students in the starting year (for this group)
                end_count = len(group_end[group])  # Total unique students in the current year (for this group)
                retained_students = group_start[group].intersection(group_end[group])  # Unique students retained
                retained_count = len(retained_students)  # Count of unique retained students

                # Ensure we're filtering by the correct group values (handling lists)
                group_condition = True
                for col, value in zip(group_by_cols, group if isinstance(group, tuple) else [group]):
                    group_condition &= df[col] == value

                # Filter for retained students and total students for the start period within the same group
                retained_student_types = df[(df['DancerID'].isin(retained_students)) & (df['School Year'].isin(end_periods)) & group_condition]
                total_student_types = df[(df['School Year'].isin(start_periods)) & group_condition]  # Total students for this group
                total_student_typesEnd = df[(df['School Year'].isin(end_periods)) & group_condition]  # Total students for this group
                     
                # Count unique student IDs for each type (retained students)
                retained_type_counts = retained_student_types.groupby(f'Type_{start_periods[0]}')['DancerID'].nunique()
                # Count unique student IDs for each type (retained students)
                retained_type_countsCurrent = retained_student_types.groupby(f'Type_{end_periods[0]}')['DancerID'].nunique()

                # Count unique student IDs for each type (total students in start period)
                total_type_counts = total_student_types.groupby(f'Type_{start_periods[0]}')['DancerID'].nunique()
                total_type_countsEnd = total_student_typesEnd.groupby(f'Type_{end_periods[0]}')['DancerID'].nunique()

                # Convert to dictionaries for easier access
                retained_type_counts_dict = retained_type_counts.to_dict()
                retained_type_countsCurrent_dict = retained_type_countsCurrent.to_dict()
                total_type_counts_dict = total_type_counts.to_dict()
                total_type_countsEnd_dict = total_type_countsEnd.to_dict()
                     
                # Calculate retention rate
                retention_rate = retained_count / start_count if start_count > 0 else 0

                # Store the results, including type counts for retained and total students
                retention_results[group] = {
                    'Total Students in Prior Year': start_count,
                    'Total Students in Current Year': end_count,
                    'Total Students Retained': retained_count,
                    'Retention Rate': retention_rate,
                }

                # Add type counts for retained and total students to retention results
                for student_type in ['8', '7', '6', '5', '4', '3', '2', '1']:
                    retention_results[group][f'Type_{student_type} Retained Count Previous'] = retained_type_counts_dict.get(f'Type {student_type}', 0)
                    retention_results[group][f'Type_{student_type} Retained Count Current'] = retained_type_countsCurrent_dict.get(f'Type {student_type}', 0)
                    retention_results[group][f'Type_{student_type} Total Count Previous Year'] = total_type_counts_dict.get(f'Type {student_type}', 0)
                    retention_results[group][f'Type_{student_type} Total Count Current Year'] = total_type_countsEnd_dict.get(f'Type {student_type}', 0)
        
        except Exception as e:
            print(f"Error processing group {group} for school years {start_periods[0]}-{end_periods[0]}: {e}")
    
    return retention_results

# Function to calculate school year over school year retention
def calculate_school_year_retention(df):
    retention_results = []
    
    years = sorted(df['School Year'].dropna().unique())  # Loop through school years

    group_by_combinations = [
        ['City'],
        ['Location'],
        ['Class'],
        ['Teacher'],
        ['Reg/Non Reg'],
    ]

    for year_start, year_end in zip(years[:-1], years[1:]):
        # Define start and end periods based on School Year
        start_periods = [year_start]  # Use only the current school year
        end_periods = [year_end]       # Use only the next school year

        for group_by in group_by_combinations:
            try:
                retention = calculate_retention(df, group_by, start_periods, end_periods)
                for group, data in retention.items():
                    retention_results.append({
                        'Group': group,
                        'School Year Start': str(year_end),
                        'Total Unique Students in Prior Year': data['Total Students in Prior Year'],
                        'Total Unique Students in Current Year': data['Total Students in Current Year'],
                        'Total Unique Students Retained': data['Total Students Retained'],
                        'Retention Rate': f"{data['Retention Rate'] * 100:.2f}",
                        'Type_8 Retained Count Previous': data['Type_8 Retained Count Previous'],
                        'Type_7 Retained Count Previous': data['Type_7 Retained Count Previous'],
                        'Type_6 Retained Count Previous': data['Type_6 Retained Count Previous'],
                        'Type_5 Retained Count Previous': data['Type_5 Retained Count Previous'],
                        'Type_4 Retained Count Previous': data['Type_4 Retained Count Previous'],
                        'Type_3 Retained Count Previous': data['Type_3 Retained Count Previous'],
                        'Type_2 Retained Count Previous': data['Type_2 Retained Count Previous'],
                        'Type_1 Retained Count Previous': data['Type_1 Retained Count Previous'],
                        'Type_8 Retained Count Current': data['Type_8 Retained Count Current'],
                        'Type_7 Retained Count Current': data['Type_7 Retained Count Current'],
                        'Type_6 Retained Count Current': data['Type_6 Retained Count Current'],
                        'Type_5 Retained Count Current': data['Type_5 Retained Count Current'],
                        'Type_4 Retained Count Current': data['Type_4 Retained Count Current'],
                        'Type_3 Retained Count Current': data['Type_3 Retained Count Current'],
                        'Type_2 Retained Count Current': data['Type_2 Retained Count Current'],
                        'Type_1 Retained Count Current': data['Type_1 Retained Count Current'],
                        'Type_8 Total Count Current Year': data['Type_8 Total Count Current Year'],
                        'Type_7 Total Count Current Year': data['Type_7 Total Count Current Year'],
                        'Type_6 Total Count Current Year': data['Type_6 Total Count Current Year'],
                        'Type_5 Total Count Current Year': data['Type_5 Total Count Current Year'],
                        'Type_4 Total Count Current Year': data['Type_4 Total Count Current Year'],
                        'Type_3 Total Count Current Year': data['Type_3 Total Count Current Year'],
                        'Type_2 Total Count Current Year': data['Type_2 Total Count Current Year'],
                        'Type_1 Total Count Current Year': data['Type_1 Total Count Current Year'],
                        'Type_8 Total Count Previous Year': data['Type_8 Total Count Previous Year'],
                        'Type_7 Total Count Previous Year': data['Type_7 Total Count Previous Year'],
                        'Type_6 Total Count Previous Year': data['Type_6 Total Count Previous Year'],
                        'Type_5 Total Count Previous Year': data['Type_5 Total Count Previous Year'],
                        'Type_4 Total Count Previous Year': data['Type_4 Total Count Previous Year'],
                        'Type_3 Total Count Previous Year': data['Type_3 Total Count Previous Year'],
                        'Type_2 Total Count Previous Year': data['Type_2 Total Count Previous Year'],
                        'Type_1 Total Count Previous Year': data['Type_1 Total Count Previous Year'],
                        'Calculation Type': 'School Year over School Year Retention'
                    })
            except Exception as e:
                print(f"Error processing group {group_by} for school years {year_start}-{year_end}: {e}")

    return pd.DataFrame(retention_results)

# Calculate retention results
# Classify attendance groups
# Apply filters
retention_df = df[(df['Class'].isin(selected_classes)) &
                 (df['Location'].isin(selected_locations)) &
                 (df['Teacher'].isin(selected_teachers)) &
                 (df['Age'].isin(selected_ages)) &
                 (df['Reg/NonReg'].isin(selected_reg_nonreg)) &
                 (df['School Year'].isin(selected_school_years)) &
                 (df['Season'].isin(selected_seasons)) &
                 (df['Session'].isin(selected_sessions))]

retention_df = classify_students_by_attendance(retention_df)

retentionWithStudentClass_df = retention_df
#st.write(retentionWithStudentClass_df)

# Calculate School Year Retention

retention_df = calculate_school_year_retention(retention_df)

if not retention_df.empty:
    # First, filter the DataFrame by cities
    city_retention_df = retention_df[retention_df['Group'].isin(selected_cities)]

    # Now, group by 'School Year Start' and calculate total retention for each year
    # We calculate total retained students and total students in the prior year for each group
    total_retention_df = city_retention_df.groupby('School Year Start').apply(
        lambda group: pd.Series({
            'Total Retained Students': group['Total Unique Students Retained'].sum(),
            'Total Students in Prior Year': group['Total Unique Students in Prior Year'].sum(),
            'Retention Rate (%)': (group['Total Unique Students Retained'].sum() / group['Total Unique Students in Prior Year'].sum()) * 100
        })
    ).reset_index()

    #Filter by Location
    location_retention_df = retention_df[retention_df['Group'].isin(selected_locations)]


# Get retention rate for each city
    # Find the maximum year in the 'school year start' column
    #max_year = int(city_retention_df['School Year Start'].max())

    # Calculate the target year as max year - 1
    #target_year = max_year - 1

    # Ensure 'School Year Start' is converted to an integer type
    city_retention_df['School Year Start'] = pd.to_numeric(city_retention_df['School Year Start'], errors='coerce').astype('Int64')
    total_retention_df['School Year Start'] = pd.to_numeric(total_retention_df['School Year Start'], errors='coerce').astype('Int64')

    # Calculate retention for each city by summing retained and total students for the prior year
    def calculate_city_retention(df, city_name):
        city_df = df[df['Group'] == city_name]
        if city_df.empty:
            return 0, 0  # Return zeroes if the city is not present in the filtered data
        total_retained = city_df['Total Unique Students Retained'].sum()
        total_prior_year = city_df['Total Unique Students in Prior Year'].sum()
        return total_retained, total_prior_year

    # Initialize totals for combined retention
    total_retained = 0
    total_prior_year = 0

    # Dictionary to store city retention rates
    city_retention_rates = {}

    # Calculate retention rates for each filtered city (if the filter exists)
    if 'Chicago' in city_retention_df['Group'].unique():
        chicago_retained, chicago_prior_year = calculate_city_retention(city_retention_df, 'Chicago')
        total_retained += chicago_retained
        total_prior_year += chicago_prior_year
        city_retention_rates['Chicago'] = chicago_retained / chicago_prior_year if chicago_prior_year > 0 else 0

    if 'Cleveland' in city_retention_df['Group'].unique():
        cleveland_retained, cleveland_prior_year = calculate_city_retention(city_retention_df, 'Cleveland')
        total_retained += cleveland_retained
        total_prior_year += cleveland_prior_year
        city_retention_rates['Cleveland'] = cleveland_retained / cleveland_prior_year if cleveland_prior_year > 0 else 0

    if 'San Diego' in city_retention_df['Group'].unique():
        sandiego_retained, sandiego_prior_year = calculate_city_retention(city_retention_df, 'San Diego')
        total_retained += sandiego_retained
        total_prior_year += sandiego_prior_year
        city_retention_rates['San Diego'] = sandiego_retained / sandiego_prior_year if sandiego_prior_year > 0 else 0

    # Calculate the total retention rate across all filtered cities
    if total_prior_year > 0:
        total_retention_rate = total_retained / total_prior_year
        city_retention_rates['Total'] = total_retention_rate


    # Retention Calculation and Graph
    st.markdown("<h5 style='text-align: left;'>School Year Retention</h5>", unsafe_allow_html=True)

    # Convert 'School Year Start' to integer if it's not already
    location_retention_df['School Year Start'] = location_retention_df['School Year Start'].astype(int)
    # Ensure 'Retention Rate' is converted to numeric (float)
    location_retention_df['Retention Rate'] = pd.to_numeric(location_retention_df['Retention Rate'], errors='coerce')

    # Plot graph ensuring the x-axis treats 'School Year Start' as categorical
    fig_RetentionYear_filtered = px.line(location_retention_df, 
                                        x='School Year Start', 
                                        y='Retention Rate', 
                                        color='Group',  # Separate lines by city
                                        markers=True, 
                                        title='Yearly Retention by Location',
                                        category_orders={"School Year Start": sorted(location_retention_df['School Year Start'].unique())})  # Ensure proper year sorting

    # Update traces (markers and lines styling)
    fig_RetentionYear_filtered.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                                            line=dict(width=3))

    # Ensure y-axis is sorted from lowest to highest and remove decimal points on x-axis
    fig_RetentionYear_filtered.update_layout(
        xaxis_title="School Year Start",
        yaxis_title="Yearly Retention by Location %",
        xaxis_tickangle=-45,
        xaxis=dict(tickmode='array', tickvals=city_retention_df['School Year Start'].unique()),  # Show only whole years on x-axis
        yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
        template='plotly_white',
        font=dict(size=14, color='black'),
        title_font=dict(size=24, color='white'),
        title_x=0.4,
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )

    # Display the chart
    st.plotly_chart(fig_RetentionYear_filtered, use_container_width=True)



    # Extract and convert retention rate values to floats, keeping two decimal places
    chicago_retention_value = round(float(city_retention_rates.get('Chicago', 0)*100), 2)
    cleveland_retention_value = round(float(city_retention_rates.get('Cleveland', 0)*100), 2)
    sandiego_retention_value = round(float(city_retention_rates.get('San Diego', 0)*100), 2)
    total_retention_value = round(float(city_retention_rates.get('Total', 0))*100, 2)

    # Display the retention rates in different columns
    col1, col2, col3, col4 = st.columns([2,2,2,2])

    with col1:
        st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Total Retention: {total_retention_value:.2f}%</h5>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<h5 style='text-align: left; margin-left: 0px;'>Chicago Retention: {chicago_retention_value:.2f}%</h5>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<h5 style='text-align: left; margin-left: 0px;'>Cleveland Retention: {cleveland_retention_value:.2f}%</h5>", unsafe_allow_html=True)

    with col4:
        st.markdown(f"<h5 style='text-align: left; margin-left: 0px;'>San Diego Retention: {sandiego_retention_value:.2f}%</h5>", unsafe_allow_html=True)


     #Plot Pie Chart
     #Sum the counts for each type
     type_counts = location_retention_df[['Type_8 Retained Count Previous', 'Type_7 Retained Count Previous', 'Type_6 Retained Count Previous', 'Type_5 Retained Count Previous', 'Type_4 Retained Count Previous', 
                                         'Type_3 Retained Count Previous', 'Type_2 Retained Count Previous', 'Type_1 Retained Count Previous']].sum()

    # Create a pie chart for the counts
     figPie = px.pie(values=type_counts, 
                 names=type_counts.index, 
                 title='Retained Students by number of Sessions Attended Previous Year',
                 labels={'names': 'Type'})

    #Customize pie chart appearance (optional)
     figPie.update_traces(textposition='inside', textinfo='percent+label')
     figPie.update_layout(showlegend=True)

    # Calculate the retention percentages for each type (Type_8 to Type_1)
     for student_type in ['8', '7', '6', '5', '4', '3', '2', '1']:
         retained_col = f'Type_{student_type} Retained Count Previous'
         total_col = f'Type_{student_type} Total Count Previous Year'
        
         #Calculate the retention percentage (retained / total)
         retention_df[f'Type_{student_type} Retention'] = retention_df[retained_col] / retention_df[total_col] * 100

    # Filter the retention_df based on selected locations
    filtered_retention_df = retention_df[retention_df['Group'].isin(selected_locations)]

    # Calculate the retention percentages for each type
     retention_percentages = {}
     for student_type in ['8', '7', '6', '5', '4', '3', '2', '1']:
         retained_col = f'Type_{student_type} Retained Count Previous'
         total_col = f'Type_{student_type} Total Count Previous Year'
        
         #Calculate the sum of retained counts and total counts for each type
         sum_retained = filtered_retention_df[retained_col].sum()
         sum_total = filtered_retention_df[total_col].sum()
        
          #Calculate the retention percentage for each type (sum_retained / sum_total)
         retention_percentages[retained_col] = (sum_retained / sum_total * 100) if sum_total > 0 else 0

     #Prepare the data for plotting
     data = pd.DataFrame({
         'Type': [f'Type_{student_type}' for student_type in ['8', '7', '6', '5', '4', '3', '2', '1']],
         'Retention Percentage': [retention_percentages[f'Type_{student_type} Retained Count Previous'] for student_type in ['8', '7', '6', '5', '4', '3', '2', '1']]
     })

    # Plot the bar chart
     fig = px.bar(data, x='Type', y='Retention Percentage', 
                 title='Retention Percentage by Number of Sessions Attended the Previous Year',
                 labels={'Retention Percentage': 'Retention %'},
                 text='Retention Percentage')

      #Customize the appearance of the bar chart
     fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
     #Fix the y-axis scale to properly reflect percentages and remove decimals
     fig.update_layout(
         yaxis=dict(range=[0, 100], showgrid=True, ticksuffix='%'),  # Set the y-axis range from 0 to 100%
         title_font=dict(size=16),
         font=dict(size=14)
 )

     #Create two columns for side-by-side layout
     col1, col2 = st.columns(2)

    # Display the bar chart in the first column
     with col1:
           st.plotly_chart(fig, use_container_width=True)

    # Display the pie chart in the second column
     with col2:
         st.plotly_chart(figPie, use_container_width=True)

# Sum the total counts for each type
    total_countsPrev = {
        'Type_8 Retention Count': filtered_retention_df['Type_8 Retained Count Previous'].sum(),
        'Type_7 Retention Count': filtered_retention_df['Type_7 Retained Count Previous'].sum(),
        'Type_6 Retention Count': filtered_retention_df['Type_6 Retained Count Previous'].sum(),
        'Type_5 Retention Count': filtered_retention_df['Type_5 Retained Count Previous'].sum(),
        'Type_4 Retention Count': filtered_retention_df['Type_4 Retained Count Previous'].sum(),
        'Type_3 Retention Count': filtered_retention_df['Type_3 Retained Count Previous'].sum(),
        'Type_2 Retention Count': filtered_retention_df['Type_2 Retained Count Previous'].sum(),
        'Type_1 Retention Count': filtered_retention_df['Type_1 Retained Count Previous'].sum(),
    }
    # Sum the total counts for each type
    total_counts = {
        'Type_8 Retention Count': filtered_retention_df['Type_8 Retained Count Current'].sum(),
        'Type_7 Retention Count': filtered_retention_df['Type_7 Retained Count Current'].sum(),
        'Type_6 Retention Count': filtered_retention_df['Type_6 Retained Count Current'].sum(),
        'Type_5 Retention Count': filtered_retention_df['Type_5 Retained Count Current'].sum(),
        'Type_4 Retention Count': filtered_retention_df['Type_4 Retained Count Current'].sum(),
        'Type_3 Retention Count': filtered_retention_df['Type_3 Retained Count Current'].sum(),
        'Type_2 Retention Count': filtered_retention_df['Type_2 Retained Count Current'].sum(),
        'Type_1 Retention Count': filtered_retention_df['Type_1 Retained Count Current'].sum(),
    }

    # Prepare the data for plotting
    dataCurrent = pd.DataFrame({
        'Type': list(total_counts.keys()),
        'Total Count': list(total_counts.values())
    })
    # Prepare the data for plotting
    dataPrevious = pd.DataFrame({
        'Type': list(total_countsPrev.keys()),
        'Total Count': list(total_countsPrev.values())
    })

    # Create the histogram using Plotly
    figCurrent = px.bar(dataCurrent, x='Type', y='Total Count', 
                title='Sum of Total Counts by Type',
                labels={'Total Count': 'Sum of Total Count'},
                text='Total Count')

    # Customize the appearance of the histogram
    figCurrent.update_traces(texttemplate='%{text:.0f}', textposition='outside', marker_color = 'green')

    # Center the title, make the font bigger, and center the graph
    figCurrent.update_layout(
        title={
            'text': "Retained Student's by Type for Current Year",
            'y': 1,  # Position the title higher
            'x': 0.5,   # Center the title horizontally
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}  # Increase title font size
        },
        yaxis_tickformat='', 
        width = 1300,
        margin=dict(l=50, r=50, t=20, b=50),  # Center the entire chart by adjusting margins
    )

    fig = px.bar(dataPrevious, x='Type', y='Total Count', 
                title='Sum of Total Counts by Type',
                labels={'Total Count': 'Sum of Total Count'},
                text='Total Count')

    # Customize the appearance of the histogram
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside', marker_color = 'pink')

    # Center the title, make the font bigger, and center the graph
    fig.update_layout(
        title={
            'text': "Retained Student's by Type for Previous Year",
            'y': 1,  # Position the title higher
            'x': 0.5,   # Center the title horizontally
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}  # Increase title font size
        },
        yaxis_tickformat='', 
        width = 1300,
        margin=dict(l=50, r=50, t=20, b=50),  # Center the entire chart by adjusting margins
    )

    # Create two columns for side-by-side layout
    col1, col2 = st.columns(2)

    # Display the bar chart in the first column
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    # Display the pie chart in the second column
    with col2:
        st.plotly_chart(figCurrent, use_container_width=True)

    # Sum the total counts for each type
    total_counts = {
        'Type_8 Total Count': filtered_retention_df['Type_8 Total Count Current Year'].sum(),
        'Type_7 Total Count': filtered_retention_df['Type_7 Total Count Current Year'].sum(),
        'Type_6 Total Count': filtered_retention_df['Type_6 Total Count Current Year'].sum(),
        'Type_5 Total Count': filtered_retention_df['Type_5 Total Count Current Year'].sum(),
        'Type_4 Total Count': filtered_retention_df['Type_4 Total Count Current Year'].sum(),
        'Type_3 Total Count': filtered_retention_df['Type_3 Total Count Current Year'].sum(),
        'Type_2 Total Count': filtered_retention_df['Type_2 Total Count Current Year'].sum(),
        'Type_1 Total Count': filtered_retention_df['Type_1 Total Count Current Year'].sum(),
    }
    # Sum the total counts for each type
    total_countsPrev = {
        'Type_8 Total Count': filtered_retention_df['Type_8 Total Count Previous Year'].sum(),
        'Type_7 Total Count': filtered_retention_df['Type_7 Total Count Previous Year'].sum(),
        'Type_6 Total Count': filtered_retention_df['Type_6 Total Count Previous Year'].sum(),
        'Type_5 Total Count': filtered_retention_df['Type_5 Total Count Previous Year'].sum(),
        'Type_4 Total Count': filtered_retention_df['Type_4 Total Count Previous Year'].sum(),
        'Type_3 Total Count': filtered_retention_df['Type_3 Total Count Previous Year'].sum(),
        'Type_2 Total Count': filtered_retention_df['Type_2 Total Count Previous Year'].sum(),
        'Type_1 Total Count': filtered_retention_df['Type_1 Total Count Previous Year'].sum(),
    }

    # Prepare the data for plotting
    dataCurrent = pd.DataFrame({
        'Type': list(total_counts.keys()),
        'Total Count': list(total_counts.values())
    })
    # Prepare the data for plotting
    dataPrevious = pd.DataFrame({
        'Type': list(total_countsPrev.keys()),
        'Total Count': list(total_countsPrev.values())
    })

    # Create the histogram using Plotly
    figCurrent = px.bar(dataCurrent, x='Type', y='Total Count', 
                title='Sum of Total Counts by Type',
                labels={'Total Count': 'Sum of Total Count'},
                text='Total Count')

    # Customize the appearance of the histogram
    figCurrent.update_traces(texttemplate='%{text:.0f}', textposition='outside', marker_color = 'green')

    # Center the title, make the font bigger, and center the graph
    figCurrent.update_layout(
        title={
            'text': "Total Unique Student's Type for Current Year",
            'y': 1,  # Position the title higher
            'x': 0.5,   # Center the title horizontally
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}  # Increase title font size
        },
        yaxis_tickformat='', 
        width = 1300,
        margin=dict(l=50, r=50, t=20, b=50),  # Center the entire chart by adjusting margins
    )

    fig = px.bar(dataPrevious, x='Type', y='Total Count', 
                title='Sum of Total Counts by Type',
                labels={'Total Count': 'Sum of Total Count'},
                text='Total Count')

    # Customize the appearance of the histogram
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside', marker_color = 'pink')

    # Center the title, make the font bigger, and center the graph
    fig.update_layout(
        title={
            'text': "Total Unique Student's Type for Previous Year",
            'y': 1,  # Position the title higher
            'x': 0.5,   # Center the title horizontally
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}  # Increase title font size
        },
        yaxis_tickformat='', 
        width = 1300,
        margin=dict(l=50, r=50, t=20, b=50),  # Center the entire chart by adjusting margins
    )

    # Create two columns for side-by-side layout
    col1, col2 = st.columns(2)

    # Display the bar chart in the first column
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    # Display the pie chart in the second column
    with col2:
        st.plotly_chart(figCurrent, use_container_width=True)

else:
    st.warning("No data available for Year over Year Retention based on the selected filters.")

#Write dataframe
st.write(retention_df)

# SESSION TO SESSION RETENTION

if not retentionWithStudentClass_df.empty:
    
    def calculate_flow_retention(df, group_by_cols, start_periods, end_periods):
        # Filter DataFrame for start and end periods based on the Season and Session flow
        group_start = df[(df['Season'] == start_periods[0][0]) & (df['Session'] == start_periods[0][1]) & (df['School Year'] == start_periods[1])].groupby(group_by_cols, observed=True)['DancerID'].apply(set)
        
        # Check if there's data for the end period, otherwise skip
        if not df[(df['Season'] == end_periods[0][0]) & (df['Session'] == end_periods[0][1]) & (df['School Year'] == end_periods[1])].empty:
            group_end = df[(df['Season'] == end_periods[0][0]) & (df['Session'] == end_periods[0][1]) & (df['School Year'] == end_periods[1])].groupby(group_by_cols, observed=True)['DancerID'].apply(set)
        else:
            return {}  # If no end period data, skip this retention calculation

        retention_results = {}

        for group in group_start.index:
            try:
                if group in group_end:
                    start_count = len(group_start[group])  # Total unique students in the starting period
                    end_count = len(group_end[group])  # Total unique students in the end period
                    retained_students = group_start[group].intersection(group_end[group])  # Unique students retained
                    retained_count = len(retained_students)  # Count of unique retained students

                    # Ensure correct filtering by group values
                    group_condition = True
                    for col, value in zip(group_by_cols, group if isinstance(group, tuple) else [group]):
                        group_condition &= df[col] == value

                    # Filter for retained students and total students for the start period within the same group
                    retained_student_types = df[(df['DancerID'].isin(retained_students)) & (df['Season'] == end_periods[0][0]) & (df['Session'] == end_periods[0][1]) & group_condition]
                    total_student_types = df[(df['Season'] == start_periods[0][0]) & (df['Session'] == start_periods[0][1]) & group_condition]

                    # Check if the prior year exists in the data
                    prior_year = start_periods[1] - 1
                    if f'Type_{prior_year}' in df.columns:
                        # Count unique student IDs for each type (retained students) in the prior year
                        retained_type_counts = retained_student_types.groupby(f'Type_{prior_year}')['DancerID'].nunique()

                        # Count unique student IDs for each type (total students in start period) in the prior year
                        total_type_counts = total_student_types.groupby(f'Type_{prior_year}')['DancerID'].nunique()
                    else:
                        # Skip calculation or assign 0s if there is no type for the prior year
                        retained_type_counts = {}
                        total_type_counts = {}

                    # Convert to dictionaries for easier access
                    retained_type_counts_dict = retained_type_counts
                    total_type_counts_dict = total_type_counts

                    # Calculate retention rate
                    retention_rate = retained_count / start_count if start_count > 0 else 0

                    # Store the results, including type counts for retained and total students
                    retention_results[group] = {
                        'Total Students in Start Period': start_count,
                        'Total Students in End Period': end_count,
                        'Total Students Retained': retained_count,
                        'Retention Rate': retention_rate,
                    }

                    # Add type counts for retained and total students to retention results
                    for student_type in ['8', '7', '6', '5', '4', '3', '2', '1']:
                        retention_results[group][f'Type_{student_type} Retained Count'] = retained_type_counts_dict.get(f'Type {student_type}', 0)
                        retention_results[group][f'Type_{student_type} Total Count'] = total_type_counts_dict.get(f'Type {student_type}', 0)

            except Exception as e:
                st.write(f"Error processing group {group} for periods {start_periods[1]}-{end_periods[1]}: {e}")

        return retention_results


    # Updated function to handle year transitions correctly, including the final year case
    def calculate_flow_based_retention(df):
        retention_results = []

        # Define the flow for intra-year and inter-year transitions
        flow = [
            (('Summer', 1), ('Summer', 2)),
            (('Summer', 2), ('Fall', 1)),
            (('Fall', 1), ('Fall', 2)),
            (('Fall', 2), ('Winter', 1)),
            (('Winter', 1), ('Winter', 2)),
            (('Winter', 2), ('Spring', 1)),
            (('Spring', 1), ('Spring', 2)),
            (('Spring', 2), ('Summer', 1))  # Transition to next year after Spring 2
        ]

        # Define groupings
        group_by_combinations = [
            ['City'],
            ['Location'],
            ['Class'],
            ['Teacher'],
            ['Reg/Non Reg'],
        ]

        # Loop through the selected years and flow
        years = sorted(df['School Year'].dropna().unique())

        for year in years:
            for (start_period, end_period) in flow:
                # Only increment the year when transitioning from Spring 2 to Summer 1
                if start_period == ('Spring', 2) and end_period == ('Summer', 1):
                    end_year = year + 1 if year + 1 in years else year  # Handle the final year case by keeping the same year if no next year
                else:
                    end_year = year

                start_periods = (start_period, year)
                end_periods = (end_period, end_year)

                # Make sure that we only calculate retention for available years and periods
                if end_periods[1] in years or (end_period[0] == 'Summer' and year == years[-1]):
                    for group_by in group_by_combinations:
                        try:
                            retention = calculate_flow_retention(df, group_by, start_periods, end_periods)
                            if retention:
                                for group, data in retention.items():
                                    retention_results.append({
                                        'Group': group,
                                        'Start Year': start_periods[1],
                                        'End Year': end_periods[1],
                                        'Start Season': start_periods[0][0],
                                        'End Season': end_periods[0][0],
                                        'Start Session': start_periods[0][1],
                                        'End Session': end_periods[0][1],
                                        'Total Unique Students in Start Period': data['Total Students in Start Period'],
                                        'Total Unique Students in End Period': data['Total Students in End Period'],
                                        'Total Unique Students Retained': data['Total Students Retained'],
                                        'Retention Rate': f"{data['Retention Rate'] * 100:.2f}",
                                        'Type_8 Retained Count': data['Type_8 Retained Count'],
                                        'Type_7 Retained Count': data['Type_7 Retained Count'],
                                        'Type_6 Retained Count': data['Type_6 Retained Count'],
                                        'Type_5 Retained Count': data['Type_5 Retained Count'],
                                        'Type_4 Retained Count': data['Type_4 Retained Count'],
                                        'Type_3 Retained Count': data['Type_3 Retained Count'],
                                        'Type_2 Retained Count': data['Type_2 Retained Count'],
                                        'Type_1 Retained Count': data['Type_1 Retained Count'],
                                        'Type_8 Total Count': data['Type_8 Total Count'],
                                        'Type_7 Total Count': data['Type_7 Total Count'],
                                        'Type_6 Total Count': data['Type_6 Total Count'],
                                        'Type_5 Total Count': data['Type_5 Total Count'],
                                        'Type_4 Total Count': data['Type_4 Total Count'],
                                        'Type_3 Total Count': data['Type_3 Total Count'],
                                        'Type_2 Total Count': data['Type_2 Total Count'],
                                        'Type_1 Total Count': data['Type_1 Total Count'],
                                        'Calculation Type': 'Flow-Based Retention'
                                    })

                        except Exception as e:
                            print(f"Error processing group {group_by} for periods {start_period}-{end_period}: {e}")

        return pd.DataFrame(retention_results)



    retention_flow_df = calculate_flow_based_retention(retentionWithStudentClass_df)
    
    #Filter by Location
    location_flowretention_df = retention_flow_df[retention_flow_df['Group'].isin(selected_locations)]

    # Retention Calculation and Graph
    st.markdown("<h5 style='text-align: left;'>Session over Session Retention</h5>", unsafe_allow_html=True)

    # Create a column combining 'Start Year', 'Start Season', and 'Start Session' for the y-axis
    location_flowretention_df['Period'] = location_flowretention_df['End Year'].astype(str) + ' ' + location_flowretention_df['End Season'] + ' ' + location_flowretention_df['End Session'].astype(str)

    # Create the line plot using Plotly
    fig = px.line(
        location_flowretention_df,
        x='Period',
        y='Retention Rate',
        title='Retention Rates Session over Session',
        markers=True,
        color='Group',
        labels={
            'Period': 'Start Period (Year, Season, Session)',
            'Retention Rate': 'Retention Rate (%)'
        }
    )

    # Customize the layout
    fig.update_layout(
        yaxis_title="Retention Rate (%)",
        xaxis_title="Year-Season-Session",
        title_x=0.4,  # Center the title
        height=600
    )

    # Streamlit display
    st.plotly_chart(fig, use_container_width=True)

    #Plot Pie Chart
    # Sum the counts for each type
    type_counts = location_flowretention_df[['Type_8 Retained Count', 'Type_7 Retained Count', 'Type_6 Retained Count', 'Type_5 Retained Count', 'Type_4 Retained Count', 
                                        'Type_3 Retained Count', 'Type_2 Retained Count', 'Type_1 Retained Count']].sum()

    # Create a pie chart for the counts
    figPie = px.pie(values=type_counts, 
                names=type_counts.index, 
                title='Retained Students by number of Sessions Attended Previous Year',
                labels={'names': 'Type'})

    # Customize pie chart appearance (optional)
    figPie.update_traces(textposition='inside', textinfo='percent+label')
    figPie.update_layout(showlegend=True)

    # Calculate the retention percentages for each type (Type_8 to Type_1)
    for student_type in ['8', '7', '6', '5', '4', '3', '2', '1']:
        retained_col = f'Type_{student_type} Retained Count'
        total_col = f'Type_{student_type} Total Count'
        
        # Calculate the retention percentage (retained / total)
        retention_flow_df[f'Type_{student_type} Retention'] = retention_flow_df[retained_col] / retention_flow_df[total_col] * 100

    # Filter the retention_df based on selected locations
    filtered_retentionflow_df = retention_flow_df[retention_flow_df['Group'].isin(selected_locations)]

    st.write(filtered_retentionflow_df)
else:
    st.warning("No data available for Session over Session Retention based on the selected filters.")
