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
# Create tabs
tab1, tab2, tab3= st.tabs(["Dashboard", "Summary Statistics", "Group By"])

# Tab 1: Dashboard
with tab1:

    # Add Display toggle
    display_toggle = st.radio("Display", options=["School Year", "Session"], index=0)

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
            ('Summer', 1), ('Summer', 2), ('Fall', 1), ('Fall', 2), ('Winter', 1), ('Winter', 2),
            ('Spring', 1), ('Spring', 2)
        ]
        
        # Create a mapping of periods to school years
        school_year_mapping = {}
        school_year_string_mapping = {}
        
        years = sorted(df['Year'].unique())  # Loop through years
        
        for year_start in years:
            # Define start periods for the current school year
            start_periods = [
                (year_start, 'Summer', 1), (year_start, 'Summer', 2),  # Summer in year_start
                (year_start, 'Fall', 1), (year_start, 'Fall', 2),      # Fall in year_start
                (year_start + 1, 'Winter', 1), (year_start + 1, 'Winter', 2),  # Winter in year_start + 1
                (year_start + 1, 'Spring', 1), (year_start + 1, 'Spring', 2)   # Spring in year_start + 1
            ]
            
            # Define the school year string in 'YYYY-YYYY' format
            school_year_str = f"{year_start}-{year_start + 1}"
            
            # Map start periods to school year start and school year string
            for period in start_periods:
                # Assign year_start as the school year start
                school_year_mapping[period] = year_start
                # Assign school_year_str as the formatted school year string
                school_year_string_mapping[period] = school_year_str

        # Create new columns in the DataFrame for the school year and school year string
        df['School Year'] = df.apply(lambda row: school_year_mapping.get((row['Year'], row['Season'], row['Session'])), axis=1)
        df['School Year String'] = df.apply(lambda row: school_year_string_mapping.get((row['Year'], row['Season'], row['Session'])), axis=1)

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

    # Create X-axis labels using School Year String
    df['Year_Season_Session'] = df['School Year String'] + ' ' + df['Season'] + ' ' + df['Session'].astype(str)

    # Determine the x-axis label based on the display toggle
    if display_toggle == "School Year":
        df['x_axisLabel'] = df['School Year String']
        df['Sort_Key'] = df['School Year']  # Sort by the start year of the school year
    else:
        df['x_axisLabel'] = df['Year_Season_Session']
        df['Sort_Key'] = df['School Year'] * 100 + df['Season_Order'] * 10 + df['Session']


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
    col_city, col_class, col_location, col_teacher, col_reg_nonreg, col_age = st.columns(6)

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
        selected_ages = df['Age'].unique()

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
        grouped_df = filtered_df.groupby('x_axisLabel').agg({'DancerID': 'count'}).reset_index()
        grouped_df.rename(columns={'DancerID': 'Number of Dancers'}, inplace=True)
        grouped_df = grouped_df.merge(df[['x_axisLabel', 'Sort_Key']].drop_duplicates(), on='x_axisLabel')
        grouped_df = grouped_df.sort_values('Sort_Key')
        total_dancers = grouped_df['Number of Dancers'].sum()

        # Plotting  dynamic graph
        fig = px.line(grouped_df, x='x_axisLabel', y='Number of Dancers', markers=True, 
                    title='Dancer Enrollment (Total Slots)')
        
        fig.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                        line=dict(width=3, color='steelblue'))

        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Dancers Enrolled",
            xaxis_tickangle=-45,
            template='plotly_white',
            font=dict(size=14, color='black'),
            title_font=dict(size=24, color='white'),
            title_x=0.4,
            xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        for i, row in grouped_df.iterrows():
            fig.add_annotation(
                x=row['x_axisLabel'],
                y=row['Number of Dancers'],
                text=str(row['Number of Dancers']),
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                xshift=10,
                font=dict(color='white', weight='bold')
            )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Total Enrollment (Slots): {total_dancers}</h5>", unsafe_allow_html=True)
    else:
        st.warning("No data available for the selected filters.")
    
    #Unique Dancers
    # Display Header for the Chart
    st.markdown("<h5 style='text-align: left;'>Total Unique Dancers</h5>", unsafe_allow_html=True)

    if not filtered_df.empty:
        # Count unique DancerIDs for each Year_Season_Session
        unique_enrollment_df = filtered_df.groupby('x_axisLabel').agg({'DancerID': 'nunique'}).reset_index()
        unique_enrollment_df.rename(columns={'DancerID': 'Number of Unique Dancers'}, inplace=True)

        # Merge to retain Sort_Key for correct ordering and sort
        unique_enrollment_df = unique_enrollment_df.merge(df[['x_axisLabel', 'Sort_Key']].drop_duplicates(), on='x_axisLabel')
        unique_enrollment_df = unique_enrollment_df.sort_values('Sort_Key')

        # Calculate total enrollment across all sessions
        total_unique_dancers = unique_enrollment_df['Number of Unique Dancers'].sum()

        # Plotting the dynamic graph
        fig = px.line(unique_enrollment_df, x='x_axisLabel', y='Number of Unique Dancers', markers=True, 
                    title='Total Unique Dancers')
        
        fig.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                        line=dict(width=3, color='pink'))

        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Unique Dancers Enrolled",
            xaxis_tickangle=-45,
            template='plotly_white',
            font=dict(size=14, color='black'),
            title_font=dict(size=24, color='white'),
            title_x=0.44,
            xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        # Add annotations to each point in the graph
        for _, row in unique_enrollment_df.iterrows():
            fig.add_annotation(
                x=row['x_axisLabel'],
                y=row['Number of Unique Dancers'],
                text=str(row['Number of Unique Dancers']),
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                xshift=10,
                font=dict(color='white', weight='bold')
            )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Display total unique dancers enrolled
        st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Total Unique Dancers Enrolled: {total_unique_dancers}</h5>", unsafe_allow_html=True)

    # DANCER ACQUISITION GRAPH
    st.markdown("<h5 style='text-align: left;'>Brand New Dancers as % of Unique Dancers</h5>", unsafe_allow_html=True)
    if not filtered_df.empty:
        df_sorted = df.sort_values('Sort_Key')
        seen_dancers = set()
        newly_acquired = []
        for _, row in df_sorted.iterrows():
            session_id = row['x_axisLabel']
            dancer_id = row['DancerID']
            if dancer_id not in seen_dancers:
                newly_acquired.append({'x_axisLabel': session_id, 'DancerID': dancer_id, 
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
            (acquired_df['x_axisLabel'].isin(filtered_df['x_axisLabel']))
        ]

        # Group by x_axisLabel to get count of newly acquired dancers
        acquired_grouped_filtered_df = acquired_filtered_df.groupby('x_axisLabel').agg({'DancerID': 'count'}).reset_index()
        acquired_grouped_filtered_df.rename(columns={'DancerID': 'Newly Acquired Students'}, inplace=True)
        acquired_grouped_filtered_df = acquired_grouped_filtered_df.merge(df[['x_axisLabel', 'Sort_Key']].drop_duplicates(), on='x_axisLabel')
        acquired_grouped_filtered_df = acquired_grouped_filtered_df.sort_values('Sort_Key')

        # Calculate unique dancers for each period
        unique_dancers_df = filtered_df.groupby('x_axisLabel').agg({'DancerID': 'nunique'}).reset_index()
        unique_dancers_df.rename(columns={'DancerID': 'Total Unique Dancers'}, inplace=True)

        # Merge to calculate acquisition as a percentage
        acquisition_percentage_df = acquired_grouped_filtered_df.merge(unique_dancers_df, on='x_axisLabel')
        acquisition_percentage_df['Acquisition %'] = (acquisition_percentage_df['Newly Acquired Students'] / 
                                                    acquisition_percentage_df['Total Unique Dancers']) * 100

        # Plot graph
        fig_acquired_filtered = px.line(acquisition_percentage_df, x='x_axisLabel', y='Acquisition %', markers=True, 
                                        title='Brand New Dancers as % of Unique Dancers')
        
        fig_acquired_filtered.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                                            line=dict(width=3, color='mediumseagreen'))

        fig_acquired_filtered.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Brand New Percentage",
            xaxis_tickangle=-45,
            template='plotly_white',
            font=dict(size=14, color='black'),
            title_font=dict(size=24, color='white'),
            title_x=0.32,
            xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        # Optional: Add annotations for each point
        for i, row in acquisition_percentage_df.iterrows():
            fig_acquired_filtered.add_annotation(
                x=row['x_axisLabel'],
                y=row['Acquisition %'],
                text=f"{row['Acquisition %']:.0f}%",
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
    #if total_dancers > 0:
        #acquisition_ratio = (total_acquired_students / total_dancers) * 100
    #else:
        #acquisition_ratio = 0

    # Display the total acquired students and Acquisition Ratio side by side
    #col1, col2 = st.columns([1, 4])
    #with col1:
    st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Total Acquisition: {total_acquired_students}</h5>", unsafe_allow_html=True)
    #with col2:
        #st.markdown(f"<h5 style='text-align: left; margin-left: 0px;'>Acquisition Ratio: {acquisition_ratio:.2f}%</h5>", unsafe_allow_html=True)

    #RETENTION
    st.markdown("<h5 style='text-align: left;'>Retention Percentage</h5>", unsafe_allow_html=True)

    if not filtered_df.empty:
        # Merge the unique enrollment and acquisition DataFrames on 'x_axisLabel'
        retention_df = unique_enrollment_df.merge(acquired_grouped_filtered_df, on='x_axisLabel', how='left')
        
        # Calculate Retained Dancers as the difference between Total Unique Dancers and Newly Acquired Students
        retention_df['Newly Acquired Students'] = retention_df['Newly Acquired Students'].fillna(0)
        retention_df['Retained Dancers'] = retention_df['Number of Unique Dancers'] - retention_df['Newly Acquired Students']
        
        # Calculate retention as a percentage
        retention_df['Retention %'] = (retention_df['Retained Dancers'] / retention_df['Number of Unique Dancers']) * 100
        
        # Sort the DataFrame based on Sort_Key for proper session ordering
        retention_df = retention_df.merge(df[['x_axisLabel', 'Sort_Key']].drop_duplicates(), on='x_axisLabel')
        retention_df = retention_df.sort_values('Sort_Key')

        # Plotting Retention Percentage Graph
        fig_retention = px.line(retention_df, x='x_axisLabel', y='Retention %', markers=True, 
                                title='Retention Percentage')
        
        fig_retention.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                                    line=dict(width=3, color='orange'))

        fig_retention.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Retention Percentage",
            xaxis_tickangle=-45,
            template='plotly_white',
            font=dict(size=14, color='black'),
            title_font=dict(size=24, color='white'),
            title_x=0.45,
            xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        # Add annotations for each point in the graph
        for _, row in retention_df.iterrows():
            fig_retention.add_annotation(
                x=row['x_axisLabel'],
                y=row['Retention %'],
                text=f"{row['Retention %']:.0f}%",
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                xshift=10,
                font=dict(color='white', weight='bold')
            )

        # Display the retention plot in Streamlit
        st.plotly_chart(fig_retention, use_container_width=True)

        # Display total retention percentage for all periods combined, if desired
        total_retained_dancers = retention_df['Retained Dancers'].sum()
        overall_retention_percentage = (retention_df['Retained Dancers'].sum() / retention_df['Number of Unique Dancers'].sum()) * 100
        st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Overall Retention Percentage: {overall_retention_percentage:.0f}%</h5>", unsafe_allow_html=True)
    else:
        st.warning("No data available for the selected filters.")

    # DROPS - Percentage Change in Retained Dancers from Previous Period
    if not filtered_df.empty: 
        # Calculate the percentage change in retained dancers from the previous period
        retention_df['Drop %'] = retention_df['Retained Dancers'].pct_change().fillna(0) * 100

        # Plot the Percentage Change in Retained Dancers Graph
        fig_drop_pct = px.line(retention_df, x='x_axisLabel', y='Drop %', markers=True, 
                            title='Percentage Change in Retained Dancers')
        
        fig_drop_pct.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                                line=dict(width=3, color='red'))

        fig_drop_pct.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Percentage Change in Retained Dancers",
            xaxis_tickangle=-45,
            template='plotly_white',
            font=dict(size=14, color='black'),
            title_font=dict(size=24, color='white'),
            title_x=0.38,
            xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        # Add annotations to each point in the graph
        for _, row in retention_df.iterrows():
            fig_drop_pct.add_annotation(
                x=row['x_axisLabel'],
                y=row['Drop %'],
                text=f"{row['Drop %']:.0f}%", 
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                xshift=10,
                font=dict(color='white', weight='bold')
            )

        # Display the percentage change in retention plot in Streamlit
        st.plotly_chart(fig_drop_pct, use_container_width=True)

        # DROPS - Difference in Retained Dancers from Previous Period
        retention_df['Drops'] = retention_df['Retained Dancers'].diff().fillna(0)

        # Plot the Difference in Retained Dancers Graph
        fig_drop_diff = px.line(retention_df, x='x_axisLabel', y='Drops', markers=True, 
                                title='Difference in Retained Dancers')
        
        fig_drop_diff.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                                    line=dict(width=3, color='blue'))

        fig_drop_diff.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Difference in Retained Dancers",
            xaxis_tickangle=-45,
            template='plotly_white',
            font=dict(size=14, color='black'),
            title_font=dict(size=24, color='white'),
            title_x=0.38,
            xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        # Add annotations to each point in the graph
        for _, row in retention_df.iterrows():
            fig_drop_diff.add_annotation(
                x=row['x_axisLabel'],
                y=row['Drops'],
                text=f"{row['Drops']:.0f}", 
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                xshift=10,
                font=dict(color='white', weight='bold')
            )

        # Display the difference in retention plot in Streamlit
        st.plotly_chart(fig_drop_diff, use_container_width=True)

        # Display total dropped dancers count
        total_dropped_dancers = retention_df['Drops'].sum()
        st.markdown(f"<h5 style='text-align: left;'>Total Retained Dancers Dropped: {total_dropped_dancers:.0f}</h5>", unsafe_allow_html=True)
    else:
        st.warning("No data available for the selected filters.")
    # Tab 2: Summary Statistics
    with tab2:
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

                # If there is a previous year, then
                if previous_year in years:
                    previous_year_attendance = attendance_counts[attendance_counts['School Year'] == previous_year].copy()

                    # Count occurrences of each DancerID
                    dancer_counts = previous_year_attendance.groupby('DancerID')['Count'].sum().reset_index(name='Attendance Count')

                    # Classify students based on attendance count
                    dancer_counts['Type'] = dancer_counts['Attendance Count'].apply(lambda x: classify_student(x, thresholds))

                    # Optionally, you can merge this back to your original DataFrame if needed
                    df = df.merge(dancer_counts[['DancerID', 'Type']], on='DancerID', how='left')
                    df.rename(columns={'Type': f'Type_{previous_year}'}, inplace=True)

                    # Plot histogram for the current year based on the previous year's Type classifications
                    type_counts = dancer_counts['Type'].value_counts().reset_index()
                    type_counts.columns = ['Type', 'Count']

                    # Create the histogram with Plotly
                    fig = px.bar(
                        type_counts, 
                        x='Type', 
                        y='Count', 
                        title=f'Retained Counts by Type for School Year {previous_year}-{previous_year + 1}',
                        labels={'Type': 'Type', 'Count': 'Retained Dancer Count'}
                    )
                    # Add labels above each bar showing the total count
                    fig.update_traces(
                        text=type_counts['Count'],  # Set text labels to show the count
                        textposition='outside'  # Position labels outside the bars
                    )

                    fig.update_layout(
                        xaxis_title="Type",
                        yaxis_title="Retained Dancer Count",
                        template='plotly_white',
                        title_x=0.4
                    )
                    
                    # Display the figure in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

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

        # Example usage in Streamlit
        st.title("Retention Analysis by School Year")
        SessionCounts_df = classify_students_by_attendance(filtered_df)

        # Define summary data for easy access
        summary_data = {
            "Total Enrollment": total_dancers,
            "Total Unique Dancers Enrolled": total_unique_dancers,
            "Average Number of Slots Attended": f"{total_dancers / total_unique_dancers:.0f}",
            "Total Acquisition": total_acquired_students,
            "Total Retained Dancers": total_retained_dancers,
            "Total Dropped Dancers": f"{total_dropped_dancers:.0f}",
            "Acquisition Ratio": f"{total_acquired_students / total_unique_dancers:.0%}",
            "Retention Ratio": f"{total_retained_dancers / total_unique_dancers:.0%}",
        }

        # Row 1: Total Enrollment, Total Unique Dancers, Average Number of Slots Attended
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<h5 style='text-align: center;'>Total Enrollment: <b>{summary_data['Total Enrollment']}</b></h5>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h5 style='text-align: center;'>Total Unique Dancers Enrolled: <b>{summary_data['Total Unique Dancers Enrolled']}</b></h5>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<h5 style='text-align: center;'>Average Number of Slots Attended: <b>{summary_data['Average Number of Slots Attended']}</b></h5>", unsafe_allow_html=True)

        # Row 2: Total Acquisition, Total Retained Dancers, Total Dropped Dancers
        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown(f"<h5 style='text-align: center;'>Total Acquisition: <b>{summary_data['Total Acquisition']}</b></h5>", unsafe_allow_html=True)
        with col5:
            st.markdown(f"<h5 style='text-align: center;'>Total Retained Dancers: <b>{summary_data['Total Retained Dancers']}</b></h5>", unsafe_allow_html=True)
        with col6:
            st.markdown(f"<h5 style='text-align: center;'>Total Dropped Dancers: <b>{summary_data['Total Dropped Dancers']}</b></h5>", unsafe_allow_html=True)

        # Row 3: Acquisition Ratio, Retention Ratio
        col7, col8 = st.columns(2)
        with col7:
            st.markdown(f"<h5 style='text-align: center;'>Acquisition Ratio: <b>{summary_data['Acquisition Ratio']}</b></h5>", unsafe_allow_html=True)
        with col8:
            st.markdown(f"<h5 style='text-align: center;'>Retention Ratio: <b>{summary_data['Retention Ratio']}</b></h5>", unsafe_allow_html=True)

# Tab 3: Group By
with tab3:
    # Add Display toggle
    # Define the display toggle in Streamlit and set groupbyVar based on the selection
    display_toggle = st.radio("Display", options=["Teacher", "City", "Location", "Class"], index=0)
    groupbyVar = display_toggle

    # DANCER ENROLLMENT GRAPH
    st.markdown("<h5 style='text-align: left;'>Dancer Enrollment (Total Slots)</h5>", unsafe_allow_html=True)
    if not filtered_df.empty:
        # Group by x_axisLabel and Teacher to get individual counts per teacher
        grouped_df = filtered_df.groupby(['x_axisLabel', groupbyVar]).agg({'DancerID': 'count'}).reset_index()
        grouped_df.rename(columns={'DancerID': 'Number of Dancers'}, inplace=True)
        grouped_df = grouped_df.merge(df[['x_axisLabel', 'Sort_Key']].drop_duplicates(), on='x_axisLabel')
        grouped_df = grouped_df.sort_values('Sort_Key')

        # Plotting dynamic graph with each teacher as a separate line
        fig = px.line(grouped_df, x='x_axisLabel', y='Number of Dancers', color= groupbyVar, markers=True, 
                    title='Dancer Enrollment (Total Slots)')
        
        fig.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                        line=dict(width=3))

        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Dancers Enrolled",
            xaxis_tickangle=-45,
            template='plotly_white',
            font=dict(size=14, color='black'),
            title_font=dict(size=24, color='white'),
            title_x=0.4,
            xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        # Optional: Add annotations if desired for each data point
        for i, row in grouped_df.iterrows():
            fig.add_annotation(
                x=row['x_axisLabel'],
                y=row['Number of Dancers'],
                text=str(row['Number of Dancers']),
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                xshift=10,
                font=dict(color='white', weight='bold')
            )

        st.plotly_chart(fig, use_container_width=True)
        total_dancers = grouped_df['Number of Dancers'].sum()
        st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Total Enrollment (Slots): {total_dancers}</h5>", unsafe_allow_html=True)
    else:
        st.warning("No data available for the selected filters.")
    #Unique Dancers
    # Display Header for the Chart
    st.markdown("<h5 style='text-align: left;'>Total Unique Dancers</h5>", unsafe_allow_html=True)

    if not filtered_df.empty:
        # Count unique DancerIDs for each Year_Season_Session
        unique_enrollment_df = filtered_df.groupby(['x_axisLabel', groupbyVar]).agg({'DancerID': 'nunique'}).reset_index()
        unique_enrollment_df.rename(columns={'DancerID': 'Number of Unique Dancers'}, inplace=True)

        # Merge to retain Sort_Key for correct ordering and sort
        unique_enrollment_df = unique_enrollment_df.merge(df[['x_axisLabel', 'Sort_Key']].drop_duplicates(), on='x_axisLabel')
        unique_enrollment_df = unique_enrollment_df.sort_values('Sort_Key')

        # Calculate total enrollment across all sessions
        total_unique_dancers = unique_enrollment_df['Number of Unique Dancers'].sum()

        # Plotting the dynamic graph
        fig = px.line(unique_enrollment_df, x='x_axisLabel', y='Number of Unique Dancers', color = groupbyVar,markers=True, 
                    title='Total Unique Dancers')
        
        fig.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                        line=dict(width=3))

        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Unique Dancers Enrolled",
            xaxis_tickangle=-45,
            template='plotly_white',
            font=dict(size=14, color='black'),
            title_font=dict(size=24, color='white'),
            title_x=0.4,
            xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        # Add annotations to each point in the graph
        for _, row in unique_enrollment_df.iterrows():
            fig.add_annotation(
                x=row['x_axisLabel'],
                y=row['Number of Unique Dancers'],
                text=str(row['Number of Unique Dancers']),
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                xshift=10,
                font=dict(color='white', weight='bold')
            )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Display total unique dancers enrolled
        st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Total Unique Dancers Enrolled: {total_unique_dancers}</h5>", unsafe_allow_html=True)

    # Dropped Dancers - Percentage Change from Previous Period
    if not filtered_df.empty: 
        # Calculate the percentage change in unique dancers from the previous period
        unique_enrollment_df['Drop %'] = unique_enrollment_df.groupby(groupbyVar)['Number of Unique Dancers'].pct_change().fillna(0) * 100

        # Plot the percentage drops graph
        fig_drops = px.line(unique_enrollment_df, x='x_axisLabel', y='Drop %', markers=True, color = groupbyVar,
                            title='Percentage Change in Unique Dancers')
        
        fig_drops.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                                line=dict(width=3))

        fig_drops.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Percentage Change in Unique Dancers",
            xaxis_tickangle=-45,
            template='plotly_white',
            font=dict(size=14, color='black'),
            title_font=dict(size=24, color='white'),
            title_x=0.3,
            xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        # Add annotations to each point in the graph
        for _, row in unique_enrollment_df.iterrows():
            fig_drops.add_annotation(
                x=row['x_axisLabel'],
                y=row['Drop %'],
                text=f"{row['Drop %']:.0f}%", 
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                xshift=10,
                font=dict(color='white', weight='bold')
            )

        # Display the drops plot in Streamlit
        st.plotly_chart(fig_drops, use_container_width=True)

        # Calculate total dropped dancers and dropped percentage
        unique_enrollment_df['Drops'] = unique_enrollment_df['Number of Unique Dancers'].diff().fillna(0)
        total_dropped_dancers = unique_enrollment_df['Drops'].sum()

        st.markdown(f"<h5 style='text-align: left;'>Total Unique Students Dropped: {total_dropped_dancers:.0f}</h5>", unsafe_allow_html=True)
    else:
        st.warning("No data available for the selected filters.")

    # DANCER ACQUISITION GRAPH
    st.markdown("<h5 style='text-align: left;'>Brand New Dancers as % of Unique Dancers</h5>", unsafe_allow_html=True)
    if not filtered_df.empty:
        df_sorted = df.sort_values('Sort_Key')
        seen_dancers = set()
        newly_acquired = []
        for _, row in df_sorted.iterrows():
            session_id = row['x_axisLabel']
            dancer_id = row['DancerID']
            if dancer_id not in seen_dancers:
                newly_acquired.append({'x_axisLabel': session_id, 'DancerID': dancer_id, 
                                    'Class': row['Class'], 'Location': row['Location'], 
                                    'Teacher': row['Teacher'], 'Age': row['Age'], 'City': row['City'],
                                    'Reg/NonReg': row['Reg/NonReg']})
                seen_dancers.add(dancer_id)

        acquired_df = pd.DataFrame(newly_acquired)
        acquired_filtered_df = acquired_df[
            (acquired_df['Class'].isin(selected_classes)) &
            (acquired_df['Location'].isin(selected_locations)) &
            (acquired_df['Teacher'].isin(selected_teachers)) &
            (acquired_df['Age'].isin(selected_ages)) &
            (acquired_df['Reg/NonReg'].isin(selected_reg_nonreg)) &
            (acquired_df['City'].isin(filtered_df['City'])) &
            (acquired_df['x_axisLabel'].isin(filtered_df['x_axisLabel']))
        ]

        #Group by x_axisLabel to get count of newly acquired dancers
        acquired_grouped_filtered_df = acquired_filtered_df.groupby(['x_axisLabel', groupbyVar]).agg({'DancerID': 'count'}).reset_index()
        acquired_grouped_filtered_df.rename(columns={'DancerID': 'Newly Acquired Students'}, inplace=True)
        acquired_grouped_filtered_df = acquired_grouped_filtered_df.merge(df[['x_axisLabel', 'Sort_Key']].drop_duplicates(), on='x_axisLabel')
        acquired_grouped_filtered_df = acquired_grouped_filtered_df.sort_values('Sort_Key')

        # Calculate unique dancers for each period
        unique_dancers_df = filtered_df.groupby(['x_axisLabel', groupbyVar]).agg({'DancerID': 'nunique'}).reset_index()
        unique_dancers_df.rename(columns={'DancerID': 'Total Unique Dancers'}, inplace=True)

        # Merge to calculate acquisition as a percentage
        acquisition_percentage_df = acquired_grouped_filtered_df.merge(unique_dancers_df, on=['x_axisLabel', groupbyVar])
        acquisition_percentage_df['Acquisition %'] = (acquisition_percentage_df['Newly Acquired Students'] / 
                                                    acquisition_percentage_df['Total Unique Dancers']) * 100

        # Plot graph
        fig_acquired_filtered = px.line(acquisition_percentage_df, x='x_axisLabel', y='Acquisition %', color = groupbyVar, markers=True, 
                                        title='Brand New Dancers as % of Unique Dancers')
        
        fig_acquired_filtered.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                                            line=dict(width=3))

        fig_acquired_filtered.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Brand New Percentage",
            xaxis_tickangle=-45,
            template='plotly_white',
            font=dict(size=14, color='black'),
            title_font=dict(size=24, color='white'),
            title_x=0.28,
            xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        # Optional: Add annotations for each point
        for i, row in acquisition_percentage_df.iterrows():
            fig_acquired_filtered.add_annotation(
                x=row['x_axisLabel'],
                y=row['Acquisition %'],
                text=f"{row['Acquisition %']:.0f}%",
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
    #if total_dancers > 0:
        #acquisition_ratio = (total_acquired_students / total_dancers) * 100
    #else:
        #acquisition_ratio = 0

    # Display the total acquired students and Acquisition Ratio side by side
    #col1, col2 = st.columns([1, 4])
    #with col1:
    st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Total Acquisition: {total_acquired_students}</h5>", unsafe_allow_html=True)
    #with col2:
        #st.markdown(f"<h5 style='text-align: left; margin-left: 0px;'>Acquisition Ratio: {acquisition_ratio:.2f}%</h5>", unsafe_allow_html=True)

    st.markdown("<h5 style='text-align: left;'>Retention Percentage</h5>", unsafe_allow_html=True)

    if not filtered_df.empty:
        # Merge the unique enrollment and acquisition DataFrames on 'x_axisLabel'
        retention_df = unique_enrollment_df.merge(acquired_grouped_filtered_df, on=['x_axisLabel', groupbyVar], how='left')
        
        # Calculate Retained Dancers as the difference between Total Unique Dancers and Newly Acquired Students
        retention_df['Newly Acquired Students'] = retention_df['Newly Acquired Students'].fillna(0)
        retention_df['Retained Dancers'] = retention_df['Number of Unique Dancers'] - retention_df['Newly Acquired Students']
        
        # Calculate retention as a percentage
        retention_df['Retention %'] = (retention_df['Retained Dancers'] / retention_df['Number of Unique Dancers']) * 100
        
        # Sort the DataFrame based on Sort_Key for proper session ordering
        retention_df = retention_df.merge(df[['x_axisLabel', 'Sort_Key']].drop_duplicates(), on='x_axisLabel')
        retention_df = retention_df.sort_values('Sort_Key')

        # Plotting Retention Percentage Graph
        fig_retention = px.line(retention_df, x='x_axisLabel', y='Retention %', color = groupbyVar, markers=True, 
                                title='Retention Percentage')
        
        fig_retention.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                                    line=dict(width=3))

        fig_retention.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Retention Percentage",
            xaxis_tickangle=-45,
            template='plotly_white',
            font=dict(size=14, color='black'),
            title_font=dict(size=24, color='white'),
            title_x=0.4,
            xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        # Add annotations for each point in the graph
        for _, row in retention_df.iterrows():
            fig_retention.add_annotation(
                x=row['x_axisLabel'],
                y=row['Retention %'],
                text=f"{row['Retention %']:.0f}%",
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                xshift=10,
                font=dict(color='white', weight='bold')
            )

        # Display the retention plot in Streamlit
        st.plotly_chart(fig_retention, use_container_width=True)

        # Display total retention percentage for all periods combined, if desired
        total_retained_dancers = retention_df['Retained Dancers'].sum()
        overall_retention_percentage = (retention_df['Retained Dancers'].sum() / retention_df['Number of Unique Dancers'].sum()) * 100
        st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Overall Retention Percentage: {overall_retention_percentage:.0f}%</h5>", unsafe_allow_html=True)
    else:
        st.warning("No data available for the selected filters.")
    #DROPS SECTION
    # DROPS - Percentage Change in Retained Dancers from Previous Period for each Teacher
    if not retention_df.empty:
        retention_df['Drop %'] = retention_df.groupby(groupbyVar)['Retained Dancers'].pct_change().fillna(0) * 100

        # Plot the Percentage Change in Retained Dancers Graph
        fig_drop_pct = px.line(retention_df, x='x_axisLabel', y='Drop %', color=groupbyVar, markers=True, 
                            title='Percentage Change in Retained Dancers')
        
        fig_drop_pct.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                                line=dict(width=3))
        
        fig_drop_pct.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Percentage Change in Retained Dancers",
            xaxis_tickangle=-45,
            template='plotly_white',
            font=dict(size=14, color='black'),
            title_font=dict(size=24, color='white'),
            title_x=0.31,
            xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        # Add annotations for each point in the graph
        for _, row in retention_df.iterrows():
            fig_drop_pct.add_annotation(
                x=row['x_axisLabel'],
                y=row['Drop %'],
                text=f"{row['Drop %']:.0f}%", 
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                xshift=10,
                font=dict(color='white', weight='bold')
            )

        # Display the percentage change in retention plot in Streamlit
        st.plotly_chart(fig_drop_pct, use_container_width=True)

        # DROPS - Difference in Retained Dancers from Previous Period for each Teacher
        retention_df['Drops'] = retention_df.groupby(groupbyVar)['Retained Dancers'].diff().fillna(0)

        # Plot the Difference in Retained Dancers Graph
        fig_drop_diff = px.line(retention_df, x='x_axisLabel', y='Drops', color=groupbyVar, markers=True, 
                                title='Difference in Retained Dancers')
        
        fig_drop_diff.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                                    line=dict(width=3))
        
        fig_drop_diff.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Difference in Retained Dancers",
            xaxis_tickangle=-45,
            template='plotly_white',
            font=dict(size=14, color='black'),
            title_font=dict(size=24, color='white'),
            title_x=0.31,
            xaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            yaxis=dict(showgrid=True, zeroline=False, showline=True, linewidth=2, linecolor='lightgrey'),
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        # Add annotations for each point in the graph
        for _, row in retention_df.iterrows():
            fig_drop_diff.add_annotation(
                x=row['x_axisLabel'],
                y=row['Drops'],
                text=f"{row['Drops']:.0f}", 
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                xshift=10,
                font=dict(color='white', weight='bold')
            )

        # Display the difference in retention plot in Streamlit
        st.plotly_chart(fig_drop_diff, use_container_width=True)

        # Display total dropped dancers count for all teachers
        total_dropped_dancers = retention_df['Drops'].sum()
        st.markdown(f"<h5 style='text-align: left;'>Total Retained Dancers Dropped: {total_dropped_dancers:.0f}</h5>", unsafe_allow_html=True)
    else:
        st.warning("No data available for the selected filters.")
