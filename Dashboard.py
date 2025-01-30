import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Import Plotly graph objects
import os
from datetime import datetime
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

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
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Summary Statistics", "Group By", "Format Data"])

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
        # Fall 1 -> Fall 2 -> Winter 1 -> Winter 2 -> Spring 1 -> Spring 2 -> Summer 1 -> Summer 2
        school_year_order = [
            ('Fall', 1), ('Fall', 2), ('Winter', 1), ('Winter', 2),
            ('Spring', 1), ('Spring', 2), ('Summer', 1), ('Summer', 2), 
        ]
        
        # Create a mapping of periods to school years
        school_year_mapping = {}
        school_year_string_mapping = {}
        
        years = sorted(df['Year'].unique())  # Loop through years
        
        for year_start in years:
            # Define start periods for the current school year
            start_periods = [
                (year_start, 'Fall', 1), (year_start, 'Fall', 2),      # Fall in year_start
                (year_start + 1, 'Winter', 1), (year_start + 1, 'Winter', 2),  # Winter in year_start + 1
                (year_start + 1, 'Spring', 1), (year_start + 1, 'Spring', 2),  # Spring in year_start + 1
                (year_start + 1, 'Summer', 1), (year_start + 1, 'Summer', 2),  # Summer in year_start + 1
            ]
            
            # Define the school year string in 'YYYY-YYYY' format
            school_year_str = f"{year_start}-{year_start - 1999}"
            
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


    # Add School Year
    df = addSchoolYear(df)

    # Convert School Year and Session to integers (if they are not strings)
    df['School Year'] = df['School Year'].fillna(0).astype(int)

    df['Session'] = df['Session'].astype(int)

    # Convert Season to string (if needed)
    df['Season'] = df['Season'].astype(str)

    # Order Seasons
    season_order = {'Fall': 1, 'Winter': 2, 'Spring': 3, 'Summer': 4}
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

    df['Sort_Key_Session'] = df['School Year'] * 100 + df['Season_Order'] * 10 + df['Session']  # Session-based sort key
    df['x_axisLabelSession'] = df['Year_Season_Session']

    # Sort the DataFrame
    df = df.sort_values('Sort_Key')

    st.write(df)

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
    col_city, col_class, col_location, col_teacher, col_reg_nonreg = st.columns(5)

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

    #with col_age:
        #selected_ages = select_all_option_expander('Age', df['Age'].unique(), sort_order='numerical')
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
                    (df['City'].isin(selected_cities)) &
                    (df['Session'].isin(selected_sessions))]

    # Initialize total_dancers
    total_dancers = 0 

# STUDENT ENROLLMENT GRAPH
    st.markdown("<h5 style='text-align: left;'></h5>", unsafe_allow_html=True)
    if not filtered_df.empty:
        grouped_df = filtered_df.groupby('x_axisLabel').agg({'DancerID': 'count'}).reset_index()
        grouped_df.rename(columns={'DancerID': 'Number of Dancers'}, inplace=True)
        grouped_df = grouped_df.merge(df[['x_axisLabel', 'Sort_Key']].drop_duplicates(), on='x_axisLabel')
        grouped_df = grouped_df.sort_values('Sort_Key')
        total_dancers = grouped_df['Number of Dancers'].sum()

        # Plotting  dynamic graph
        fig = px.line(grouped_df, x='x_axisLabel', y='Number of Dancers', markers=True, 
                    title='Student Enrollment (Total Slots Filled)')
        
        fig.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                        line=dict(width=3, color='steelblue'))

        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Students Enrolled",
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
        st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Total Slots Filled: {total_dancers}</h5>", unsafe_allow_html=True)
    
#UNIQUE STUDENTS CALCULATION

        unique_enrollment_df = filtered_df.groupby(['Year_Season_Session', 'School Year String']).agg({'DancerID': 'nunique'}).reset_index()
        unique_enrollment_df.rename(columns={'DancerID': 'Number of Unique Dancers'}, inplace=True)
        drops_df = unique_enrollment_df


        # Determine x_axisLabel based on toggle
        if display_toggle == "School Year":
            # Rename 'School Year' to x_axisLabel
            unique_enrollment_df['x_axisLabel'] = unique_enrollment_df['School Year String'].astype(str)
            unique_enrollment_df = unique_enrollment_df.groupby('x_axisLabel').agg({'Number of Unique Dancers': 'sum'}).reset_index()
        else:
            # Rename 'Year_Season_Session' to x_axisLabel
            unique_enrollment_df['x_axisLabel'] = unique_enrollment_df['Year_Season_Session']


        # Merge to retain Sort_Key for correct ordering and sort
        unique_enrollment_df = unique_enrollment_df.merge(df[['x_axisLabel', 'Sort_Key']].drop_duplicates(), on='x_axisLabel')
        unique_enrollment_df = unique_enrollment_df.sort_values('Sort_Key')

        # Calculate total enrollment across all sessions
        total_unique_dancers = unique_enrollment_df['Number of Unique Dancers'].sum()

# BRAND NEW STUDENTS
        # BRAND NEW STUDENTS CALCULATION
        # Sort the DataFrame for consistent processing
        df_sorted = df.sort_values('Year_Season_Session')

        # Identify brand new students
        seen_dancers = set()
        newly_acquired = []

        for _, row in df_sorted.iterrows():
            session_id = row['Year_Season_Session']  # Always use Year_Season_Session as base
            dancer_id = row['DancerID']
            if dancer_id not in seen_dancers:
                newly_acquired.append({'Year_Season_Session': session_id, 'School Year String': row['School Year String'],
                                    'DancerID': dancer_id, 'Class': row['Class'], 'Location': row['Location'], 
                                    'Teacher': row['Teacher'], 'Age': row['Age'], 
                                    'Reg/NonReg': row['Reg/NonReg']})
                seen_dancers.add(dancer_id)

        # Create a DataFrame for newly acquired students
        acquired_df = pd.DataFrame(newly_acquired)

        acquired_filtered_df = acquired_df[
            (acquired_df['Class'].isin(selected_classes)) &
            (acquired_df['Location'].isin(selected_locations)) &
            (acquired_df['Teacher'].isin(selected_teachers)) &
            (acquired_df['Age'].isin(selected_ages)) &
            (acquired_df['Reg/NonReg'].isin(selected_reg_nonreg)) &
            (acquired_df['Year_Season_Session'].isin(filtered_df['Year_Season_Session']))
        ]

        # Count new dancers at the Year_Season_Session level
        new_students_df = acquired_filtered_df.groupby(['Year_Season_Session', 'School Year String']).agg({'DancerID': 'nunique'}).reset_index()
        new_students_df.rename(columns={'DancerID': 'Number of New Students'}, inplace=True)

        new_students_drops_df = new_students_df

        # Determine x_axisLabel based on toggle
        if display_toggle == "School Year":
            # Rename 'School Year' to x_axisLabel
            new_students_df['x_axisLabel'] = new_students_df['School Year String'].astype(str)
            new_stuents_df = new_students_df.groupby('x_axisLabel').agg({'Number of New Students': 'sum'}).reset_index()
        else:
            # Rename 'Year_Season_Session' to x_axisLabel
            new_students_df['x_axisLabel'] = new_students_df['Year_Season_Session']

        new_students_df = new_students_df.merge(df[['x_axisLabel', 'Sort_Key']].drop_duplicates(), on='x_axisLabel')
        new_students_df = new_students_df.sort_values('Sort_Key')
        new_students_df = new_students_df.groupby('x_axisLabel', as_index=False).agg({'Number of New Students': 'sum'})

        # Calculate total new students
        total_new_students = new_students_df['Number of New Students'].sum()

        # Merge to calculate acquisition as a percentage
        acquisition_percentage_df = new_students_df.merge(unique_enrollment_df, on='x_axisLabel')
        acquisition_percentage_df['Acquisition %'] = (
            acquisition_percentage_df['Number of New Students'] /
            acquisition_percentage_df['Number of Unique Dancers']
        ) * 100

        # Plot graph for Brand New Students Acquisition Percentage
        fig_new_students = px.line(
            acquisition_percentage_df,
            x='x_axisLabel',
            y='Acquisition %',
            markers=True,
            title='Brand New Students (Acquisition Percentage)'
        )

        fig_new_students.update_traces(
            marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
            line=dict(width=3, color='mediumseagreen')
        )

        fig_new_students.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Acquisition Percentage (%)",
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

        # Add annotations to each point with percentage and total new students
        for _, row in acquisition_percentage_df.iterrows():
            fig_new_students.add_annotation(
                x=row['x_axisLabel'],
                y=row['Acquisition %'],
                text=f"{row['Acquisition %']:.0f}% ({row['Number of New Students']})",  # Show both percentage and total
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                xshift=10,
                font=dict(color='white', weight='bold')
            )

        # Display the plot in Streamlit
        st.plotly_chart(fig_new_students, use_container_width=True)

        # Display total new students
        total_new_students = acquisition_percentage_df['Number of New Students'].sum()
        st.markdown(
            f"<h5 style='text-align: left; margin-left: 75px;'>Total Brand New Students: {total_new_students}</h5>",
            unsafe_allow_html=True
        )
#UNIQUE STUDENTS GRAPH
    # Plotting the dynamic graph
        fig = px.line(unique_enrollment_df, x='x_axisLabel', y='Number of Unique Dancers', markers=True, 
                    title='Total Unique Students')
        
        fig.update_traces(marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
                        line=dict(width=3, color='pink'))

        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Unique Students Enrolled",
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
        st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Total Unique Students: {total_unique_dancers}</h5>", unsafe_allow_html=True)
#RETENTION
        # Merge the unique enrollment and acquisition DataFrames on 'x_axisLabel'
        retention_df = unique_enrollment_df.merge(new_students_df, on='x_axisLabel', how='left')
        
        # Calculate Retained Dancers as the difference between Total Unique Dancers and Newly Acquired Students
        retention_df['Retained Students'] = retention_df['Number of Unique Dancers'] - retention_df['Number of New Students']
        
        # Calculate retention as a percentage
        retention_df['Retention %'] = (retention_df['Retained Students'] / retention_df['Number of Unique Dancers']) * 100
        
        # Sort the DataFrame based on Sort_Key for proper session ordering
        #retention_df = retention_df.merge(df[['x_axisLabel', 'Sort_Key']].drop_duplicates(), on='x_axisLabel')
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
            title_x=0.42,
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
                text=f"{row['Retention %']:.0f}% ({row['Retained Students']})",
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                xshift=10,
                font=dict(color='white', weight='bold')
            )

        # Display the retention plot in Streamlit
        st.plotly_chart(fig_retention, use_container_width=True)

        # Display total retention percentage for all periods combined, if desired
        total_retained_dancers = retention_df['Retained Students'].sum()
        overall_retention_percentage = (retention_df['Retained Students'].sum() / retention_df['Number of Unique Dancers'].sum()) * 100
        st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Percent of Students Retained: {overall_retention_percentage:.0f}%</h5>", unsafe_allow_html=True)

# DROPS: Unique Students P1 - Retained Students P2
        st.markdown(f"<h5 style='text-align: left; margin-left: 75px;'>Drops: Unique Students (previous period) - Retained Students (current period)</h5>", unsafe_allow_html=True)
    # PERCENT CHANGE
        drops_df = drops_df.merge(new_students_drops_df, on='Year_Season_Session', how='left')

        drops_df['x_axisLabelSession'] = drops_df['Year_Season_Session']

        drops_df = drops_df.merge(df[['x_axisLabelSession', 'Sort_Key_Session']].drop_duplicates(), on='x_axisLabelSession')

        drops_df = drops_df.sort_values('Sort_Key_Session')

        drops_df['Retained Students'] = drops_df['Number of Unique Dancers'] - drops_df['Number of New Students']

        # Ensure unique enrollment data is sorted correctly and includes a shifted column for previous period's unique dancers
        drops_df['Previous Period Unique Dancers'] = drops_df['Number of Unique Dancers'].shift(1)

        # Calculate Drops as the difference between Previous Period Unique Dancers and current period's Retained Dancers
        drops_df['Drops'] = drops_df['Previous Period Unique Dancers'] - drops_df['Retained Students']

        st.write(drops_df)

        if display_toggle == "School Year":
            # Aggregate data by School Year
            drops_df['x_axisLabel'] = drops_df['School Year String_x']
            drops_df = drops_df.groupby('x_axisLabel', as_index=False).agg({
                'Drops': 'sum',
            })
        else:
            # Use Year_Season_Session for the x-axis
            drops_df['x_axisLabel'] = drops_df['Year_Season_Session']


        # Calculate the percentage change in drops
        drops_df['Drop %'] = drops_df['Drops'].pct_change().fillna(0) * 100

        # Plot the Percentage Change in Drops Graph
        fig_drop_pct = px.line(
            drops_df,
            x='x_axisLabel',
            y='Drop %',
            markers=True,
            title='Percentage Change in Drops'
        )

        fig_drop_pct.update_traces(
            marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
            line=dict(width=3, color='red')
        )

        fig_drop_pct.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Percentage Change in Drops",
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

        # Add annotations to each point in the graph to display the percentage change in drops
        for _, row in drops_df.iterrows():
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

        # Display the percentage change in drops plot in Streamlit
        st.plotly_chart(fig_drop_pct, use_container_width=True)


    # DROPS: Count
        # Plot the count of drops
        fig_drop_diff = px.line(drops_df, x='x_axisLabel', y='Drops', markers=True, 
                                title='Number of Drops')

        fig_drop_diff.update_traces(
            marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
            line=dict(width=3, color='blue')
        )

        fig_drop_diff.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Count of Dropped Students",
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

        # Add annotations for each point in the graph
        for _, row in drops_df.iterrows():
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
        total_dropped_dancers = drops_df['Drops'].sum()
        st.markdown(f"<h5 style='text-align: left;'>Total Dropped Students: {total_dropped_dancers:.0f}</h5>", unsafe_allow_html=True)
    else:
        st.warning("No data available for the selected filters.")
#Tab 2: Summary Statistics
with tab2:
        # Define summary data for easy access
        summary_data = {
            "Total Enrollment": total_dancers,
            "Total Unique Dancers Enrolled": total_unique_dancers,
            "Average Number of Slots Attended": f"{total_dancers / total_unique_dancers:.0f}",
            "Total Acquisition": total_new_students,
            "Total Retained Dancers": total_retained_dancers,
            "Total Dropped Dancers": f"{total_dropped_dancers:.0f}",
            "Acquisition Ratio": f"{total_new_students / total_unique_dancers:.0%}",
            "Retention Ratio": f"{total_retained_dancers / total_unique_dancers:.0%}",
        }

        st.markdown(f"<h5 style='text-align: left;'>Total Slots Filled: <b>{summary_data['Total Enrollment']}</b></h5>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align: left;'>Total Brand New Students: <b>{summary_data['Total Acquisition']}</b></h5>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align: left;'>Average Slots Attended: <b>{summary_data['Average Number of Slots Attended']}</b></h5>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align: left;'>Total Unique Students: <b>{summary_data['Total Unique Dancers Enrolled']}</b></h5>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align: left;'>Total Retained Students: <b>{summary_data['Total Retained Dancers']}</b></h5>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align: left;'>Total Dropped Students: <b>{summary_data['Total Dropped Dancers']}</b></h5>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align: left;'>Brand New Students (Acquisition %): <b>{summary_data['Acquisition Ratio']}</b></h5>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align: left;'>Percent of Students Retained: <b>{summary_data['Retention Ratio']}</b></h5>", unsafe_allow_html=True)

        # Function to classify students by attendance and create histograms
        def classify_students_by_attendance(df):
            # Set custom thresholds for each type
            thresholds = {
                'Took_8': 8,
                'Took_7': 7,
                'Took_6': 6, 
                'Took_5': 5,  
                'Took_4': 4,
                'Took_3': 3,
                'Took_2': 2,
                'Took_1': 1,
                'Took_0': 0,
            }

            # Remove duplicates for the same student in the same 'School Year', 'Season', and 'Session'
            df_unique = df.drop_duplicates(subset=['School Year', 'Season', 'Session', 'DancerID'])

            # Group by 'School Year', 'Season', 'Session', and 'DancerID', and count occurrences
            attendance_counts = df_unique.groupby(['School Year', 'Season', 'Session', 'DancerID']).size().reset_index(name='Count')

            # Ensure 'School Year' is of integer type
            attendance_counts['School Year'] = attendance_counts['School Year'].astype(int)

            # Loop through each school year in the DataFrame
            years = sorted(df['School Year'].unique())
            years = [int(year) for year in years]

            # Define the correct order for "Took" categories
            took_order = ['Took 0', 'Took 1', 'Took 2', 'Took 3', 'Took 4', 'Took 5', 'Took 6', 'Took 7', 'Took 8']

            for year in years:
                previous_year_attendance = attendance_counts[attendance_counts['School Year'] == year].copy()

                # Count occurrences of each DancerID
                dancer_counts = previous_year_attendance.groupby('DancerID')['Count'].sum().reset_index(name='Attendance Count')

                # Classify students based on attendance count
                dancer_counts['Took'] = dancer_counts['Attendance Count'].apply(lambda x: classify_student(x, thresholds))

                # Optionally, you can merge this back to your original DataFrame if needed
                df = df.merge(dancer_counts[['DancerID', 'Took']], on='DancerID', how='left')
                df.rename(columns={'Took': f'Took_{year}'}, inplace=True)

                # Calculate counts and percentages
                type_counts = dancer_counts['Took'].value_counts().reset_index()
                type_counts.columns = ['Took', 'Count']
                total_count = type_counts['Count'].sum()
                type_counts['Percentage'] = (type_counts['Count'] / total_count) * 100  # Calculate percentages

                # Ensure "Took" is ordered correctly
                type_counts['Took'] = pd.Categorical(type_counts['Took'], categories=took_order, ordered=True)
                type_counts = type_counts.sort_values('Took')

                # Create the histogram with Plotly
                fig = px.bar(
                    type_counts, 
                    x='Took', 
                    y='Percentage',  # Use percentage for the y-axis
                    title=f'How many Sessions each unique Student Took {year}-{year + 1}',
                    labels={'Took': 'Sessions Took', 'Percentage': 'Percentage of Total (%)'}
                )

                # Add labels above each bar showing the count and percentage
                fig.update_traces(
                    text=type_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1),  # Show count and percentage
                    textposition='outside',  # Position labels outside the bars
                    cliponaxis=False
                )

                fig.update_layout(
                    xaxis_title="Sessions Took",
                    yaxis_title="Percentage of Total (%)",
                    template='plotly_white',
                    title_x=0.4,
                    margin=dict(t=80),
                    xaxis=dict(categoryorder='array', categoryarray=took_order)  # Ensure correct order
                )

                # Display the figure in Streamlit
                st.plotly_chart(fig, use_container_width=True)

            return df  # Return the DataFrame with classifications (moved outside the loop)

        # Function to classify based on attendance thresholds
        def classify_student(attendance, thresholds):
            if attendance >= thresholds['Took_8']:
                return 'Took 8'
            elif attendance == thresholds['Took_7']:
                return 'Took 7'
            elif attendance == thresholds['Took_6']:
                return 'Took 6'
            elif attendance == thresholds['Took_5']:
                return 'Took 5'
            elif attendance == thresholds['Took_4']:
                return 'Took 4'
            elif attendance == thresholds['Took_3']:
                return 'Took 3'
            elif attendance == thresholds['Took_2']:
                return 'Took 2'
            elif attendance == thresholds['Took_1']:
                return 'Took 1'
            else:
                return 'Took 0'

        # Example usage in Streamlit
        st.title("Unique Students by Number of Sessions")
        if not filtered_df.empty:
            SessionCounts_df = classify_students_by_attendance(filtered_df)
        else:
            st.warning("No data available for the selected filters.")
# Tab 3: Group By
with tab3:
    # Add Display toggle
    # Define the display toggle in Streamlit and set groupbyVar based on the selection
    display_toggleVar = st.radio("Display", options=["City", "Teacher", "Location", "Class"], index=0)
    groupbyVar = display_toggleVar

# STUDENT ENROLLMENT GRAPH
    st.markdown("<h5 style='text-align: left;'></h5>", unsafe_allow_html=True)

    if not filtered_df.empty:
        # Grouping by x_axisLabel and groupbyVar, counting unique DancerID
        grouped_df = (
            filtered_df.groupby(['x_axisLabel', groupbyVar])
            .agg({'DancerID': 'count'})
            .reset_index()
        )
        grouped_df.rename(columns={'DancerID': 'Number of Dancers'}, inplace=True)

        # Generate a complete grid of all combinations of x_axisLabel and groupbyVar
        unique_x_axis = grouped_df['x_axisLabel'].unique()
        unique_groupby_values = filtered_df[groupbyVar].unique()
        complete_grid = pd.MultiIndex.from_product([unique_x_axis, unique_groupby_values], names=['x_axisLabel', groupbyVar]).to_frame(index=False)

        # Merge the complete grid with the grouped data
        grouped_df = complete_grid.merge(grouped_df, on=['x_axisLabel', groupbyVar], how='left')

        # Merge with Sort_Key for sorting purposes
        grouped_df = grouped_df.merge(
            df[['x_axisLabel', 'Sort_Key']].drop_duplicates(),
            on='x_axisLabel'
        )
        grouped_df = grouped_df.sort_values('Sort_Key')

        # Total dancers calculation
        total_dancers = grouped_df['Number of Dancers'].sum()

        # Plotting dynamic graph
        fig = px.line(
            grouped_df,
            x='x_axisLabel',
            y='Number of Dancers',
            color=groupbyVar,  # Differentiate lines by groupbyVar
            markers=True,
            title='Student Enrollment (Total Slots Filled)'
        )

        fig.update_traces(
            connectgaps=True,
            marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
            line=dict(width=3)
        )

        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Students Enrolled",
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

        # Adding annotations for each data point
        for i, row in grouped_df.iterrows():
            if not pd.isna(row['Number of Dancers']):  # Avoid annotating missing points
                fig.add_annotation(
                    x=row['x_axisLabel'],
                    y=row['Number of Dancers'],
                    text=f"{int(row['Number of Dancers'])}",  # Format as an integer
                    showarrow=False,
                    xanchor='left',
                    yanchor='middle',
                    xshift=10,
                    font=dict(color='white', weight='bold')
                )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"<h5 style='text-align: left; margin-left: 75px;'>Total Slots Filled: {int(total_dancers)}</h5>",
            unsafe_allow_html=True
        )
# UNIQUE DANCERS
    # Grouping by Year_Season_Session, School Year String, and groupbyVar
    unique_enrollment_df = (
        filtered_df.groupby(['Year_Season_Session', 'School Year String', groupbyVar])
        .agg({'DancerID': 'nunique'})
        .reset_index()
    )
    unique_enrollment_df.rename(columns={'DancerID': 'Number of Unique Dancers'}, inplace=True)
    drops_df = unique_enrollment_df

    # Generate a complete grid of all combinations of x_axisLabel and groupbyVar
    if display_toggle == "School Year":
        # Rename 'School Year String' to x_axisLabel
        unique_enrollment_df['x_axisLabel'] = unique_enrollment_df['School Year String'].astype(str)
        unique_x_axis = unique_enrollment_df['x_axisLabel'].unique()
    else:
        # Rename 'Year_Season_Session' to x_axisLabel
        unique_enrollment_df['x_axisLabel'] = unique_enrollment_df['Year_Season_Session']
        unique_x_axis = unique_enrollment_df['x_axisLabel'].unique()

    unique_groupby_values = filtered_df[groupbyVar].unique()

    # Create a DataFrame with all combinations of x_axisLabel and groupbyVar
    complete_grid = pd.MultiIndex.from_product([unique_x_axis, unique_groupby_values], names=['x_axisLabel', groupbyVar]).to_frame(index=False)

    # Merge the complete grid with the grouped data
    unique_enrollment_df = complete_grid.merge(unique_enrollment_df, on=['x_axisLabel', groupbyVar], how='left')

    # Merge to retain Sort_Key for correct ordering and sort
    unique_enrollment_df = unique_enrollment_df.merge(
        df[['x_axisLabel', 'Sort_Key']].drop_duplicates(),
        on='x_axisLabel'
    )
    unique_enrollment_df = unique_enrollment_df.sort_values('Sort_Key')

    # Calculate total enrollment across all sessions
    total_unique_dancers = unique_enrollment_df['Number of Unique Dancers'].sum()

    # Group by x_axisLabel and groupbyVar
    grouped_unique_enrollment_df = (
        unique_enrollment_df.groupby(['x_axisLabel', groupbyVar], as_index=False)
        .agg({'Number of Unique Dancers': 'sum'})
    )

# BRAND NEW STUDENTS
    # Sort the DataFrame for consistent processing
    df_sorted = df.sort_values('Year_Season_Session')

    # Identify brand new students
    seen_dancers = set()
    newly_acquired = []

    for _, row in df_sorted.iterrows():
        session_id = row['Year_Season_Session']  # Always use Year_Season_Session as base
        dancer_id = row['DancerID']
        if dancer_id not in seen_dancers:
            newly_acquired.append({
                'Year_Season_Session': session_id,
                'School Year String': row['School Year String'],
                'DancerID': dancer_id,
                'Class': row['Class'],
                'Location': row['Location'],
                'Teacher': row['Teacher'],
                'Age': row['Age'],
                'Reg/NonReg': row['Reg/NonReg'],
                'City': row['City']
            })
            seen_dancers.add(dancer_id)

    # Create a DataFrame for newly acquired students
    acquired_df = pd.DataFrame(newly_acquired)

    # Apply filters for selected attributes
    acquired_filtered_df = acquired_df[
        (acquired_df['Class'].isin(selected_classes)) &
        (acquired_df['Location'].isin(selected_locations)) &
        (acquired_df['Teacher'].isin(selected_teachers)) &
        (acquired_df['Age'].isin(selected_ages)) &
        (acquired_df['Reg/NonReg'].isin(selected_reg_nonreg)) &
        (acquired_df['City'].isin(selected_cities)) &
        (acquired_df['Year_Season_Session'].isin(filtered_df['Year_Season_Session']))
    ]

    # Count new dancers at the Year_Season_Session, School Year String, and groupbyVar levels
    new_students_df = (
        acquired_filtered_df.groupby(['Year_Season_Session', 'School Year String', groupbyVar])
        .agg({'DancerID': 'nunique'})
        .reset_index()
    )
    new_students_df.rename(columns={'DancerID': 'Number of New Students'}, inplace=True)

    new_students_drops_df = new_students_df

    # Generate a complete grid of all combinations of x_axisLabel and groupbyVar
    if display_toggle == "School Year":
        # Rename 'School Year String' to x_axisLabel
        new_students_df['x_axisLabel'] = new_students_df['School Year String'].astype(str)
        unique_x_axis = new_students_df['x_axisLabel'].unique()
    else:
        # Rename 'Year_Season_Session' to x_axisLabel
        new_students_df['x_axisLabel'] = new_students_df['Year_Season_Session']
        unique_x_axis = new_students_df['x_axisLabel'].unique()

    unique_groupby_values = filtered_df[groupbyVar].unique()

    # Create a DataFrame with all combinations of x_axisLabel and groupbyVar
    complete_grid = pd.MultiIndex.from_product([unique_x_axis, unique_groupby_values], names=['x_axisLabel', groupbyVar]).to_frame(index=False)

    # Merge the complete grid with the grouped data
    new_students_df = complete_grid.merge(new_students_df, on=['x_axisLabel', groupbyVar], how='left')

    # Merge to retain Sort_Key for correct ordering and sort
    new_students_df = new_students_df.merge(
        df[['x_axisLabel', 'Sort_Key']].drop_duplicates(),
        on='x_axisLabel'
    )

    new_students_retention_df = new_students_df

    new_students_df = (
        new_students_df.groupby(['x_axisLabel', groupbyVar, 'Sort_Key'], as_index=False)
        .agg({'Number of New Students': 'sum'})
    )

    # Calculate total new students
    total_new_students = new_students_df['Number of New Students'].sum()

    # Merge to calculate acquisition as a percentage
    acquisition_percentage_df = new_students_df.merge(grouped_unique_enrollment_df, on=['x_axisLabel', groupbyVar])
    acquisition_percentage_df['Acquisition %'] = (
        acquisition_percentage_df['Number of New Students'] /
        acquisition_percentage_df['Number of Unique Dancers']
    ) * 100

    # Plot graph for Brand New Students Acquisition Percentage
    fig_new_students = px.line(
        acquisition_percentage_df,
        x='x_axisLabel',
        y='Acquisition %',
        color=groupbyVar,  # Differentiate lines by groupbyVar
        markers=True,
        title='Brand New Students (Acquisition Percentage)'
    )

    fig_new_students.update_traces(
        connectgaps=True,
        marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
        line=dict(width=3)
    )

    fig_new_students.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Acquisition Percentage (%)",
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

    # Add annotations to each point with percentage and total new students
    for _, row in acquisition_percentage_df.iterrows():
        fig_new_students.add_annotation(
            x=row['x_axisLabel'],
            y=row['Acquisition %'],
            text=f"{row['Acquisition %']:.0f}% ({int(row['Number of New Students']) if not pd.isna(row['Number of New Students']) else 'N/A'})",
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            xshift=10,
            font=dict(color='white', weight='bold')
        )

    # Display the plot in Streamlit
    st.plotly_chart(fig_new_students, use_container_width=True)

    # Display total new students
    total_new_students = acquisition_percentage_df['Number of New Students'].sum()
    st.markdown(
        f"<h5 style='text-align: left; margin-left: 75px;'>Total Brand New Students: {int(total_new_students)}</h5>",
        unsafe_allow_html=True
    )
# UNIQUE STUDENTS GRAPH

    # Plotting the dynamic graph
    fig = px.line(
        grouped_unique_enrollment_df,
        x='x_axisLabel',
        y='Number of Unique Dancers',
        color=groupbyVar,  # Differentiate lines by groupbyVar
        markers=True,
        title='Total Unique Students'
    )

    fig.update_traces(
        connectgaps = True,
        marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
        line=dict(width=3)
    )

    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Unique Students Enrolled",
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
    for _, row in grouped_unique_enrollment_df.iterrows():
        fig.add_annotation(
            x=row['x_axisLabel'],
            y=row['Number of Unique Dancers'],
            text=int(row['Number of Unique Dancers']),
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            xshift=10,
            font=dict(color='white', weight='bold')
        )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display total unique dancers enrolled
    total_unique_dancers = grouped_unique_enrollment_df['Number of Unique Dancers'].sum()
    st.markdown(
        f"<h5 style='text-align: left; margin-left: 75px;'>Total Unique Students: {int(total_unique_dancers)}</h5>",
        unsafe_allow_html=True
    )
# RETENTION
    # Merge the unique enrollment and acquisition DataFrames on 'x_axisLabel' and 'groupbyVar'
    retention_df = unique_enrollment_df.merge(new_students_retention_df, on=['x_axisLabel', groupbyVar, 'Sort_Key'], how='left')

    retention_df = (
        retention_df.groupby(['x_axisLabel', groupbyVar, 'Sort_Key'], as_index=False)
        .agg({'Number of New Students': 'sum', 'Number of Unique Dancers': 'sum'})
    )

    retention_df = retention_df.sort_values('Sort_Key')

    # Calculate Retained Dancers as the difference between Total Unique Dancers and Newly Acquired Students
    retention_df['Retained Students'] = retention_df['Number of Unique Dancers'] - retention_df['Number of New Students']

    # Calculate retention as a percentage
    retention_df['Retention %'] = (retention_df['Retained Students'] / retention_df['Number of Unique Dancers']) * 100

    # Sort the DataFrame based on Sort_Key for proper session ordering
    retention_df = retention_df.sort_values('Sort_Key')

    # Plotting Retention Percentage Graph
    fig_retention = px.line(
        retention_df,
        x='x_axisLabel',
        y='Retention %',
        color=groupbyVar,  # Differentiate lines by groupbyVar
        markers=True,
        title='Retention Percentage'
    )

    fig_retention.update_traces(
        connectgaps=True,
        marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
        line=dict(width=3)
    )

    fig_retention.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Retention Percentage",
        xaxis_tickangle=-45,
        template='plotly_white',
        font=dict(size=14, color='black'),
        title_font=dict(size=24, color='white'),
        title_x=0.42,
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
            text=f"{row['Retention %']:.0f}% ({int(row['Retained Students']) if not pd.isna(row['Retained Students']) else 'N/A'})",
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            xshift=10,
            font=dict(color='white', weight='bold')
        )

    # Display the retention plot in Streamlit
    st.plotly_chart(fig_retention, use_container_width=True)

    # Display total retention percentage for all periods combined, if desired
    total_retained_dancers = retention_df['Retained Students'].sum()
    overall_retention_percentage = (
        retention_df['Retained Students'].sum() / retention_df['Number of Unique Dancers'].sum()
    ) * 100
    st.markdown(
        f"<h5 style='text-align: left; margin-left: 75px;'>Percent of Students Retained: {overall_retention_percentage:.0f}%</h5>",
        unsafe_allow_html=True
    )
# DROPS: Unique Students P1 - Retained Students P2
    st.markdown(
        f"<h5 style='text-align: left; margin-left: 75px;'>Drops: Unique Students (previous period) - Retained Students (current period)</h5>",
        unsafe_allow_html=True
    )

    # Merge the unique enrollment and acquisition DataFrames, including groupbyVar
    drops_df = drops_df.merge(new_students_drops_df, on=['Year_Season_Session', groupbyVar], how='left')

    drops_df['x_axisLabelSession'] = drops_df['Year_Season_Session']

    drops_df = drops_df.merge(
        df[['x_axisLabelSession', 'Sort_Key_Session']].drop_duplicates(),
        on='x_axisLabelSession'
    )

    # Generate a complete grid of all combinations of x_axisLabel and groupbyVar
    if display_toggle == "School Year":
        drops_df['x_axisLabel'] = drops_df['School Year String_x']
        unique_x_axis = drops_df['x_axisLabel'].unique()
    else:
        drops_df['x_axisLabel'] = drops_df['Year_Season_Session']
        unique_x_axis = drops_df['x_axisLabel'].unique()

    unique_groupby_values = drops_df[groupbyVar].unique()

    # Create a DataFrame with all combinations of x_axisLabel and groupbyVar
    complete_grid = pd.MultiIndex.from_product([unique_x_axis, unique_groupby_values], names=['x_axisLabel', groupbyVar]).to_frame(index=False)

    # Merge the complete grid with drops_df to include missing combinations
    drops_df = complete_grid.merge(drops_df, on=['x_axisLabel', groupbyVar], how='left')

    # Sort and calculate drops logic
    drops_df = drops_df.sort_values('Sort_Key_Session')

    # Ensure unique enrollment data is sorted correctly and includes a shifted column for previous period's unique dancers
    drops_df['Retained Students'] = drops_df['Number of Unique Dancers'] - drops_df['Number of New Students']
    drops_df['Previous Period Unique Dancers'] = drops_df.groupby(groupbyVar)['Number of Unique Dancers'].shift(1)

    # Calculate Drops as the difference between Previous Period Unique Dancers and current period's Retained Dancers
    drops_df['Drops'] = drops_df['Previous Period Unique Dancers'] - drops_df['Retained Students']

    if display_toggle == "School Year":
        # Aggregate data by School Year and groupbyVar
        drops_df = drops_df.groupby(['x_axisLabel', groupbyVar], as_index=False).agg({
            'Drops': 'sum',
        })
    else:
        # Use Year_Season_Session for the x-axis
        drops_df['x_axisLabel'] = drops_df['Year_Season_Session']

    # Calculate the percentage change in drops by groupbyVar
    drops_df['Drop %'] = drops_df.groupby(groupbyVar)['Drops'].pct_change().fillna(0) * 100

    # Plot the Percentage Change in Drops Graph
    fig_drop_pct = px.line(
        drops_df,
        x='x_axisLabel',
        y='Drop %',
        color=groupbyVar,  # Differentiate lines by groupbyVar
        markers=True,
        title='Percentage Change in Drops'
    )

    fig_drop_pct.update_traces(
        marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
        line=dict(width=3)
    )

    fig_drop_pct.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Percentage Change in Drops",
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

    # Add annotations to each point in the graph to display the percentage change in drops
    for _, row in drops_df.iterrows():
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

    # Display the percentage change in drops plot in Streamlit
    st.plotly_chart(fig_drop_pct, use_container_width=True)

    # DROPS: Count
    # Plot the count of drops
    fig_drop_diff = px.line(
        drops_df,
        x='x_axisLabel',
        y='Drops',
        color=groupbyVar,  # Differentiate lines by groupbyVar
        markers=True,
        title='Number of Drops'
    )

    fig_drop_diff.update_traces(
        marker=dict(size=10, symbol="circle", line=dict(width=1, color='darkslategray')),
        line=dict(width=3)
    )

    fig_drop_diff.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Count of Dropped Students",
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

    # Add annotations for each point in the graph
    for _, row in drops_df.iterrows():
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
    total_dropped_dancers = drops_df['Drops'].sum()
    st.markdown(
        f"<h5 style='text-align: left;'>Total Dropped Students: {total_dropped_dancers:.0f}</h5>",
        unsafe_allow_html=True
    )
# Tab 4: Format Data
with tab4:
    def extract_info_from_filename(filename):
        file_name = os.path.basename(filename)
        parts = os.path.splitext(file_name)[0].split('_')
        if len(parts) >= 7:
            return {
                "Location": parts[0],
                "Reg/NonReg": parts[1],
                "Season": parts[2],
                "Session": parts[3],
                "Year": parts[4],
                "Class": parts[5],
                "Teacher": parts[6],
                "Source": file_name 
            }
        else:
            raise ValueError(f"Filename format invalid: {filename}")

    def clean_last_name(last_name):
        return re.sub(r"\s?\(.*\)", "", last_name)

    def calculate_age(birth_date, year):
        birth_year = birth_date.year
        birth_month = birth_date.month
        birth_day = birth_date.day

        age = year - birth_year
        if (birth_month > datetime.now().month) or (birth_month == datetime.now().month and birth_day > datetime.now().day):
            age -= 1
        return age

    def process_files(file_buffers):
        consolidated_data = []
        for file_buffer in file_buffers:
            try:
                info = extract_info_from_filename(file_buffer.name)

                session = int(info["Session"]) if info["Session"].isdigit() else None
                year = int(info["Year"]) if info["Year"].isdigit() else None

                df = pd.read_excel(file_buffer)
                
                for index, row in df.iterrows():
                    cleaned_last_name = clean_last_name(row['Last Name'])
                    
                    # Attempt to convert birth_date, with error handling
                    try:
                        birth_date = datetime.strptime(row['Birth Date'], '%b %d, %Y')
                    except (ValueError, TypeError) as e:
                        # If the date is invalid or None, set to default value (Jan 1, 2000)
                        birth_date = datetime(2000, 1, 1)
                        print(f"Error parsing date for {row['Last Name']} at index {index}: {e}")
                    
                    formatted_birth_date = birth_date.strftime('%b %d, %Y')
                    
                    # Calculate age based on year, or default to None if no year is provided
                    if year:
                        age = calculate_age(birth_date, year)
                    else:
                        age = None 

                    consolidated_data.append({
                        "DancerID": f"{row['First Name']}_{cleaned_last_name}_{formatted_birth_date}",
                        "FirstName": row['First Name'],
                        "LastName": cleaned_last_name,
                        "Phone": row['Phone'],
                        "Email": row['Email'],
                        "Address": row['Address'],
                        "BirthDate": formatted_birth_date,
                        "Age": age,
                        "City": None,
                        "Location": info["Location"],
                        "Reg/NonReg": info["Reg/NonReg"],
                        "Season": info["Season"],
                        "Session": session,
                        "Year": year,
                        "Class": info["Class"],
                        "Teacher": info["Teacher"],
                        "Source": info["Source"]
                    })
            except Exception as e:
                st.error(f"Error processing file {file_buffer.name}: {e}")

        return pd.DataFrame(consolidated_data)

    def save_to_excel(dataframe):
        output_path = f"Consolidated_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        dataframe.to_excel(output_path, index=False)
        return output_path

    def format_data_tab():
        st.title("Data Format Application")
        st.write("Upload your Excel files to process and format the data.")

        uploaded_files = st.file_uploader("Upload Excel Files", type=['xls', 'xlsx', 'xlsm'], accept_multiple_files=True)

        if st.button("Process Files"):
            if uploaded_files:
                consolidated_df = process_files(uploaded_files)
                if not consolidated_df.empty:
                    st.success("Files processed successfully!")
                    st.write(consolidated_df)

                    output_file = save_to_excel(consolidated_df)
                    with open(output_file, "rb") as file:
                        st.download_button(
                            label="Download Consolidated Data",
                            data=file,
                            file_name=output_file,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.warning("No data was consolidated. Please check your files.")
            else:
                st.warning("Please upload at least one file.")

    # Main entry point
    if __name__ == "__main__":
        format_data_tab()
