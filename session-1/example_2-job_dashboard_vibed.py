# AI Job Dashboard with Streamlit (Vibe Coded)
## ABB #4 - Session 1

import streamlit as st
import pandas as pd
import plotly.express as px

# Set page title and layout
st.set_page_config(
    page_title="Top Paying Jobs Dashboard",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/ai_job_data.csv")
    
    # Clean salary data (some rows have N/A)
    df['salary_min'] = pd.to_numeric(df['salary_min'], errors='coerce')
    df['salary_max'] = pd.to_numeric(df['salary_max'], errors='coerce')
    
    # Calculate average salary for sorting and visualization
    df['avg_salary'] = (df['salary_min'] + df['salary_max']) / 2
    
    # Fill NaN values with 0 for filtering purposes
    df.fillna({'avg_salary': 0, 'salary_min': 0, 'salary_max': 0}, inplace=True)
    
    return df

# Load the data
df = load_data()

# Dashboard title
st.title("Top Paying Jobs Dashboard")

# Sidebar for filtering
st.sidebar.header("Filters")

# Company filter
companies = sorted(df['company_name'].unique())
selected_companies = st.sidebar.multiselect(
    "Select Companies:",
    options=companies,
    default=None
)

# Filter data based on selections
if selected_companies:
    filtered_df = df[df['company_name'].isin(selected_companies)]
else:
    filtered_df = df

# Number of top jobs to show
top_n = st.sidebar.slider("Number of top jobs to display:", 5, 50, 10)

# Sort by average salary and take top N jobs
top_jobs_df = filtered_df.sort_values(by='avg_salary', ascending=False).head(top_n)

# Main content
st.header(f"Top {top_n} Highest Paying Jobs")

if top_jobs_df.empty:
    st.warning("No data available with the selected filters.")
else:
    # Create bar chart with Plotly
    fig = px.bar(
        top_jobs_df,
        x='avg_salary',
        y='job_title',
        orientation='h',
        color='company_name',
        hover_data=['salary_min', 'salary_max', 'company_name'],
        labels={
            'avg_salary': 'Average Salary ($)',
            'job_title': 'Job Title',
            'company_name': 'Company',
            'salary_min': 'Minimum Salary',
            'salary_max': 'Maximum Salary'
        },
        title=f"Top {top_n} Highest Paying Jobs",
        height=600
    )
    
    # Customize the layout
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_tickprefix='$',
        xaxis_tickformat=',',
        legend_title_text='Company'
    )
    
    # Display the bar chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Display data table
    st.subheader("Details")
    display_df = top_jobs_df[['company_name', 'job_title', 'salary_min', 'salary_max', 'avg_salary']].reset_index(drop=True)
    display_df = display_df.rename(columns={
        'company_name': 'Company',
        'job_title': 'Job Title',
        'salary_min': 'Min Salary ($)',
        'salary_max': 'Max Salary ($)',
        'avg_salary': 'Avg Salary ($)'
    })
    
    # Format salary columns
    display_df['Min Salary ($)'] = display_df['Min Salary ($)'].apply(lambda x: f"${x:,.0f}" if x > 0 else "N/A")
    display_df['Max Salary ($)'] = display_df['Max Salary ($)'].apply(lambda x: f"${x:,.0f}" if x > 0 else "N/A")
    display_df['Avg Salary ($)'] = display_df['Avg Salary ($)'].apply(lambda x: f"${x:,.0f}" if x > 0 else "N/A")
    
    st.dataframe(display_df, use_container_width=True) 
