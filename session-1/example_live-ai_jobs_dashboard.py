import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Set the style for the plots
plt.style.use('ggplot')
sns.set_palette("viridis")

# Load the data
df = pd.read_csv('job_listings.csv')

# Data cleaning and preprocessing
# Convert salary to USD if not already in USD
currency_to_usd = {
    'USD': 1,
    'EUR': 1.1,  # Approximate conversion rate
    'GBP': 1.3,  # Approximate conversion rate
    'CHF': 1.1   # Approximate conversion rate
}

# Create columns for salary in USD
df['salary_min_usd'] = df.apply(lambda row: row['salary_min'] * currency_to_usd.get(row['salary_currency'], 1) 
                              if pd.notnull(row['salary_min']) else np.nan, axis=1)
df['salary_max_usd'] = df.apply(lambda row: row['salary_max'] * currency_to_usd.get(row['salary_currency'], 1) 
                              if pd.notnull(row['salary_max']) else np.nan, axis=1)

# Calculate average salary
df['avg_salary_usd'] = (df['salary_min_usd'] + df['salary_max_usd']) / 2

# Filter to only include USD jobs
df_usd = df[df['salary_currency'] == 'USD']

# Create a dashboard with multiple visualizations
plt.figure(figsize=(18, 10))

# 1. Top 10 highest paying jobs by maximum salary
plt.subplot(2, 2, 1)
top_jobs = df_usd.sort_values('salary_max_usd', ascending=False).head(10)
sns.barplot(x='salary_max_usd', y='title', data=top_jobs)
plt.title('Top 10 Highest Paying AI Jobs in USD (Max Salary)')
plt.xlabel('Maximum Salary (USD)')
plt.ylabel('')
plt.tight_layout()

# 2. Distribution of salaries by job title type
plt.subplot(2, 2, 2)
# Extract job role from title (e.g., 'Data Scientist', 'ML Engineer')
df_usd['job_role'] = df_usd['title'].str.extract(r'(Data Scientist|ML Engineer|Machine Learning|AI|Engineer|Researcher)', flags=re.IGNORECASE)
salary_by_role = df_usd.groupby('job_role')['avg_salary_usd'].mean().sort_values(ascending=False)
sns.barplot(x=salary_by_role.index, y=salary_by_role.values)
plt.title('Average Salary by Job Role (USD)')
plt.xlabel('Role')
plt.ylabel('Average Salary (USD)')
plt.xticks(rotation=45)
plt.tight_layout()

# 3. Salary range for top 5 companies
plt.subplot(2, 2, 3)
top_companies = df_usd.groupby('company')['avg_salary_usd'].mean().sort_values(ascending=False).head(5)
top_companies_df = df_usd[df_usd['company'].isin(top_companies.index)]
sns.boxplot(x='company', y='avg_salary_usd', data=top_companies_df)
plt.title('Salary Ranges for Top 5 Companies (USD Jobs)')
plt.xlabel('Company')
plt.ylabel('Average Salary (USD)')
plt.xticks(rotation=45)
plt.tight_layout()

# 4. AI skills word cloud from job titles
plt.subplot(2, 2, 4)
# Create a horizontal bar chart of words in job titles
title_words = ' '.join(df_usd['title']).lower()
words = pd.Series(title_words.split()).value_counts().head(10)
sns.barplot(x=words.values, y=words.index)
plt.title('Most Common Words in AI Job Titles (USD Jobs)')
plt.xlabel('Count')
plt.ylabel('')
plt.tight_layout()

# Add overall title
plt.suptitle('AI Jobs Dashboard - USD Salary Analysis', fontsize=16, y=1.02)
plt.tight_layout()

# Save the dashboard
plt.savefig('ai_jobs_dashboard_usd.png', dpi=300, bbox_inches='tight')

# Create an interactive dashboard using matplotlib widgets
from matplotlib.widgets import Button, Slider

# Function to create an interactive dashboard
def create_interactive_dashboard():
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle('Interactive AI Jobs Dashboard - USD Salary Analysis', fontsize=16)
    
    # Initial number of top jobs to display
    n_jobs = 10
    
    def update(val=None):
        # Clear all subplots
        for ax in axs.flatten():
            ax.clear()
        
        # 1. Top N highest paying jobs by maximum salary
        top_n_jobs = df_usd.sort_values('salary_max_usd', ascending=False).head(int(n_jobs_slider.val))
        sns.barplot(x='salary_max_usd', y='title', data=top_n_jobs, ax=axs[0, 0])
        axs[0, 0].set_title(f'Top {int(n_jobs_slider.val)} Highest Paying AI Jobs in USD (Max Salary)')
        axs[0, 0].set_xlabel('Maximum Salary (USD)')
        axs[0, 0].set_ylabel('')
        
        # 2. Distribution of salaries by job role
        df_usd['job_role'] = df_usd['title'].str.extract(r'(Data Scientist|ML Engineer|Machine Learning|AI|Engineer|Researcher)', flags=re.IGNORECASE)
        salary_by_role = df_usd.groupby('job_role')['avg_salary_usd'].mean().sort_values(ascending=False)
        sns.barplot(x=salary_by_role.index, y=salary_by_role.values, ax=axs[0, 1])
        axs[0, 1].set_title('Average Salary by Job Role (USD)')
        axs[0, 1].set_xlabel('Role')
        axs[0, 1].set_ylabel('Average Salary (USD)')
        axs[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Salary range for top companies
        top_companies = df_usd.groupby('company')['avg_salary_usd'].mean().sort_values(ascending=False).head(5)
        top_companies_df = df_usd[df_usd['company'].isin(top_companies.index)]
        sns.boxplot(x='company', y='avg_salary_usd', data=top_companies_df, ax=axs[1, 0])
        axs[1, 0].set_title('Salary Ranges for Top 5 Companies (USD Jobs)')
        axs[1, 0].set_xlabel('Company')
        axs[1, 0].set_ylabel('Average Salary (USD)')
        axs[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Salary distribution
        sns.histplot(df_usd['avg_salary_usd'].dropna(), bins=20, kde=True, ax=axs[1, 1])
        axs[1, 1].set_title('Salary Distribution (USD Jobs)')
        axs[1, 1].set_xlabel('Average Salary (USD)')
        axs[1, 1].set_ylabel('Count')
        
        fig.tight_layout()
        plt.draw()
    
    # Add a slider for selecting the number of top jobs
    ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03])
    n_jobs_slider = Slider(ax_slider, 'Top N Jobs', 5, 20, valinit=n_jobs, valstep=1)
    n_jobs_slider.on_changed(update)
    
    # Initialize the plot
    update()
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()

# Run the static dashboard
print("Generating static dashboard with USD jobs only...")
print("Static dashboard saved as 'ai_jobs_dashboard_usd.png'")

# Run the interactive dashboard when script is executed directly
if __name__ == '__main__':
    print("\nStarting interactive dashboard with USD jobs only...")
    create_interactive_dashboard() 