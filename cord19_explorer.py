# cord19_explorer_fixed.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
import re
import time
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load and preprocess data efficiently
@st.cache_data
def load_data(sample_size=None):
    # Show progress
    progress_bar = st.progress(0, text="Loading data...")
    
    try:
        # For large files, we can read in chunks or sample
        if sample_size:
            # Read only the first n rows for faster loading
            df = pd.read_csv('metadata.csv', nrows=sample_size)
            progress_bar.progress(30, text="Sample data loaded. Processing...")
        else:
            # Try to read the full file with optimized dtypes
            dtype_spec = {
                'title': 'string',
                'abstract': 'string',
                'journal': 'string',
                'authors': 'string',
                'url': 'string'
            }
            df = pd.read_csv('metadata.csv', dtype=dtype_spec)
            progress_bar.progress(30, text="Full data loaded. Processing...")
    except FileNotFoundError:
        # If the file isn't available, use sample data for demonstration
        st.warning("Using sample data as metadata.csv was not found. Please upload the full dataset for complete analysis.")
        # Create sample data with similar structure
        dates = pd.date_range('2019-01-01', '2022-12-31', freq='D')
        journals = ['The Lancet', 'Nature', 'Science', 'JAMA', 'NEJM', 'BMJ', 
                   'PLOS One', 'Clinical Infectious Diseases', None]
        
        sample_size = sample_size if sample_size else 5000
        sample_data = {
            'title': [f'COVID-19 Research Paper {i}' for i in range(sample_size)],
            'abstract': [f'This is an abstract about coronavirus research, pandemic impacts, and medical findings {i}' for i in range(sample_size)],
            'publish_time': [dates[i % len(dates)].strftime('%Y-%m-%d') for i in range(sample_size)],
            'journal': [journals[i % len(journals)] for i in range(sample_size)],
            'authors': [f'Author {i}, Coauthor {i}' for i in range(sample_size)],
            'url': [f'https://example.com/paper/{i}' for i in range(sample_size)]
        }
        df = pd.DataFrame(sample_data)
        progress_bar.progress(30, text="Sample data created. Processing...")
    
    # Data cleaning and preparation
    # Convert publish_time to datetime (only for non-null values)
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    progress_bar.progress(50, text="Dates processed...")
    
    # Extract year from publication date
    df['year'] = df['publish_time'].dt.year
    
    # Calculate abstract word count (only for non-null values)
    df['abstract_word_count'] = df['abstract'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    progress_bar.progress(70, text="Text processed...")
    
    # Fill missing journal values
    df['journal'] = df['journal'].fillna('Unknown Journal')
    
    progress_bar.progress(100, text="Data preparation complete!")
    time.sleep(0.5)  # Let the user see the completion message
    progress_bar.empty()
    
    return df

# Initialize session state for data loading options
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'use_sample' not in st.session_state:
    st.session_state.use_sample = True
if 'sample_size' not in st.session_state:
    st.session_state.sample_size = 10000

# Sidebar for data loading options
st.sidebar.title("Data Loading Options")
st.sidebar.write("Choose how to load the data for better performance")

# Data loading options
use_sample = st.sidebar.checkbox(
    "Use sample data for faster loading", 
    value=st.session_state.use_sample,
    help="Uncheck to load the full dataset (may be slow)"
)

sample_size = None
if use_sample:
    sample_size = st.sidebar.slider(
        "Sample size", 
        min_value=1000, 
        max_value=50000, 
        value=st.session_state.sample_size,
        step=1000,
        help="Number of rows to load from the dataset"
    )

load_button = st.sidebar.button("Load Data")

if load_button:
    st.session_state.use_sample = use_sample
    st.session_state.sample_size = sample_size
    st.session_state.data_loaded = False
    # Clear cache to force reload
    load_data.clear()

# Load data based on options
if not st.session_state.data_loaded:
    with st.spinner("Loading data. This may take a while for large datasets..."):
        df = load_data(sample_size if use_sample else None)
        st.session_state.df = df
        st.session_state.data_loaded = True
else:
    df = st.session_state.df

# Only proceed if data is loaded
if st.session_state.data_loaded:
    # Sidebar for user inputs
    st.sidebar.title("CORD-19 Data Explorer")
    st.sidebar.write("Filter the data to customize your analysis")
    
    # Year range selector
    min_year = int(df['year'].min()) if not pd.isna(df['year'].min()) else 2019
    max_year = int(df['year'].max()) if not pd.isna(df['year'].max()) else 2022
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Journal selector with most common journals first
    journal_counts = df['journal'].value_counts()
    top_journals = journal_counts.head(20).index.tolist()
    selected_journals = st.sidebar.multiselect(
        "Select Journals (top 20 shown)",
        options=top_journals,
        default=top_journals[:5] if len(top_journals) > 5 else top_journals
    )
    
    # Filter data based on selections
    filtered_df = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1]) &
        (df['journal'].isin(selected_journals))
    ]
    
    # Main content
    st.title("📚 CORD-19 Research Dataset Explorer")
    st.write("""
    This interactive dashboard explores the COVID-19 Open Research Dataset (CORD-19), 
    which contains scholarly articles about COVID-19 and the coronavirus family.
    """)
    
    # Display dataset info
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Papers", len(df))
    col2.metric("Filtered Papers", len(filtered_df))
    col3.metric("Time Range", f"{year_range[0]} - {year_range[1]}")
    col4.metric("Journals Selected", len(selected_journals))
    
    # Show basic data info in an expander
    with st.expander("Show Data Information"):
        st.subheader("Data Structure")
        st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Data Type': df.dtypes.values
        })
        st.dataframe(col_info)
        
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Publication Trends", 
        "Journal Analysis", 
        "Word Analysis", 
        "Data Samples"
    ])
    
    with tab1:
        st.header("Publication Trends Over Time")
        
        # Yearly distribution - fixed to show integer years
        yearly_counts = filtered_df['year'].value_counts().sort_index()
        
        # Convert year values to integers for proper display
        yearly_index = [int(year) for year in yearly_counts.index if not pd.isna(year)]
        yearly_values = [yearly_counts[year] for year in yearly_counts.index if not pd.isna(year)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(yearly_index, yearly_values)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Publications')
        ax.set_title('Publications by Year')
        st.pyplot(fig)
        
        # Show monthly trend only if dataset isn't too large
        if len(filtered_df) < 10000:
            # Create a copy and drop rows with missing publish_time
            df_time = filtered_df.dropna(subset=['publish_time']).copy()
            
            # Group by year and month
            df_time['year_month'] = df_time['publish_time'].dt.to_period('M').astype(str)
            monthly_counts = df_time.groupby('year_month').size().reset_index(name='count')
            
            # If we have too many months, sample every nth month for better readability
            if len(monthly_counts) > 24:  # More than 2 years of monthly data
                # Show quarterly instead
                df_time['quarter'] = df_time['publish_time'].dt.to_period('Q').astype(str)
                quarterly_counts = df_time.groupby('quarter').size().reset_index(name='count')
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(quarterly_counts['quarter'], quarterly_counts['count'], marker='o')
                ax.set_xlabel('Time (Quarter)')
                ax.set_ylabel('Number of Publications')
                ax.set_title('COVID-19 Publications Over Time (Quarterly)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info("Showing quarterly data for better readability. Use a smaller date range for monthly view.")
            else:
                # Show monthly data
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(monthly_counts['year_month'], monthly_counts['count'], marker='o')
                ax.set_xlabel('Time (Year-Month)')
                ax.set_ylabel('Number of Publications')
                ax.set_title('COVID-19 Publications Over Time (Monthly)')
                
                # Reduce the number of x-axis labels for better readability
                n = max(1, len(monthly_counts) // 10)  # Show every nth label
                xtick_labels = ax.get_xticklabels()
                for i, label in enumerate(xtick_labels):
                    if i % n != 0:
                        label.set_visible(False)
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("Monthly trend chart disabled for large datasets to improve performance.")
    
    with tab2:
        st.header("Journal Analysis")
        
        # Top journals by publication count
        top_journals = filtered_df['journal'].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_journals.index, top_journals.values)
        ax.set_xlabel('Number of Publications')
        ax.set_title('Top 10 Journals by Publication Count')
        st.pyplot(fig)
        
        # Show journal statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Journal Statistics")
            st.write(f"Total journals: {filtered_df['journal'].nunique()}")
            st.write(f"Average papers per journal: {filtered_df['journal'].value_counts().mean():.2f}")
        
        with col2:
            st.subheader("Top 5 Journals")
            for i, (journal, count) in enumerate(top_journals.head(5).items(), 1):
                st.write(f"{i}. {journal}: {count} papers")
    
    with tab3:
        st.header("Word Analysis")
        
        # Word cloud of titles - only if dataset isn't too large
        if len(filtered_df) < 5000:
            st.subheader("Word Cloud of Paper Titles")
            
            # Combine all titles
            text = ' '.join(filtered_df['title'].dropna().astype(str))
            
            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Common Words in Paper Titles')
            st.pyplot(fig)
        else:
            st.info("Word cloud disabled for large datasets to improve performance.")
        
        # Abstract word count distribution
        st.subheader("Abstract Length Distribution")
        
        # Use a sample for large datasets
        sample_size = min(5000, len(filtered_df))
        sample_data = filtered_df['abstract_word_count'].sample(sample_size) if len(filtered_df) > 5000 else filtered_df['abstract_word_count']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(sample_data, bins=30, edgecolor='black')
        ax.set_xlabel('Word Count')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Abstract Word Counts')
        st.pyplot(fig)
    
    with tab4:
        st.header("Data Samples")
        
        # Show filtered data with pagination
        st.subheader("Filtered Research Papers")
        
        # Add pagination
        page_size = 20
        page_number = st.number_input("Page", min_value=1, max_value=max(1, len(filtered_df) // page_size + 1), value=1)
        
        start_idx = (page_number - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))
        
        st.dataframe(filtered_df.iloc[start_idx:end_idx][['title', 'journal', 'year', 'authors']])
        
        st.write(f"Showing records {start_idx + 1} to {end_idx} of {len(filtered_df)}")
        
        # Option to show raw data
        if st.checkbox("Show Raw Data (Caution: may be slow for large datasets)"):
            st.subheader("Raw Data")
            st.dataframe(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This dashboard provides an exploratory analysis of the CORD-19 dataset, 
    which contains COVID-19 and coronavirus-related research papers.
    """)
    
    st.markdown("### Data Source")
    st.markdown("""
    The data is from the [CORD-19 dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) 
    provided by the Allen Institute for AI.
    """)
else:
    st.info("Please configure data loading options in the sidebar and click 'Load Data' to begin.")