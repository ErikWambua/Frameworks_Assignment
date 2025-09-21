# Frameworks_Assignment

# 📚 CORD-19 Data Explorer

A Streamlit-based web application for exploring and visualizing the COVID-19 Open Research Dataset (CORD-19), which contains scholarly articles about COVID-19 and the coronavirus family.

## ✨ Features

- 🔍 **Interactive Data Exploration**: Filter data by year range and journal selection
- 📈 **Publication Trends**: Visualize COVID-19 publications over time (yearly and monthly/quarterly)
- 📑 **Journal Analysis**: Identify top journals publishing COVID-19 research
- ☁️ **Word Analysis**: Generate word clouds from paper titles and analyze abstract lengths
- ⚡ **Data Sampling**: Option to work with sample data for faster performance with large datasets
- 📱 **Responsive Design**: Adapts to different screen sizes with an organized tab interface

## 📂 Dataset

The application uses the metadata.csv file from the [CORD-19 dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) provided by the Allen Institute for AI.

The dataset includes:
- 📝 Paper titles and abstracts
- 📅 Publication dates
- 👨‍🔬 Authors and journals
- 🌍 Source information

## ⚙️ Installation

1. Clone or download this repository
2. Install the required packages:
   ```bash
   pip install pandas matplotlib seaborn streamlit wordcloud
   ```
3. Download the metadata.csv file from the CORD-19 dataset and place it in the same directory as the script

## ▶️ Usage

Run the Streamlit application:

```bash
streamlit run cord19_explorer.py
```

📌 The application will open in your default web browser. Use the sidebar options to:

- ✅ Choose between sample data or full data loading
- ✅ Select an appropriate sample size if using sample data
- ✅ Filter by year range and specific journals
- ✅ Explore different analyses through the tab interface

## 🗂️ Project Structure

```
cord19-explorer/
│
├── cord19_explorer.py      # Main Streamlit application
├── metadata.csv            # CORD-19 metadata file (not included, must be downloaded)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 📦 Dependencies

- 🐍 Python 3.7+
- 📊 pandas
- 📊 matplotlib
- 📊 seaborn
- 🌐 streamlit
- ☁️ wordcloud

## 🚀 Performance Notes

For large datasets, the application includes several optimizations:

- ⚡ Data sampling options to improve loading times
- 🎛️ Selective visualization (some charts are disabled for very large datasets)
- 🔄 Caching mechanisms to avoid redundant processing
- 🗂️ Pagination for data tables

## 📜 License

📖 This project is provided for educational purposes as part of an assignment. The CORD-19 dataset is available under the COVID-19 Open Research Dataset (CORD-19) License.

## 🙏 Acknowledgments

- 🧠 Allen Institute for AI for providing the CORD-19 dataset
- 💻 Streamlit for the excellent web application framework
- 🐍 The Python data science community for the powerful libraries used in this project
