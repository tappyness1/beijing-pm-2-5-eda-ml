# Beijing Air Quality Forecasting Project

## Motivation

Air pollution is a major public health concern in urban areas, especially in cities like Beijing. This project aims to analyze historical air quality data and build predictive models for PM2.5 concentrations, helping inform policy and public awareness.

## Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **matplotlib** & **seaborn**: Data visualization
- **scikit-learn**: Machine learning and model evaluation
- **statsmodels**: Time series analysis

## Repository Structure

- `README.md`: Project overview and instructions
- `blog.md`: Narrative summary of the analysis and findings
- `conda.yml`: Environment specification for reproducibility
- `eda.ipynb`: Exploratory Data Analysis notebook
- `forecasting.ipynb`: Time series forecasting and model evaluation notebook
- `images/`: Visualizations and figures used in the analysis
    - `image.png`, `image-1.png`, ...: Plots and charts referenced in the blog and notebooks
- `src/`: Source code for data processing and modeling
    - `datapipeline.py`: Data cleaning, feature engineering, and pipeline logic
    - `ml_model.py`: Model definition and training routines

## Summary of Results

- **EDA revealed** strong seasonality and meteorological influences on PM2.5 levels, with colder months and high pressure associated with higher pollution.
- **Forecasting models** (Random Forest) achieved reasonable accuracy on training data but higher error on test data, indicating challenges in generalizing to unseen periods.
- **Feature engineering** (lagged and rolling features) improved model performance but further enhancements are needed for extreme events.

## Acknowledgements

- Data sourced from the UCI Machine Learning Repository: [Beijing PM2.5 Data](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)
- Libraries and tools from the open-source Python ecosystem
- Visual inspiration from Unsplash and other free image resources

---

For questions or contributions, please open an issue or submit a pull request.
