
# Player Clustering Project

This project aims to cluster football players based on their performance metrics using machine learning techniques. 
The main objective is to categorize players into clusters to identify patterns and similarities between them.

## Project Structure

```
player-clustering/
├── data/                    # Contains CSV files with player performance metrics
│   ├── Final_RWB_Scores_and_Clusters.csv
│   └── League_Normalised_RB_Scoring_ALL.csv
├── scripts/                 # Python scripts for clustering
│   └── cluster.py
├── outputs/                 # HTML output files for visualizing clusters
│   ├── cluster0.html
│   └── all_clusters.html
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Getting Started

### Prerequisites

To run this project, ensure you have Python installed. You can install the required packages using:

```
pip install -r requirements.txt
```

### Running the Clustering Script

To execute the clustering algorithm and generate output files, run the following command:

```
python scripts/cluster.py
```

## Output

The script generates HTML files containing visualizations of the clusters. These can be found in the `outputs/` directory.

## Results

- `cluster0.html`: Visualization of the first cluster.
- `all_clusters.html`: Comprehensive visualization of all clusters.
