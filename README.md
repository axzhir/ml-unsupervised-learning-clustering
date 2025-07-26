# 🧠 Machine Learning Project: Unsupervised Learning Clustering with World Happiness Report

In this project, you will tackle an **unsupervised learning** problem by exploring clustering techniques using the **World Happiness Report (WHR)** dataset. This dataset summarizes economic and social indicators for countries worldwide, linked to happiness scores reported by people living in those countries.

For more info about the data, check out the [WHR 2018 website](http://worldhappiness.report/ed/2018).

---

### 📌 Project Tasks

* Build your DataFrame and define your ML problem
* Prepare the data by cleaning and feature engineering
* Perform **KMeans clustering** and analyze the clusters
* Perform **Hierarchical clustering** and analyze the results

---

### 🛠️ Example Code Snippet

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
whr_df = pd.read_csv('world_happiness_report.csv')

# Basic data cleaning and feature selection
features = ['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
X = whr_df[features].dropna()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to DataFrame
whr_df['Cluster'] = pd.Series(clusters, index=X.index)

# Plot cluster centers (example)
plt.bar(range(len(kmeans.cluster_centers_[0])), kmeans.cluster_centers_[0])
plt.title('Cluster 0 Centers')
plt.show()
```
