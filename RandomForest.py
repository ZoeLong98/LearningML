import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Load the dataset
data = pd.read_csv('datasets/forest+fires/forestfires.csv')

# Select features and target variable
X = data[['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']]
y = data['area']

# Map categorical variables to numerical values
Mon = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
Week = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
X.loc[:, 'month'] = X.loc[:, 'month'].map(Mon)
X.loc[:, 'day'] = X.loc[:, 'day'].map(Week)
y = np.log(y + 1)

# Apply K-Means clustering to classify y into 4 classes
kmeans = KMeans(n_clusters=4, random_state=42)
# data['cluster'] = kmeans.fit_predict(y.values.reshape(-1, 1))
# 创建 K-means 模型
kmeans = KMeans(n_clusters=4)
kmeans.fit(y.values.reshape(-1, 1))

# 获取聚类中心和聚类标签
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# 绘制一维数据点并根据聚类标签着色
plt.figure(figsize=(18, 6))
plt.scatter(y, np.zeros_like(y), c=labels, cmap='plasma', s=6)  # s=点的大小
plt.scatter(centers, np.zeros_like(centers), c='red', s=20, label='Centroids')  # 聚类中心
plt.title('1D K-means Clustering Visualization')
plt.xlabel('Data Points')
plt.axhline(y=0, color='black', linewidth=0.5)  # Draw a single horizontal line at y=0
plt.gca().axes.get_yaxis().set_visible(False)  # Make y-axis invisible
# Make the plot border invisible
for spine in plt.gca().spines.values():
    spine.set_visible(False)
# Move the x-axis ticks to y=0
plt.gca().tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
plt.gca().spines['bottom'].set_position(('data', 0))
plt.legend()
plt.show()


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate and print the accuracy score and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot feature importances
feature_importances = model.feature_importances_
features = X.columns
indices = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# Visualize the decision process
plt.figure(figsize=(100, 50))
plot_tree(model.estimators_[0], feature_names=features, filled=True, rounded=True, class_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
plt.show()
