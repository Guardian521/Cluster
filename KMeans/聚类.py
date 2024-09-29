import csv
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

data = []
with open('features.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        file_name = row[0]
        row = [0 if value == '' else float(value) for value in row[1:]]
        data.append([file_name] + row)

df = pd.DataFrame(data)

file_names = df.iloc[:, 0]
features = df.iloc[:, 1:]

imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# 标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

# 使用PCA降维
pca = PCA(n_components=20)
features_pca = pca.fit_transform(features_scaled)

# 使用KMeans聚类
kmeans = KMeans(n_clusters=7, random_state=42)
df['clustering'] = kmeans.fit_predict(features_pca)

#评价指标
#calinski_harabasz指标
calinski_harabasz = calinski_harabasz_score(features_pca, df['clustering'])
print("Calinski-Harabasz Index:", calinski_harabasz)
#davies_bouldin指标
davies_bouldin = davies_bouldin_score(features_pca, df['clustering'])
print("Davies-Bouldin Index:", davies_bouldin)
#轮廓系数
silhouette_avg = silhouette_score(features_pca, df['clustering'])
print("Silhouette Score:", silhouette_avg)

result_df = pd.concat([file_names, df['clustering'], pd.DataFrame(features_pca)], axis=1)

# 保存
output_csv_path = '/上的各种课/opencv/features_with_clusters.csv'
result_df.to_csv(output_csv_path, index=False)
print("结果已保存到", output_csv_path)
