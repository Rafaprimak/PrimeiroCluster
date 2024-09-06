import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

# Coleta de Dados
df = pd.read_csv('olympics2024.csv')

# Converter colunas para valores numéricos, forçando erros para NaN
df['Gold'] = pd.to_numeric(df['Gold'], errors='coerce')
df['Total'] = pd.to_numeric(df['Total'], errors='coerce')

# Tratamento de Dados
# Eliminar dados ausentes
df.dropna(subset=['Gold', 'Total'], inplace=True)

# Eliminar outliers usando o método IQR
Q1 = df[['Gold', 'Total']].quantile(0.25)
Q3 = df[['Gold', 'Total']].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[['Gold', 'Total']] < (Q1 - 1.5 * IQR)) | (df[['Gold', 'Total']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Calcular a proporção de medalhas de ouro pelo total de medalhas
df['Gold_to_Total'] = df['Gold'] / df['Total']

# Padronização dos Valores
scaler = StandardScaler()
df[['Gold_to_Total']] = scaler.fit_transform(df[['Gold_to_Total']])

# Treinamento do Modelo K-means
numero_de_clusters = 3
kmeans = KMeans(n_clusters=numero_de_clusters, random_state=0)
labels = kmeans.fit_predict(df[['Gold_to_Total']])

# Avaliação do Modelo
silhouette_avg = silhouette_score(df[['Gold_to_Total']], labels)
print("Silhouette Score: ", silhouette_avg)

# Adicionar os rótulos dos clusters ao DataFrame
df['Cluster'] = labels

# Exportação dos Resultados
with pd.ExcelWriter('resultados_olympics2024_kmeans.xlsx') as writer:
    df.to_excel(writer, sheet_name='Clusters', index=False)
    
    # Estatísticas Descritivas
    for cluster_num in range(numero_de_clusters):
        cluster_data = df[df['Cluster'] == cluster_num]
        estatisticas_descritivas = cluster_data.describe()
        estatisticas_descritivas.to_excel(writer, sheet_name=f'Cluster_{cluster_num}_Stats')

# Plotar os clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Gold', y='Total', hue='Cluster', palette='viridis', s=100, alpha=0.6, edgecolor='w')
plt.title('Clusters de Países por Proporção de Medalhas de Ouro (K-means)')
plt.xlabel('Medalhas de Ouro (Padronizado)')
plt.ylabel('Total de Medalhas (Padronizado)')
plt.legend()
plt.grid(True)
plt.savefig('clusters_olympics2024_kmeans.png')
plt.show()