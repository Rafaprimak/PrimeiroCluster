import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

# Coleta de Dados
df = pd.read_csv('olympics2024.csv')

# Tratamento de Dados
# Eliminar dados ausentes
df.dropna(inplace=True)

# Eliminar outliers usando o método IQR
Q1 = df[['Gold', 'Silver', 'Bronze', 'Total']].quantile(0.25)
Q3 = df[['Gold', 'Silver', 'Bronze', 'Total']].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[['Gold', 'Silver', 'Bronze', 'Total']] < (Q1 - 1.5 * IQR)) | (df[['Gold', 'Silver', 'Bronze', 'Total']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Padronização dos Valores
scaler = StandardScaler()
df[['Gold', 'Silver', 'Bronze', 'Total']] = scaler.fit_transform(df[['Gold', 'Silver', 'Bronze', 'Total']])

# Treinamento do Modelo
numero_de_clusters = 3
kmeans = KMeans(n_clusters=numero_de_clusters, random_state=0)
kmeans.fit(df[['Gold', 'Silver', 'Bronze', 'Total']])

# Avaliação do Modelo
labels = kmeans.labels_
silhouette_avg = silhouette_score(df[['Gold', 'Silver', 'Bronze', 'Total']], labels)
print("Silhouette Score: ", silhouette_avg)

# Adicionar os rótulos dos clusters ao DataFrame
df['Cluster'] = labels

# Exportação dos Resultados
with pd.ExcelWriter('resultados_olympics2024.xlsx') as writer:
    df.to_excel(writer, sheet_name='Clusters', index=False)
    
    # Estatísticas Descritivas
    for cluster_num in range(numero_de_clusters):
        cluster_data = df[df['Cluster'] == cluster_num]
        estatisticas_descritivas = cluster_data.describe()
        estatisticas_descritivas.to_excel(writer, sheet_name=f'Cluster_{cluster_num}_Stats')

# Plotar os clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Gold', y='Total', hue='Cluster', palette='viridis', s=100, alpha=0.6, edgecolor='w')
plt.title('Clusters de Países por Medalhas de Ouro e Total de Medalhas')
plt.xlabel('Medalhas de Ouro (Padronizado)')
plt.ylabel('Total de Medalhas (Padronizado)')
plt.legend()
plt.grid(True)
plt.savefig('clusters_olympics2024.png')
plt.show()