import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv('olympics2024.csv')

# Verificar se as colunas 'Gold' e 'Total' existem
if 'Gold' not in df.columns or 'Total' not in df.columns:
    raise KeyError("As colunas 'Gold' e/ou 'Total' não estão presentes no DataFrame.")

# Padronização dos Valores
scaler = StandardScaler()
df[['Gold', 'Total']] = scaler.fit_transform(df[['Gold', 'Total']])

# Treinamento do Modelo GMM
numero_de_clusters = 3
gmm = GaussianMixture(n_components=numero_de_clusters, random_state=0)
labels = gmm.fit_predict(df[['Gold', 'Total']])

# Avaliação do Modelo
davies_bouldin_avg = davies_bouldin_score(df[['Gold', 'Total']], labels)
print(f"Índice de Davies-Bouldin: {davies_bouldin_avg}")

# Adicionar os rótulos dos clusters ao DataFrame
df['Cluster'] = labels

# Exportar estatísticas descritivas para Excel
with pd.ExcelWriter('olympics2024_clusters_stats.xlsx') as writer:
    for cluster_num in range(numero_de_clusters):
        cluster_data = df[df['Cluster'] == cluster_num]
        estatisticas_descritivas = cluster_data.describe()
        estatisticas_descritivas.to_excel(writer, sheet_name=f'Cluster_{cluster_num}_Stats')

# Plotar os clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Gold', y='Total', hue='Cluster', palette='viridis', s=100, alpha=0.6, edgecolor='w')
plt.title(f'Clusters de Total de medalhas por Proporção de Medalhas de Ouro (GMM) - Índice de Davies-Bouldin: {davies_bouldin_avg:.2f}')
plt.xlabel('Medalhas de Ouro (Padronizado)')
plt.ylabel('Total de Medalhas (Padronizado)')
plt.legend()
plt.grid(True)
plt.savefig('clusters_olympics2024_gmm.png')
plt.show()

# Exportação dos Resultados
df.to_csv('olympics2024_clusters.csv', index=False)