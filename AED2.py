import pandas as pd
from sklearn.preprocessing import StandardScaler


# Etapa 1

print("========== ETAPA 1: PRÉ-PROCESSAMENTO ==========")
print() 

# Carregar o dataset
df = pd.read_csv("Mall_Customers.csv")

# Codificar a variável categórica Gender (Male = 1, Female = 0)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Selecionar features numéricas relevantes
features = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Normalizar os dados com StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Converter para DataFrame normalizado
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

# Visualizar os dados normalizados 
print(features_scaled_df.head())



# Etapa 2

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print()
print("========== ETAPA 2: SELEÇÃO DO NÚMERO DE CLUSTERS ==========")
print()  

# Faixa de valores de K a testar
K_range = range(2, 11)
inertia = []
silhouette_scores = []

# Calcular inércia (Elbow) e Silhouette Score para cada K
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))

# Plot do Método do Cotovelo
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Número de clusters (K)')
plt.ylabel('Inércia (Soma das Distâncias)')
plt.title('Método do Cotovelo')

# Escalas personalizadas
plt.xticks(range(1, 11)) 
plt.yticks(range(100, 701, 100))  
plt.grid(True)
plt.show()

# Exibir os Silhouette Scores
for k, score in zip(K_range, silhouette_scores):
    print(f"K={k}: Silhouette Score = {score:.4f}")

best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
print(f"\nMelhor K com base no Silhouette Score: K={best_k}")



# Etapa 3

from sklearn.cluster import KMeans
import pandas as pd

print()
print("========== ETAPA 3: MODELAGEM COM K-MENS ==========")
print()

# Definir número de clusters
k = 6

# Treinar o modelo KMeans
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(features_scaled_df)

# Extrair os centroides (em escala normalizada)
centroides = pd.DataFrame(kmeans.cluster_centers_, columns=features_scaled_df.columns)
centroides['Cluster'] = centroides.index

# Exibir os centroides arredondados para melhor leitura

print("Centroides dos clusters (valores normalizados):")
print(centroides.round(2))


#ETAPA 4

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd

print()
print("\n========== ETAPA 4: VISUALIZAÇÃO DOS CLUSTERS ==========")
print()

# Rodar KMeans com K=6
k = 6
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(features_scaled_df)

# Pegar os índices das colunas de Renda e Gasto
col_income = 'Annual Income (k$)'
col_spend = 'Spending Score (1-100)'
col_idx_income = features.columns.get_loc(col_income)
col_idx_spend = features.columns.get_loc(col_spend)

# Pegar centroides normalizados e preparar estrutura para desnormalizar
centroids_scaled = kmeans.cluster_centers_
centroids_dummy = pd.DataFrame(0.0, index=range(k), columns=features.columns)
centroids_dummy.iloc[:, col_idx_income] = centroids_scaled[:, col_idx_income]
centroids_dummy.iloc[:, col_idx_spend] = centroids_scaled[:, col_idx_spend]

# Desnormalizar os centroides apenas para renda e gasto
centroids_original = scaler.inverse_transform(centroids_dummy)
centroids_2d_original = pd.DataFrame(centroids_original, columns=features.columns)

# Plot 2D com escalas ajustadas
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=col_income,
    y=col_spend,
    hue='Cluster',
    palette='viridis',
    data=df,
    legend='full'
)

# Adicionar centroides
plt.scatter(
    x=[0], y=[0],
    s=500,
    c='red',
    marker='X',
    label='Centroids'
)

# Ajustar escalas dos eixos
plt.xticks([0, 20, 40, 60, 80, 100, 120, 140])
plt.yticks([0, 20, 40, 60, 80, 100])
plt.xlim(-10, 140)
plt.ylim(-10, 100)

plt.title('Segmentação de clientes - k-means')
plt.xlabel('Renda anual (k$)')
plt.ylabel('Pontuação de gasto (1-100)')
plt.legend()
plt.grid(True)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Gráfico 3D - Idade x Renda x Gasto
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Pegar os clusters como array para usar no colorbar
clusters = df['Cluster'].to_numpy()

# Plotar os pontos
scatter = ax.scatter(
    df['Annual Income (k$)'],             # X
    df['Spending Score (1-100)'],         # Y
    df['Age'],                            # Z
    c=clusters,
    cmap='viridis',
    s=50
)

# Adicionar barra de cores ao lado
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Cluster')

# Rótulos e título
ax.set_title("Clusters - Renda x Gasto x Idade")
ax.set_xlabel("Renda Anual (k$)")
ax.set_ylabel("Pontuação de Gasto (1-100)")
ax.set_zlabel("Idade")

plt.show()


#ETAPA 5 

print("========== ETAPA 5: ESTATÍSTICAS POR CLUSTER ==========")

# Agrupar por cluster e calcular médias
stats = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(1)

# Renomear colunas para o padrão do relatório
stats.columns = ['Idade Média', 'Renda Média (k$)', 'Gasto Médio']

# Resetar índice para mostrar Cluster como coluna
stats = stats.reset_index()

# Exibir como tabela formatada
print("Cluster   Idade Média   Renda Média (k$)   Gasto Médio")
for _, row in stats.iterrows():
    print(f"{int(row['Cluster']):<9} {row['Idade Média']:<13} {row['Renda Média (k$)']:<19} {row['Gasto Médio']}")
    
print("\nTabela 1: Métricas médias por cluster")


