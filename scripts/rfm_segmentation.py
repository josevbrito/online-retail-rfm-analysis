# -*- coding: utf-8 -*-
"""
RFM Segmentation Module
Módulo responsável pela segmentação de clientes usando K-Means
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def find_optimal_clusters(rfm_data, max_clusters=10):
    """
    Encontra o número ótimo de clusters usando o método do cotovelo
    
    Args:
        rfm_data (pd.DataFrame): Dados RFM
        max_clusters (int): Número máximo de clusters para testar
        
    Returns:
        list: Lista com valores WCSS para cada número de clusters
    """
    wcss = []
    
    for i in range(1, max_clusters + 1):
        # Escala os dados
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_data)
        
        # Aplica K-Means
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)
    
    return wcss


def plot_elbow_curve(wcss, max_clusters=10, save_path=None):
    """
    Plota a curva do cotovelo para determinar o número ótimo de clusters
    
    Args:
        wcss (list): Lista com valores WCSS
        max_clusters (int): Número máximo de clusters testados
        save_path (str): Caminho para salvar o gráfico (opcional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
    plt.title('Método do Cotovelo para Segmentação RFM')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('WCSS (Soma dos Quadrados Dentro do Cluster)')
    plt.xticks(range(1, max_clusters + 1))
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico do cotovelo salvo em: {save_path}")
    
    plt.show()


def create_rfm_segmentation_models(rfm_data, n_clusters=5):
    """
    Cria e treina os modelos de segmentação RFM (Scaler + KMeans)
    
    Args:
        rfm_data (pd.DataFrame): Dados RFM
        n_clusters (int): Número de clusters
        
    Returns:
        tuple: (scaler_fitted, kmeans_fitted, rfm_with_clusters)
    """
    print(f"Criando modelos de segmentação RFM com {n_clusters} clusters...")
    
    # Treina o scaler
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)
    
    # Treina o K-Means
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(rfm_scaled)
    
    # Adiciona clusters aos dados originais
    rfm_with_clusters = rfm_data.copy()
    rfm_with_clusters['Cluster'] = clusters
    
    # Exibe resumo dos clusters
    print("\nResumo dos Clusters:")
    cluster_summary = rfm_with_clusters.groupby('Cluster').agg(
        Count=('Recency', 'count'),
        AvgRecency=('Recency', 'mean'),
        AvgFrequency=('Frequency', 'mean'),
        AvgMonetary=('Monetary', 'mean')
    ).round(2)
    print(cluster_summary)
    
    return scaler, kmeans, rfm_with_clusters


def predict_rfm_segments(rfm_data, scaler, kmeans_model):
    """
    Prediz segmentos RFM para novos dados
    
    Args:
        rfm_data (pd.DataFrame): Novos dados RFM
        scaler: Scaler treinado
        kmeans_model: Modelo K-Means treinado
        
    Returns:
        np.array: Array com os clusters preditos
    """
    rfm_scaled = scaler.transform(rfm_data)
    clusters = kmeans_model.predict(rfm_scaled)
    return clusters