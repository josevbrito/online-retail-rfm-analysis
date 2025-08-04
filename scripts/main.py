# -*- coding: utf-8 -*-
"""
Main Script - Treinamento de Modelos RFM
Script principal para treinar e salvar os modelos de segmentação RFM
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Importa os módulos customizados
from data_cleaner import load_online_retail_data, DataCleanerTransformer
from rfm_calculator import calculate_rfm_metrics
from rfm_segmentation import find_optimal_clusters, plot_elbow_curve, create_rfm_segmentation_models
from model_utils import save_models, create_sample_transaction, predict_customer_segment_from_transactions


def main():
    """
    Função principal para executar o pipeline completo de treinamento
    """
    print("="*60)
    print("PIPELINE DE TREINAMENTO - MODELOS RFM")
    print("="*60)
    
    # 1. Carregando os dados
    print("\n1. Carregando dados...")
    data_path = "data/Online Retail.xlsx"
    retail_raw = load_online_retail_data(data_path)
    
    if retail_raw is None:
        print("Erro: Não foi possível carregar os dados.")
        return
    
    # 2. Limpando os dados
    print("\n2. Limpando dados...")
    cleaner = DataCleanerTransformer()
    retail_clean = cleaner.fit_transform(retail_raw)
    
    print(f"Dados originais: {retail_raw.shape}")
    print(f"Dados limpos: {retail_clean.shape}")
    
    # 3. Calculando as métricas RFM
    print("\n3. Calculando métricas RFM...")
    rfm_data = calculate_rfm_metrics(retail_clean)
    
    # 4. Determinando o número ótimo de clusters
    print("\n4. Determinando número ótimo de clusters...")
    wcss = find_optimal_clusters(rfm_data, max_clusters=10)
    
    # Plotando a curva do cotovelo
    try:
        plot_elbow_curve(wcss, max_clusters=10, save_path="images/elbow_curve.png")
    except:
        print("Aviso: Não foi possível plotar a curva do cotovelo (matplotlib pode não estar configurado)")
    
    # 5. Criando os modelos de segmentação
    print("\n5. Criando modelos de segmentação...")
    n_clusters = 5  # Pela imagem, pode ser: 3, 4 ou 5
    scaler, kmeans_model, rfm_with_clusters = create_rfm_segmentation_models(rfm_data, n_clusters)
    
    # 6. Salvando os modelos
    print("\n6. Salvando modelos...")
    save_models(scaler, kmeans_model, models_dir="models")
    
    # 7. Teste de inferência
    print("\n7. Testando inferência com dados de exemplo...")
    test_inference(scaler, kmeans_model)
    
    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*60)
    print("\nArquivos gerados:")
    print("- models/rfm_scaler.pkl")
    print("- models/rfm_kmeans_model.pkl")
    print("- images/elbow_curve.png")


def test_inference(scaler, kmeans_model):
    """
    Testa a inferência com dados de exemplo
    
    Args:
        scaler: StandardScaler treinado
        kmeans_model: Modelo K-Means treinado
    """
    # Cria transação de exemplo
    sample_transaction = create_sample_transaction()
    
    print("Dados de exemplo para teste:")
    print(sample_transaction)
    
    # Prediz segmento
    result = predict_customer_segment_from_transactions(
        sample_transaction, scaler, kmeans_model
    )
    
    if result:
        print(f"\nResultado da predição:")
        print(f"- Customer ID: {result['CustomerID']}")
        print(f"- Recency: {result['Recency']} dias")
        print(f"- Frequency: {result['Frequency']} pedidos")
        print(f"- Monetary: £{result['Monetary']:,.2f}")
        print(f"- Cluster: {result['Cluster']}")
        
        # Interpretação do cluster
        from model_utils import interpret_cluster
        interpretation = interpret_cluster(result['Cluster'])
        print(f"- Interpretação: {interpretation}")
    else:
        print("Erro na predição de exemplo")


if __name__ == "__main__":
    # Cria diretório de modelos se não existir
    os.makedirs("models", exist_ok=True)
    
    # Executa pipeline principal
    main()  