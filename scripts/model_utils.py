# -*- coding: utf-8 -*-
"""
Model Utilities Module
Utilitários para salvar, carregar e usar os modelos treinados
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime


def save_models(scaler, kmeans_model, models_dir="models"):
    """
    Salva os modelos RFM treinados
    
    Args:
        scaler: StandardScaler treinado
        kmeans_model: Modelo K-Means treinado
        models_dir (str): Diretório para salvar os modelos
    """
    # Cria o diretório se não existir
    os.makedirs(models_dir, exist_ok=True)
    
    # Salva os modelos
    scaler_path = os.path.join(models_dir, 'rfm_scaler.pkl')
    kmeans_path = os.path.join(models_dir, 'rfm_kmeans_model.pkl')
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(kmeans_model, kmeans_path)
    
    print(f"Modelos salvos:")
    print(f"- Scaler: {scaler_path}")
    print(f"- K-Means: {kmeans_path}")


def load_models(models_dir="models"):
    """
    Carrega os modelos RFM salvos
    
    Args:
        models_dir (str): Diretório dos modelos
        
    Returns:
        tuple: (scaler, kmeans_model) ou (None, None) se houver erro
    """
    try:
        scaler_path = os.path.join(models_dir, 'rfm_scaler.pkl')
        kmeans_path = os.path.join(models_dir, 'rfm_kmeans_model.pkl')
        
        scaler = joblib.load(scaler_path)
        kmeans_model = joblib.load(kmeans_path)
        
        print("Modelos carregados com sucesso!")
        return scaler, kmeans_model
        
    except FileNotFoundError as e:
        print(f"Erro: Arquivo de modelo não encontrado - {e}")
        return None, None
    except Exception as e:
        print(f"Erro ao carregar modelos: {e}")
        return None, None


def predict_customer_segment_from_transactions(transactions_df, scaler, kmeans_model, 
                                               snapshot_date=None):
    """
    Prediz o segmento RFM de um cliente a partir de suas transações
    
    Args:
        transactions_df (pd.DataFrame): DataFrame com transações do cliente
        scaler: StandardScaler treinado
        kmeans_model: Modelo K-Means treinado
        snapshot_date (datetime): Data de referência (opcional)
        
    Returns:
        dict: Dicionário com CustomerID, métricas RFM e cluster predito
    """
    from data_cleaner import DataCleanerTransformer
    from rfm_calculator import RFMCalculatorTransformer
    
    try:
        # Limpa os dados
        cleaner = DataCleanerTransformer()
        clean_data = cleaner.transform(transactions_df)
        
        if clean_data.empty:
            print("Nenhum dado válido após limpeza")
            return None
        
        # Calcula RFM
        rfm_calculator = RFMCalculatorTransformer(snapshot_date=snapshot_date)
        rfm_data = rfm_calculator.fit_transform(clean_data)
        
        if rfm_data.empty:
            print("Nenhum dado RFM calculado")
            return None
        
        # Prediz cluster
        rfm_scaled = scaler.transform(rfm_data)
        cluster = kmeans_model.predict(rfm_scaled)[0]
        
        # Retorna resultado
        customer_id = rfm_data.index[0]
        result = {
            'CustomerID': customer_id,
            'Recency': rfm_data.iloc[0]['Recency'],
            'Frequency': rfm_data.iloc[0]['Frequency'],
            'Monetary': rfm_data.iloc[0]['Monetary'],
            'Cluster': cluster
        }
        
        return result
        
    except Exception as e:
        print(f"Erro na predição: {e}")
        return None


def create_sample_transaction():
    """
    Cria uma transação de exemplo para teste
    
    Returns:
        pd.DataFrame: DataFrame com transação de exemplo
    """
    sample_data = pd.DataFrame({
        'InvoiceNo': ['555001', '555002', '555003'],
        'StockCode': ['22300', '22400', '22500'],
        'Description': ['Item A', 'Item B', 'Item C'],
        'Quantity': [5, 2, 10],
        'InvoiceDate': [
            datetime(2011, 12, 1, 10, 0), 
            datetime(2011, 12, 5, 14, 30), 
            datetime(2011, 12, 8, 9, 0)
        ],
        'UnitPrice': [10.5, 20.0, 5.75],
        'CustomerID': [99999, 99999, 99999],
        'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom']
    })
    
    return sample_data


def interpret_cluster(cluster_id, cluster_summary=None):
    """
    Interpreta o significado de um cluster
    
    Args:
        cluster_id (int): ID do cluster
        cluster_summary (pd.DataFrame): Resumo dos clusters (opcional)
        
    Returns:
        str: Interpretação do cluster
    """
    interpretations = {
        0: "Clientes de Alto Valor - Compraram recentemente, com alta frequência e alto valor monetário",
        1: "Clientes Leais - Frequência alta, mas podem não ter comprado muito recentemente",
        2: "Clientes Potenciais - Compraram recentemente, mas baixa frequência",
        3: "Clientes em Risco - Não compraram recentemente, frequência moderada",
        4: "Clientes Perdidos - Baixa recência, frequência e valor monetário"
    }
    
    return interpretations.get(cluster_id, f"Cluster {cluster_id}")