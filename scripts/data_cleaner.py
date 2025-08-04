    # -*- coding: utf-8 -*-
"""
Data Cleaner Module
Módulo responsável pela limpeza e pré-processamento dos dados de transações
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador para limpar e filtrar o DataFrame de transações.
    Remove cancelamentos, CustomerID nulos, e transações inválidas (qtd/preço <= 0).
    Cria a coluna 'TotalPrice'.
    """
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df_cleaned = X.copy()

        # Removendo transações canceladas
        df_cleaned = df_cleaned[~df_cleaned['InvoiceNo'].astype(str).str.contains('C', na=False)].copy()

        # Removendo registros com CustomerID ausente
        df_cleaned.dropna(subset=['CustomerID'], inplace=True)
        df_cleaned['CustomerID'] = df_cleaned['CustomerID'].astype(int)

        # Removendo transações com quantidade ou preço <= 0
        df_cleaned = df_cleaned[(df_cleaned['Quantity'] > 0) & (df_cleaned['UnitPrice'] > 0)].copy()

        # Convertendo InvoiceDate para datetime
        df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])

        # Criando TotalPrice
        df_cleaned['TotalPrice'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']

        return df_cleaned


def load_online_retail_data(file_path="data/Online Retail.xlsx"):
    """
    Carrega o dataset Online Retail
    
    Args:
        file_path (str): Caminho para o arquivo Excel
        
    Returns:
        pd.DataFrame: Dataset carregado ou None se houver erro
    """
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"Dataset '{file_path}' carregado com sucesso.")
        print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{file_path}' não encontrado.")
        print("Criando dados simulados para demonstração...")
        return create_simulated_data()
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return None


def create_simulated_data():
    """
    Cria dados simulados para demonstração caso o arquivo real não seja encontrado
    
    Returns:
        pd.DataFrame: Dataset simulado
    """
    np.random.seed(42)
    n_transactions = 50000
    
    retail_raw = pd.DataFrame({
        'InvoiceNo': [f"53{1000 + i}" for i in range(n_transactions)],
        'StockCode': [f"2{2000 + np.random.randint(0, 5000)}" for _ in range(n_transactions)],
        'Description': [f"Product_{np.random.randint(1, 1000)}" for _ in range(n_transactions)],
        'Quantity': np.random.randint(1, 100, n_transactions),
        'InvoiceDate': pd.date_range('2010-12-01', '2011-12-09', periods=n_transactions),
        'UnitPrice': np.random.uniform(0.5, 50.0, n_transactions),
        'CustomerID': np.random.randint(12000, 20000, n_transactions),
        'Country': np.random.choice(['United Kingdom', 'Germany', 'France', 'Spain', 'Netherlands'],
                                   n_transactions, p=[0.7, 0.1, 0.08, 0.07, 0.05])
    })
    
    # Adiciona algumas transações canceladas
    cancel_mask = np.random.choice([True, False], n_transactions, p=[0.05, 0.95])
    retail_raw.loc[cancel_mask, 'InvoiceNo'] = 'C' + retail_raw.loc[cancel_mask, 'InvoiceNo']
    retail_raw.loc[cancel_mask, 'Quantity'] = -retail_raw.loc[cancel_mask, 'Quantity']
    
    print("Dados simulados criados com sucesso!")
    return retail_raw