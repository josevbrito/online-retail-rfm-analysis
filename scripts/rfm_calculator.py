# -*- coding: utf-8 -*-
"""
RFM Calculator Module
Módulo responsável pelo cálculo das métricas RFM (Recency, Frequency, Monetary)
"""

import pandas as pd
from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin


class RFMCalculatorTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador para calcular as métricas RFM (Recency, Frequency, Monetary).
    Assume que o DataFrame de entrada já está limpo e possui as colunas:
    'CustomerID', 'InvoiceNo', 'InvoiceDate', 'TotalPrice'.
    """
    
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date
        self.snapshot_date_ = None

    def fit(self, X, y=None):
        if self.snapshot_date is None:
            self.snapshot_date_ = X['InvoiceDate'].max() + timedelta(days=1)
        else:
            self.snapshot_date_ = pd.to_datetime(self.snapshot_date)
        return self

    def transform(self, X, y=None):
        # Calculando o RFM
        rfm_transformed = X.groupby('CustomerID').agg(
            Recency=('InvoiceDate', lambda date: (self.snapshot_date_ - date.max()).days),
            Frequency=('InvoiceNo', 'nunique'),
            Monetary=('TotalPrice', 'sum')
        ).reset_index()
        
        rfm_transformed.set_index('CustomerID', inplace=True)
        
        # Retorna apenas as colunas RFM
        return rfm_transformed[['Recency', 'Frequency', 'Monetary']]


def calculate_rfm_metrics(df_clean, snapshot_date=None):
    """
    Função utilitária para calcular métricas RFM
    
    Args:
        df_clean (pd.DataFrame): DataFrame limpo com transações
        snapshot_date (datetime): Data de referência para cálculo da recência
        
    Returns:
        pd.DataFrame: DataFrame com métricas RFM indexado por CustomerID
    """
    rfm_calculator = RFMCalculatorTransformer(snapshot_date=snapshot_date)
    rfm_data = rfm_calculator.fit_transform(df_clean)
    
    print("Métricas RFM calculadas:")
    print(f"- Total de clientes: {len(rfm_data)}")
    print(f"- Recência média: {rfm_data['Recency'].mean():.1f} dias")
    print(f"- Frequência média: {rfm_data['Frequency'].mean():.1f} pedidos")
    print(f"- Valor monetário médio: £{rfm_data['Monetary'].mean():,.2f}")
    
    return rfm_data