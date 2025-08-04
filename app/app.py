from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
import joblib
import pandas as pd
import os
from datetime import datetime
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_change_in_production_2025_secure_rfm_app')

# --- Configurações da Aplicação ---
class Config:
    MODELS_PATH = 'models'
    MAX_RECENCY = 365  # Máximo de dias para recência
    MAX_FREQUENCY = 300  # Máximo de pedidos
    MAX_MONETARY = 300000  # Máximo valor monetário
    
    # Configurações de validação
    VALIDATION_RULES = {
        'recency': {'min': 0, 'max': MAX_RECENCY},
        'frequency': {'min': 1, 'max': MAX_FREQUENCY},
        'monetary': {'min': 0.01, 'max': MAX_MONETARY}
    }

# --- Carregamento dos Modelos ---
class ModelManager:
    def __init__(self):
        self.rfm_scaler = None
        self.rfm_kmeans_model = None
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Carrega os modelos RFM com tratamento de erro robusto"""
        try:
            scaler_path = os.path.join(Config.MODELS_PATH, 'rfm_scaler.pkl')
            kmeans_path = os.path.join(Config.MODELS_PATH, 'rfm_kmeans_model.pkl')
            
            if not os.path.exists(scaler_path) or not os.path.exists(kmeans_path):
                logger.error(f"Arquivos de modelo não encontrados em {Config.MODELS_PATH}/")
                return False
            
            self.rfm_scaler = joblib.load(scaler_path)
            self.rfm_kmeans_model = joblib.load(kmeans_path)
            
            logger.info("✅ Modelos RFM carregados com sucesso")
            self.models_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelos: {e}")
            self.models_loaded = False
            return False
    
    def predict_cluster(self, recency, frequency, monetary):
        """Prediz o cluster RFM para um cliente"""
        if not self.models_loaded:
            raise ValueError("Modelos não carregados")
        
        customer_data = pd.DataFrame([[recency, frequency, monetary]],
                                   columns=['Recency', 'Frequency', 'Monetary'])
        
        customer_scaled = self.rfm_scaler.transform(customer_data)
        predicted_cluster = self.rfm_kmeans_model.predict(customer_scaled)[0]
        
        return int(predicted_cluster)

# Instância global do gerenciador de modelos
model_manager = ModelManager()

# --- Definições de Clusters Atualizadas ---
CLUSTER_DEFINITIONS = {
    0: {
        "name": "Clientes Regulares Ativos",
        "description": "O maior grupo de clientes. Compram com frequência moderada e valor médio, sendo a base consistente do negócio.",
        "characteristics": "Recência: Média (44 dias) | Frequência: Média-Baixa (3.6 pedidos) | Monetário: Médio (£1.339)",
        "strategy": "Focar na retenção, incentivar compras recorrentes e explorar oportunidades de upsell.",
        "color": "#28a745",
        "icon": "🌱"
    },
    1: {
        "name": "Clientes Adormecidos / Em Risco",
        "description": "Clientes que não compram há muito tempo e têm baixo valor. Apresentam alto risco de churn e precisam de reativação.",
        "characteristics": "Recência: Alta (249 dias) | Frequência: Baixa (1.5 pedidos) | Monetário: Baixo (£478)",
        "strategy": "Campanhas de reengajamento com ofertas personalizadas, pesquisas de satisfação ou descontos de 'volta'.",
        "color": "#dc3545",
        "icon": "⚠️"
    },
    2: {
        "name": "Clientes Leais de Alto Valor",
        "description": "Compram muito recentemente, com alta frequência e gastam um valor significativo. São clientes fiéis e valiosos.",
        "characteristics": "Recência: Baixa (16 dias) | Frequência: Alta (21 pedidos) | Monetário: Alto (£12.832)",
        "strategy": "Programas de fidelidade, atendimento prioritário, e ofertas exclusivas para manter o engajamento e a lealdade.",
        "color": "#007bff",
        "icon": "🤝"
    },
    3: {
        "name": "Super Campeões (Alta Frequência)",
        "description": "Um grupo pequeno, mas extremamente ativo. Compram com altíssima frequência e geram um valor muito alto. Podem ser atacadistas ou grandes entusiastas.",
        "characteristics": "Recência: Muito Baixa (6.5 dias) | Frequência: Muito Alta (120.5 pedidos) | Monetário: Muito Alto (£55.313)",
        "strategy": "Reconhecimento VIP, acesso antecipado a produtos, comunicação personalizada e feedback direto para aprimorar a experiência.",
        "color": "#ffc107",
        "icon": "🏆"
    },
    4: {
        "name": "Ultra VIPs (Maior Valor Monetário)",
        "description": "O grupo mais valioso em termos de gasto total. Compram muito recentemente e com frequência alta, gerando receita excepcional.",
        "characteristics": "Recência: Muito Baixa (7.7 dias) | Frequência: Alta (42.8 pedidos) | Monetário: Extremamente Alto (£190.863)",
        "strategy": "Tratamento exclusivo, gerente de conta dedicado, convites para eventos especiais e personalização máxima para garantir a satisfação contínua.",
        "color": "#6f42c1",
        "icon": "👑"
    }
}

def validate_input(recency, frequency, monetary):
    """Valida os inputs do usuário"""
    errors = []
    
    # Validação de recência
    if not (Config.VALIDATION_RULES['recency']['min'] <= recency <= Config.VALIDATION_RULES['recency']['max']):
        errors.append(f"Recência deve estar entre {Config.VALIDATION_RULES['recency']['min']} e {Config.VALIDATION_RULES['recency']['max']} dias")
    
    # Validação de frequência
    if not (Config.VALIDATION_RULES['frequency']['min'] <= frequency <= Config.VALIDATION_RULES['frequency']['max']):
        errors.append(f"Frequência deve estar entre {Config.VALIDATION_RULES['frequency']['min']} e {Config.VALIDATION_RULES['frequency']['max']} pedidos")
    
    # Validação de valor monetário
    if not (Config.VALIDATION_RULES['monetary']['min'] <= monetary <= Config.VALIDATION_RULES['monetary']['max']):
        errors.append(f"Valor monetário deve estar entre £{Config.VALIDATION_RULES['monetary']['min']:.2f} e £{Config.VALIDATION_RULES['monetary']['max']:.2f}")
    
    return errors

@app.route('/', methods=['GET', 'POST'])
def index():
    """Rota principal da aplicação"""
    prediction_result = None
    
    if request.method == 'POST':
        try:
            # Extração e conversão dos dados
            recency = int(request.form.get('recency_manual', 0))
            frequency = int(request.form.get('frequency_manual', 0))
            monetary = float(request.form.get('monetary_manual', 0))
            
            # Validação dos inputs
            validation_errors = validate_input(recency, frequency, monetary)
            if validation_errors:
                for error in validation_errors:
                    flash(error, "error")
                return render_template('index.html', 
                                     cluster_definitions=CLUSTER_DEFINITIONS,
                                     models_available=model_manager.models_loaded,
                                     config=Config)
            
            # Verificação se os modelos estão carregados
            if not model_manager.models_loaded:
                flash("❌ Modelos não disponíveis. Verifique se os arquivos estão na pasta 'models/'", "error")
                return render_template('index.html', 
                                     cluster_definitions=CLUSTER_DEFINITIONS,
                                     models_available=model_manager.models_loaded,
                                     config=Config)
            
            # Predição do cluster
            predicted_cluster = model_manager.predict_cluster(recency, frequency, monetary)
            cluster_info = CLUSTER_DEFINITIONS.get(predicted_cluster, {
                "name": "Cluster Desconhecido",
                "description": "Cluster não identificado",
                "characteristics": "Sem informações disponíveis",
                "strategy": "Análise manual necessária",
                "color": "#6c757d",
                "icon": "❓"
            })
            
            prediction_result = {
                'cluster_id': predicted_cluster,
                'cluster_info': cluster_info,
                'input_data': {
                    'recency': recency,
                    'frequency': frequency,
                    'monetary': monetary
                },
                'timestamp': datetime.now().strftime("%d/%m/%Y às %H:%M")
            }
            
            flash(f"✅ Análise realizada com sucesso! Cliente classificado como: {cluster_info['name']}", "success")
            
        except ValueError as e:
            flash("❌ Erro de validação: Verifique se todos os campos foram preenchidos com valores numéricos válidos", "error")
            logger.error(f"Erro de validação: {e}")
        except Exception as e:
            flash(f"❌ Erro interno: {str(e)}", "error")
            logger.error(f"Erro na predição: {e}")
    
    return render_template('index.html', 
                         prediction_result=prediction_result,
                         cluster_definitions=CLUSTER_DEFINITIONS,
                         models_available=model_manager.models_loaded,
                         config=Config)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint para predição (para futuras integrações)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dados não fornecidos'}), 400
        
        recency = data.get('recency')
        frequency = data.get('frequency')
        monetary = data.get('monetary')
        
        if None in [recency, frequency, monetary]:
            return jsonify({'error': 'Campos obrigatórios: recency, frequency, monetary'}), 400
        
        # Validação
        validation_errors = validate_input(recency, frequency, monetary)
        if validation_errors:
            return jsonify({'error': validation_errors}), 400
        
        if not model_manager.models_loaded:
            return jsonify({'error': 'Modelos não disponíveis'}), 503
        
        # Predição
        predicted_cluster = model_manager.predict_cluster(recency, frequency, monetary)
        cluster_info = CLUSTER_DEFINITIONS.get(predicted_cluster)
        
        return jsonify({
            'cluster_id': predicted_cluster,
            'cluster_name': cluster_info['name'],
            'description': cluster_info['description'],
            'strategy': cluster_info['strategy'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro na API: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

@app.route('/health')
def health_check():
    """Endpoint de health check"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': model_manager.models_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    """Handler para páginas não encontradas"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handler para erros internos"""
    logger.error(f"Erro interno: {error}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("🚀 Iniciando aplicação RFM Customer Segmentation...")
    print(f"📁 Procurando modelos em: {Config.MODELS_PATH}/")
    print(f"🔧 Modelos carregados: {'✅' if model_manager.models_loaded else '❌'}")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )