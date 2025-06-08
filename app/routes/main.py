from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@main_bp.route('/upload')
def upload_page():
    """Página de upload"""
    return render_template('upload.html')

@main_bp.route('/chatbot')
def chatbot_page():
    """Página do chatbot"""
    return render_template('chatbot.html')

@main_bp.route('/analytics')
def analytics_page():
    """Página de analytics"""
    return render_template('analytics.html')

@main_bp.route('/results')
def results_page():
    """Página de resultados"""
    return render_template('results.html')
