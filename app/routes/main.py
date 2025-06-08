from flask import Blueprint, render_template, request, jsonify, redirect, url_for
import json
from pathlib import Path

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

@main_bp.route('/results')
def results_page():
    """Página de resultados"""
    results_file = Path('data/processed/batch_results.json')
    results = []
    
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    
    return render_template('results.html', results=results)

@main_bp.route('/analytics')
def analytics_page():
    """Página de análises"""
    return render_template('analytics.html')
