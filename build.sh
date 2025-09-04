#!/bin/bash
# Script de build para Render.com
# Este script se ejecuta durante el proceso de build

echo "🚀 Iniciando proceso de build..."

# Instalar dependencias
echo "📦 Instalando dependencias..."
pip install -r requirements.txt

# Verificar que el archivo CSV existe
if [ ! -f "falla_frenos.csv" ]; then
    echo "❌ Error: No se encontró el archivo falla_frenos.csv"
    exit 1
fi

# Entrenar el modelo
echo "🤖 Entrenando modelo de regresión logística..."
python init_model.py

# Verificar que el modelo se creó correctamente
if [ ! -f "modelo_frenos.pkl" ] || [ ! -f "scaler_frenos.pkl" ] || [ ! -f "feature_names.pkl" ]; then
    echo "❌ Error: No se crearon todos los archivos del modelo"
    exit 1
fi

echo "✅ Build completado exitosamente!"
echo "📁 Archivos del modelo creados:"
ls -la *.pkl
