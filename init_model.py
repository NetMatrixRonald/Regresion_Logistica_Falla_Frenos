#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de inicialización para entrenar y guardar el modelo
Este script debe ejecutarse antes de usar la API web
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def entrenar_y_guardar_modelo():
    """Entrenar el modelo y guardarlo para uso posterior"""
    print("🚀 Iniciando entrenamiento del modelo...")
    
    try:
        # Cargar datos
        print("📊 Cargando datos...")
        with open('falla_frenos.csv', 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()
        
        # Procesar datos
        if ',' in first_line and first_line.count('"') > 0:
            df = pd.read_csv('falla_frenos.csv', header=None)
            column_names = df.iloc[0, 0].split(',')
            
            data_rows = []
            for idx, row in df.iterrows():
                if idx == 0:
                    continue
                values = row.iloc[0].split(',')
                processed_values = []
                for i, val in enumerate(values):
                    val = val.strip().strip('"')
                    processed_values.append(int(val))
                data_rows.append(processed_values)
            
            df = pd.DataFrame(data_rows, columns=column_names)
        else:
            df = pd.read_csv('falla_frenos.csv')
        
        print(f"✅ Datos cargados: {df.shape}")
        
        # Preparar datos
        X = df.drop('falla_frenos', axis=1)
        y = df['falla_frenos']
        
        # Escalar variables numéricas
        print("⚖️ Escalando variables...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Entrenar modelo
        print("🤖 Entrenando modelo de regresión logística...")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        print(f"📈 Accuracy del modelo: {accuracy*100:.2f}%")
        
        # Guardar modelo y componentes
        print("💾 Guardando modelo...")
        with open('modelo_frenos.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open('scaler_frenos.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        with open('feature_names.pkl', 'wb') as f:
            pickle.dump(X.columns.tolist(), f)
        
        print("✅ Modelo entrenado y guardado exitosamente!")
        print("📁 Archivos creados:")
        print("   - modelo_frenos.pkl")
        print("   - scaler_frenos.pkl")
        print("   - feature_names.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        return False

if __name__ == "__main__":
    entrenar_y_guardar_modelo()
