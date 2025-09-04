#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicción de Fallas de Frenos usando Regresión Logística - VERSIÓN WEB
Implementación siguiendo la metodología CRISP-DM con API REST

Autor: [Tu Nombre]
Fecha: [Fecha]
"""

# Importación de librerías necesarias
import pandas as pd
import numpy as np
import pickle
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Crear aplicación FastAPI
app = FastAPI(
    title="🚗 API de Predicción de Fallas de Frenos",
    description="API para predecir fallas de frenos usando Regresión Logística y metodología CRISP-DM",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de datos para la API
class VehiculoInput(BaseModel):
    kms_recorridos: int
    años_uso: int
    ultima_revision: int
    temperatura_frenos: int
    cambios_pastillas: int
    estilo_conduccion: int
    carga_promedio: int
    luz_alarma_freno: int

class PrediccionResponse(BaseModel):
    prediccion: int
    probabilidad_falla: float
    probabilidad_no_falla: float
    riesgo: str
    recomendaciones: List[str]

# Variables globales para el modelo
model = None
scaler = None
feature_names = None

def cargar_modelo():
    """Cargar el modelo entrenado y el scaler"""
    global model, scaler, feature_names
    
    try:
        # Verificar si existen archivos del modelo
        if os.path.exists('modelo_frenos.pkl') and os.path.exists('scaler_frenos.pkl'):
            # Cargar modelo y scaler guardados
            with open('modelo_frenos.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scaler_frenos.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open('feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
            print("✅ Modelo cargado desde archivos guardados")
        else:
            # Entrenar modelo desde cero
            print("🔄 Entrenando modelo desde cero...")
            entrenar_modelo()
            
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        # Entrenar modelo desde cero como respaldo
        entrenar_modelo()

def entrenar_modelo():
    """Entrenar el modelo de regresión logística"""
    global model, scaler, feature_names
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        print("📊 Cargando y procesando datos...")
        
        # Cargar datos
        df = cargar_datos()
        
        # Preparar datos
        X = df.drop('falla_frenos', axis=1)
        y = df['falla_frenos']
        
        # Escalar variables numéricas
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
        
        # Guardar modelo y scaler
        feature_names = X.columns.tolist()
        
        with open('modelo_frenos.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('scaler_frenos.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names, f)
            
        print("✅ Modelo entrenado y guardado exitosamente")
        
        # Evaluar modelo
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        print(f"📈 Accuracy del modelo: {accuracy*100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error al entrenar el modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Error al entrenar el modelo: {str(e)}")

def cargar_datos():
    """Cargar y procesar los datos del CSV"""
    try:
        # Leer el archivo como texto primero para ver su estructura
        with open('falla_frenos.csv', 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()
        
        # Si el archivo tiene comas dentro de comillas, usar separador personalizado
        if ',' in first_line and first_line.count('"') > 0:
            # Leer el archivo y procesar manualmente
            df = pd.read_csv('falla_frenos.csv', header=None)
            
            # La primera fila contiene los nombres de las columnas
            column_names = df.iloc[0, 0].split(',')
            
            # Procesar todas las filas de datos
            data_rows = []
            for idx, row in df.iterrows():
                if idx == 0:  # Saltar la fila de encabezados
                    continue
                # Dividir por comas y limpiar comillas
                values = row.iloc[0].split(',')
                # Convertir a tipos apropiados
                processed_values = []
                for i, val in enumerate(values):
                    val = val.strip().strip('"')
                    processed_values.append(int(val))
                data_rows.append(processed_values)
            
            # Crear DataFrame con los datos procesados
            df = pd.DataFrame(data_rows, columns=column_names)
            
        else:
            # Formato normal de CSV
            df = pd.read_csv('falla_frenos.csv')
        
        print(f"✅ Datos cargados: {df.shape}")
        return df
        
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        raise HTTPException(status_code=500, detail=f"Error al cargar datos: {str(e)}")

def predecir_falla_frenos(vehiculo: VehiculoInput):
    """Predecir si un vehículo se quedará sin frenos"""
    global model, scaler, feature_names
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    try:
        # Crear array con los datos de entrada
        input_data = np.array([[
            vehiculo.kms_recorridos,
            vehiculo.años_uso,
            vehiculo.ultima_revision,
            vehiculo.temperatura_frenos,
            vehiculo.cambios_pastillas,
            vehiculo.estilo_conduccion,
            vehiculo.carga_promedio,
            vehiculo.luz_alarma_freno
        ]])
        
        # Escalar los datos
        input_scaled = scaler.transform(input_data)
        
        # Realizar predicción
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Determinar nivel de riesgo
        if prediction == 1:
            if probability[1] > 0.8:
                riesgo = "ALTO"
                recomendaciones = [
                    "Revisar sistema de frenos inmediatamente",
                    "Cambiar pastillas de freno si es necesario",
                    "Verificar temperatura de operación",
                    "Considerar cambio de estilo de conducción"
                ]
            else:
                riesgo = "MEDIO"
                recomendaciones = [
                    "Revisar sistema de frenos pronto",
                    "Monitorear indicadores de desgaste",
                    "Programar revisión en las próximas semanas"
                ]
        else:
            if probability[0] > 0.8:
                riesgo = "BAJO"
                recomendaciones = [
                    "Mantener mantenimiento regular",
                    "Continuar con revisiones programadas",
                    "Monitorear indicadores de desgaste"
                ]
            else:
                riesgo = "BAJO-MEDIO"
                recomendaciones = [
                    "Mantener mantenimiento regular",
                    "Monitorear indicadores de desgaste",
                    "Considerar revisión preventiva"
                ]
        
        return PrediccionResponse(
            prediccion=prediction,
            probabilidad_falla=probability[1],
            probabilidad_no_falla=probability[0],
            riesgo=riesgo,
            recomendaciones=recomendaciones
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

# Endpoints de la API

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "mensaje": "🚗 API de Predicción de Fallas de Frenos",
        "version": "1.0.0",
        "descripcion": "API para predecir fallas de frenos usando Regresión Logística",
        "endpoints": {
            "/": "Información de la API",
            "/health": "Estado de salud del sistema",
            "/predict": "Realizar predicción de falla de frenos",
            "/model-info": "Información del modelo entrenado"
        }
    }

@app.get("/health")
async def health_check():
    """Verificar el estado de salud del sistema"""
    try:
        if model is not None and scaler is not None:
            return {
                "status": "healthy",
                "modelo_cargado": True,
                "scaler_cargado": True,
                "mensaje": "✅ Sistema funcionando correctamente"
            }
        else:
            return {
                "status": "initializing",
                "modelo_cargado": False,
                "scaler_cargado": False,
                "mensaje": "🔄 Sistema inicializando..."
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "mensaje": "❌ Error en el sistema"
        }

@app.post("/predict", response_model=PrediccionResponse)
async def predict_falla_frenos(vehiculo: VehiculoInput):
    """Predecir si un vehículo se quedará sin frenos"""
    return predecir_falla_frenos(vehiculo)

@app.get("/model-info")
async def model_info():
    """Obtener información del modelo entrenado"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Modelo no disponible")
        
        # Obtener coeficientes del modelo
        coefficients = model.coef_[0]
        
        # Crear información de variables
        variables_info = []
        for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
            variables_info.append({
                "variable": name,
                "coeficiente": float(coef),
                "importancia_absoluta": float(abs(coef)),
                "efecto": "positivo" if coef > 0 else "negativo"
            })
        
        # Ordenar por importancia absoluta
        variables_info.sort(key=lambda x: x["importancia_absoluta"], reverse=True)
        
        return {
            "tipo_modelo": "Regresión Logística",
            "variables_entrada": len(feature_names),
            "variables": variables_info,
            "intercepto": float(model.intercept_[0]),
            "clases": model.classes_.tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener información del modelo: {str(e)}")

@app.get("/example-prediction")
async def example_prediction():
    """Ejemplo de predicción con datos de muestra"""
    try:
        # Ejemplo 1: Vehículo con alto riesgo
        vehiculo_alto_riesgo = VehiculoInput(
            kms_recorridos=150000,
            años_uso=12,
            ultima_revision=18,
            temperatura_frenos=95,
            cambios_pastillas=5,
            estilo_conduccion=1,
            carga_promedio=800,
            luz_alarma_freno=1
        )
        
        # Ejemplo 2: Vehículo con bajo riesgo
        vehiculo_bajo_riesgo = VehiculoInput(
            kms_recorridos=25000,
            años_uso=3,
            ultima_revision=6,
            temperatura_frenos=35,
            cambios_pastillas=1,
            estilo_conduccion=0,
            carga_promedio=300,
            luz_alarma_freno=0
        )
        
        prediccion_alto = predecir_falla_frenos(vehiculo_alto_riesgo)
        prediccion_bajo = predecir_falla_frenos(vehiculo_bajo_riesgo)
        
        return {
            "ejemplo_alto_riesgo": {
                "datos_vehiculo": vehiculo_alto_riesgo.dict(),
                "prediccion": prediccion_alto.dict()
            },
            "ejemplo_bajo_riesgo": {
                "datos_vehiculo": vehiculo_bajo_riesgo.dict(),
                "prediccion": prediccion_bajo.dict()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en ejemplo de predicción: {str(e)}")

# Inicializar el modelo al arrancar la aplicación
@app.on_event("startup")
async def startup_event():
    """Evento que se ejecuta al arrancar la aplicación"""
    print("🚀 Iniciando API de Predicción de Fallas de Frenos...")
    cargar_modelo()
    print("✅ API lista para recibir solicitudes")

if __name__ == "__main__":
    # Para desarrollo local
    uvicorn.run(app, host="0.0.0.0", port=8000)
