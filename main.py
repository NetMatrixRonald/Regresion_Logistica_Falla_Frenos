
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicción de Fallas de Frenos usando Regresión Logística
Implementación siguiendo la metodología CRISP-DM

Autor: [Tu Nombre]
Fecha: [Fecha]
"""

# Importación de librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Configuración para mostrar gráficos
plt.style.use('default')
sns.set_palette("husl")

def main():
    """
    Función principal que ejecuta todo el flujo CRISP-DM
    """
    print("🚗 PREDICCIÓN DE FALLAS DE FRENOS - REGRESIÓN LOGÍSTICA")
    print("=" * 60)
    
    # ============================================================================
    # PASO 1: COMPRENSIÓN DEL NEGOCIO Y LOS DATOS
    # ============================================================================
    print("\n📊 PASO 1: COMPRENSIÓN DEL NEGOCIO Y LOS DATOS")
    print("-" * 50)
    
    # Cargar el archivo de datos con el formato correcto
    try:
        # Leer el archivo como texto primero para ver su estructura
        with open('falla_frenos.csv', 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()
        
        print(f"🔍 Primera línea del archivo: {first_line}")
        
        # Si el archivo tiene comas dentro de comillas, usar separador personalizado
        if ',' in first_line and first_line.count('"') > 0:
            # Leer el archivo y procesar manualmente
            df = pd.read_csv('falla_frenos.csv', header=None)
            
            # La primera fila contiene los nombres de las columnas
            column_names = df.iloc[0, 0].split(',')
            print(f"📋 Nombres de columnas detectados: {column_names}")
            
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
                    if i in [0, 1, 2, 3, 4, 6, 7, 8]:  # Variables numéricas
                        processed_values.append(int(val))
                    else:  # Variable categórica (estilo_conduccion)
                        processed_values.append(int(val))
                data_rows.append(processed_values)
            
            # Crear DataFrame con los datos procesados
            df = pd.DataFrame(data_rows, columns=column_names)
            
        else:
            # Formato normal de CSV
            df = pd.read_csv('falla_frenos.csv')
        
        print("✅ Archivo de datos cargado exitosamente")
        print(f"📁 Forma del dataset: {df.shape}")
        
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo 'falla_frenos.csv'")
        return
    except Exception as e:
        print(f"❌ Error al cargar el archivo: {e}")
        return
    
    # Mostrar información básica del dataset
    print("\n📋 Información del dataset:")
    print(df.info())
    
    # Mostrar las primeras filas
    print("\n🔍 Primeras 5 filas del dataset:")
    print(df.head())
    
    # Mostrar estadísticas descriptivas
    print("\n📈 Estadísticas descriptivas:")
    print(df.describe())
    
    # Distribución de la variable objetivo
    print("\n🎯 Distribución de la variable objetivo (falla_frenos):")
    target_dist = df['falla_frenos'].value_counts()
    print(target_dist)
    print(f"Porcentaje de fallas: {target_dist[1]/len(df)*100:.2f}%")
    print(f"Porcentaje de no fallas: {target_dist[0]/len(df)*100:.2f}%")
    
    # Visualizar la distribución de la variable objetivo
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    df['falla_frenos'].value_counts().plot(kind='bar', color=['lightblue', 'lightcoral'])
    plt.title('Distribución de Fallas de Frenos')
    plt.xlabel('Falla de Frenos')
    plt.ylabel('Frecuencia')
    plt.xticks([0, 1], ['No (0)', 'Sí (1)'])
    
    # Distribución de variables numéricas por clase
    plt.subplot(1, 2, 2)
    df.boxplot(column='kms_recorridos', by='falla_frenos', ax=plt.gca())
    plt.title('Distribución de KM por Clase de Falla')
    plt.suptitle('')
    plt.tight_layout()
    plt.show()
    
    # ============================================================================
    # PASO 2: PREPARACIÓN DE LOS DATOS
    # ============================================================================
    print("\n🔧 PASO 2: PREPARACIÓN DE LOS DATOS")
    print("-" * 50)
    
    # Verificar valores nulos
    print("🔍 Verificación de valores nulos:")
    print(df.isnull().sum())
    
    if df.isnull().sum().sum() > 0:
        print("⚠️ Se encontraron valores nulos. Tratando...")
        df = df.dropna()
        print(f"✅ Dataset después de limpiar: {df.shape}")
    else:
        print("✅ No se encontraron valores nulos")
    
    # Separar features y variable objetivo
    X = df.drop('falla_frenos', axis=1)
    y = df['falla_frenos']
    
    print(f"\n📊 Features (X): {X.shape}")
    print(f"🎯 Variable objetivo (y): {y.shape}")
    
    # Verificar tipos de datos
    print("\n📋 Tipos de datos de las variables:")
    print(X.dtypes)
    
    # Identificar variables numéricas y categóricas
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n🔢 Variables numéricas: {numeric_features}")
    print(f"🏷️ Variables categóricas: {categorical_features}")
    
    # Escalar variables numéricas
    print("\n⚖️ Escalando variables numéricas...")
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    print("✅ Variables numéricas escaladas con StandardScaler")
    
    # Dividir datos en entrenamiento y prueba (70% - 30%)
    print("\n✂️ Dividiendo datos en entrenamiento (70%) y prueba (30%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"📚 Datos de entrenamiento: {X_train.shape}")
    print(f"🧪 Datos de prueba: {X_test.shape}")
    
    # Verificar distribución balanceada en ambos conjuntos
    print(f"\n📊 Distribución en entrenamiento:")
    print(f"   - No fallas: {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"   - Fallas: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
    
    print(f"\n📊 Distribución en prueba:")
    print(f"   - No fallas: {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)")
    print(f"   - Fallas: {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)")
    
    # ============================================================================
    # PASO 3: MODELADO (REGRESIÓN LOGÍSTICA)
    # ============================================================================
    print("\n🤖 PASO 3: MODELADO - REGRESIÓN LOGÍSTICA")
    print("-" * 50)
    
    # Crear y entrenar el modelo de regresión logística
    print("🏗️ Creando modelo de regresión logística...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    print("🎯 Entrenando el modelo...")
    model.fit(X_train, y_train)
    
    print("✅ Modelo entrenado exitosamente")
    
    # Realizar predicciones
    print("🔮 Realizando predicciones...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print("✅ Predicciones completadas")
    
    # ============================================================================
    # PASO 4: EVALUACIÓN DEL MODELO
    # ============================================================================
    print("\n📊 PASO 4: EVALUACIÓN DEL MODELO")
    print("-" * 50)
    
    # Calcular métricas de evaluación
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("📈 MÉTRICAS DE EVALUACIÓN:")
    print(f"   - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   - Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   - Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"   - F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    
    # Mostrar reporte de clasificación completo
    print("\n📋 REPORTE DE CLASIFICACIÓN COMPLETO:")
    print(classification_report(y_test, y_pred, target_names=['No Falla', 'Falla']))
    
    # Matriz de confusión
    print("\n🔄 MATRIZ DE CONFUSIÓN:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Falla', 'Falla'],
                yticklabels=['No Falla', 'Falla'])
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.show()
    
    # Coeficientes del modelo para interpretar importancia de variables
    print("\n🔍 INTERPRETACIÓN DE VARIABLES:")
    feature_names = X.columns.tolist()
    coefficients = model.coef_[0]
    
    # Crear DataFrame con coeficientes
    coef_df = pd.DataFrame({
        'Variable': feature_names,
        'Coeficiente': coefficients,
        'Importancia_Absoluta': np.abs(coefficients)
    })
    
    # Ordenar por importancia absoluta
    coef_df = coef_df.sort_values('Importancia_Absoluta', ascending=False)
    
    print("📊 Coeficientes del modelo (ordenados por importancia):")
    print(coef_df)
    
    # Visualizar importancia de variables
    plt.figure(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in coef_df['Coeficiente']]
    plt.barh(coef_df['Variable'], coef_df['Coeficiente'], color=colors)
    plt.title('Importancia de Variables en el Modelo de Regresión Logística')
    plt.xlabel('Coeficiente')
    plt.ylabel('Variable')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ============================================================================
    # PASO 5: DESPLIEGUE / USO DEL MODELO
    # ============================================================================
    print("\n🚀 PASO 5: DESPLIEGUE / USO DEL MODELO")
    print("-" * 50)
    
    def predecir_falla_frenos(kms, años, revision, temp, pastillas, estilo, carga, luz_alarma):
        """
        Función para predecir si un vehículo se quedará sin frenos
        
        Parámetros:
        - kms: kilómetros recorridos
        - años: años de uso
        - revision: meses desde la última revisión
        - temp: temperatura de los frenos en °C
        - pastillas: número de cambios de pastillas
        - estilo: estilo de conducción (0=normal, 1=agresivo)
        - carga: carga promedio en kg
        - luz_alarma: luz de alarma de freno (0=no, 1=sí)
        
        Retorna:
        - 1: El vehículo se quedará sin frenos
        - 0: Los frenos funcionan correctamente
        """
        # Crear array con los datos de entrada
        input_data = np.array([[kms, años, revision, temp, pastillas, estilo, carga, luz_alarma]])
        
        # Escalar los datos usando el mismo scaler del entrenamiento
        input_scaled = input_data.copy()
        input_scaled[0, :len(numeric_features)] = scaler.transform(input_data[:, :len(numeric_features)])
        
        # Realizar predicción
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        return prediction, probability
    
    # Ejemplos de uso de la función
    print("🧪 EJEMPLOS DE PREDICCIÓN:")
    print("-" * 30)
    
    # Ejemplo 1: Vehículo con alto riesgo
    print("🚗 Ejemplo 1 - Vehículo con alto riesgo:")
    print("   - KM: 150,000, Años: 12, Revisión: 18 meses")
    print("   - Temp: 95°C, Pastillas: 5, Estilo: Agresivo")
    print("   - Carga: 800kg, Luz alarma: Sí")
    
    pred1, prob1 = predecir_falla_frenos(150000, 12, 18, 95, 5, 1, 800, 1)
    print(f"   🎯 Predicción: {'FALLA (1)' if pred1 == 1 else 'NO FALLA (0)'}")
    print(f"   📊 Probabilidad: {prob1[pred1]*100:.1f}%")
    
    # Ejemplo 2: Vehículo con bajo riesgo
    print("\n🚗 Ejemplo 2 - Vehículo con bajo riesgo:")
    print("   - KM: 25,000, Años: 3, Revisión: 6 meses")
    print("   - Temp: 35°C, Pastillas: 1, Estilo: Normal")
    print("   - Carga: 300kg, Luz alarma: No")
    
    pred2, prob2 = predecir_falla_frenos(25000, 3, 6, 35, 1, 0, 300, 0)
    print(f"   🎯 Predicción: {'FALLA (1)' if pred2 == 1 else 'NO FALLA (0)'}")
    print(f"   📊 Probabilidad: {prob2[pred2]*100:.1f}%")
    
    # Función interactiva para el usuario
    print("\n🎮 FUNCIÓN INTERACTIVA DE PREDICCIÓN:")
    print("=" * 50)
    
    def interfaz_prediccion():
        """Interfaz interactiva para que el usuario ingrese datos"""
        print("\n🔧 INGRESE LOS DATOS DEL VEHÍCULO:")
        print("(Presione Enter para usar valores por defecto)")
        
        try:
            # Valores por defecto
            defaults = {
                'kms': 50000,
                'años': 5,
                'revision': 8,
                'temp': 45,
                'pastillas': 2,
                'estilo': 0,
                'carga': 400,
                'luz_alarma': 0
            }
            
            # Solicitar datos al usuario
            kms = input(f"Kilómetros recorridos [{defaults['kms']}]: ").strip()
            kms = int(kms) if kms else defaults['kms']
            
            años = input(f"Años de uso [{defaults['años']}]: ").strip()
            años = int(años) if años else defaults['años']
            
            revision = input(f"Meses desde última revisión [{defaults['revision']}]: ").strip()
            revision = int(revision) if revision else defaults['revision']
            
            temp = input(f"Temperatura de frenos (°C) [{defaults['temp']}]: ").strip()
            temp = int(temp) if temp else defaults['temp']
            
            pastillas = input(f"Cambios de pastillas [{defaults['pastillas']}]: ").strip()
            pastillas = int(pastillas) if pastillas else defaults['pastillas']
            
            estilo = input(f"Estilo de conducción (0=normal, 1=agresivo) [{defaults['estilo']}]: ").strip()
            estilo = int(estilo) if estilo else defaults['estilo']
            
            carga = input(f"Carga promedio (kg) [{defaults['carga']}]: ").strip()
            carga = int(carga) if carga else defaults['carga']
            
            luz_alarma = input(f"Luz de alarma de freno (0=no, 1=sí) [{defaults['luz_alarma']}]: ").strip()
            luz_alarma = int(luz_alarma) if luz_alarma else defaults['luz_alarma']
            
            # Realizar predicción
            print("\n🔮 ANALIZANDO DATOS...")
            prediction, probability = predecir_falla_frenos(kms, años, revision, temp, pastillas, estilo, carga, luz_alarma)
            
            # Mostrar resultado
            print("\n" + "="*50)
            print("🎯 RESULTADO DE LA PREDICCIÓN:")
            print("="*50)
            
            if prediction == 1:
                print("🚨 ¡ADVERTENCIA! El vehículo se quedará sin frenos")
                print("🔴 RIESGO ALTO - Se requiere intervención inmediata")
            else:
                print("✅ Los frenos funcionan correctamente")
                print("🟢 RIESGO BAJO - El vehículo está en buenas condiciones")
            
            print(f"\n📊 Probabilidad de falla: {probability[1]*100:.1f}%")
            print(f"📊 Probabilidad de no falla: {probability[0]*100:.1f}%")
            
            # Recomendaciones
            print("\n💡 RECOMENDACIONES:")
            if prediction == 1:
                print("   - Revisar sistema de frenos inmediatamente")
                print("   - Cambiar pastillas de freno si es necesario")
                print("   - Verificar temperatura de operación")
                print("   - Considerar cambio de estilo de conducción")
            else:
                print("   - Mantener mantenimiento regular")
                print("   - Continuar con revisiones programadas")
                print("   - Monitorear indicadores de desgaste")
            
            print("="*50)
            
        except ValueError:
            print("❌ Error: Por favor ingrese valores numéricos válidos")
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
    
    # Ejecutar interfaz interactiva
    try:
        interfaz_prediccion()
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego! Programa terminado por el usuario")
    
    # ============================================================================
    # RESUMEN FINAL
    # ============================================================================
    print("\n📋 RESUMEN DEL MODELO:")
    print("=" * 50)
    print(f"✅ Dataset cargado: {df.shape}")
    print(f"✅ Variables de entrada: {len(feature_names)}")
    print(f"✅ Modelo entrenado: Regresión Logística")
    print(f"✅ Accuracy del modelo: {accuracy*100:.2f}%")
    print(f"✅ Variable más importante: {coef_df.iloc[0]['Variable']}")
    print(f"✅ Función de predicción disponible")
    
    print("\n🎉 ¡PROGRAMA COMPLETADO EXITOSAMENTE!")
    print("=" * 50)

if __name__ == "__main__":
    main()
