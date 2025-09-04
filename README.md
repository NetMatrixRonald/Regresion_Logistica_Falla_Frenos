# 🚗 Predicción de Fallas de Frenos - Regresión Logística

## 📋 Descripción del Proyecto

Este proyecto implementa un modelo de **Regresión Logística** para predecir si un automóvil se quedará sin frenos, siguiendo la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining).

## 🎯 Objetivo

Predecir si un automóvil se quedará sin frenos (variable objetivo: `falla_frenos`, binaria: 1 = sí, 0 = no) usando únicamente un modelo de Regresión Logística.

## 📊 Variables de Entrada (Features)

- **kms_recorridos** (numérico): Kilómetros recorridos por el vehículo
- **años_uso** (numérico): Años de uso del vehículo
- **ultima_revision** (numérico): Meses desde la última revisión
- **temperatura_frenos** (numérico): Temperatura de los frenos en °C
- **cambios_pastillas** (numérico): Número de cambios de pastillas realizados
- **estilo_conduccion** (categórico): 0 = normal, 1 = agresivo
- **carga_promedio** (numérico): Carga promedio en kg
- **luz_alarma_freno** (binario): 0 = no, 1 = sí

## ⚙️ Implementación CRISP-DM

### 1. **Comprensión del Negocio y los Datos**
- Carga del archivo `falla_frenos.csv`
- Exploración de la estructura y distribución de datos
- Visualización de la variable objetivo

### 2. **Preparación de los Datos**
- Verificación y tratamiento de valores nulos
- Escalado de variables numéricas con StandardScaler
- División de datos: 70% entrenamiento, 30% prueba

### 3. **Modelado**
- Entrenamiento de modelo LogisticRegression de scikit-learn
- Configuración de hiperparámetros

### 4. **Evaluación**
- Cálculo de métricas: accuracy, precision, recall, F1-score
- Matriz de confusión visual
- Interpretación de coeficientes del modelo

### 5. **Despliegue/Uso**
- Función de predicción para nuevos datos
- Interfaz interactiva para entrada manual de valores
- Ejemplos de uso con casos reales

## 📦 Dependencias

```bash
pip install -r requirements.txt
```

### Librerías principales:
- **pandas**: Manipulación y análisis de datos
- **numpy**: Operaciones numéricas
- **scikit-learn**: Machine Learning (LogisticRegression, StandardScaler, métricas)
- **matplotlib**: Visualización de gráficos
- **seaborn**: Gráficos estadísticos avanzados

## 🚀 Instalación y Ejecución

### 1. Clonar o descargar el proyecto
```bash
git clone [URL_DEL_REPOSITORIO]
cd [NOMBRE_DEL_DIRECTORIO]
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el programa
```bash
python main.py
```

## 📁 Estructura de Archivos

```
proyecto/
├── main.py              # Código principal del proyecto
├── falla_frenos.csv     # Dataset de entrenamiento
├── requirements.txt     # Dependencias del proyecto
└── README.md           # Este archivo
```

## 🔍 Características del Código

- ✅ **Completamente comentado** en español
- ✅ **Manejo de errores** robusto
- ✅ **Visualizaciones** informativas
- ✅ **Interfaz interactiva** para predicciones
- ✅ **Métricas completas** de evaluación
- ✅ **Interpretación** de variables importantes
- ✅ **Función de predicción** reutilizable

## 📈 Resultados Esperados

El programa generará:
- Análisis exploratorio de los datos
- Métricas de rendimiento del modelo
- Gráficos de distribución y matriz de confusión
- Interpretación de la importancia de variables
- Función interactiva para nuevas predicciones

## 🎮 Uso de la Función de Predicción

### Ejemplo de uso programático:
```python
from main import predecir_falla_frenos

# Predicción para un vehículo
prediccion, probabilidad = predecir_falla_frenos(
    kms=100000,           # 100,000 km
    años=8,               # 8 años
    revision=12,          # 12 meses sin revisar
    temp=75,              # 75°C
    pastillas=3,          # 3 cambios
    estilo=1,             # Conducción agresiva
    carga=600,            # 600 kg
    luz_alarma=1          # Luz de alarma encendida
)

print(f"Predicción: {'FALLA' if prediccion == 1 else 'NO FALLA'}")
print(f"Probabilidad: {probabilidad[prediccion]*100:.1f}%")
```

### Uso interactivo:
El programa incluye una interfaz interactiva que solicita los datos del vehículo y proporciona:
- Predicción inmediata
- Probabilidades de falla/no falla
- Recomendaciones específicas
- Interpretación de resultados

## 🔧 Personalización

El código está diseñado para ser fácilmente modificable:
- Cambiar la proporción de división train/test
- Ajustar hiperparámetros del modelo
- Agregar nuevas métricas de evaluación
- Modificar las visualizaciones

## 📊 Interpretación de Resultados

### Coeficientes del Modelo:
- **Coeficientes positivos**: Aumentan la probabilidad de falla
- **Coeficientes negativos**: Disminuyen la probabilidad de falla
- **Valor absoluto mayor**: Variable más importante

### Métricas de Evaluación:
- **Accuracy**: Porcentaje total de predicciones correctas
- **Precision**: De los predichos como falla, cuántos realmente fallaron
- **Recall**: De los que realmente fallaron, cuántos fueron detectados
- **F1-Score**: Media armónica entre precision y recall

## 🚨 Casos de Uso

Este modelo es útil para:
- **Talleres mecánicos**: Identificar vehículos en riesgo
- **Flotas de vehículos**: Mantenimiento preventivo
- **Seguros**: Evaluación de riesgo
- **Conductores**: Conciencia sobre el estado de los frenos

## 📝 Notas Técnicas

- El modelo usa **estratificación** para mantener proporciones de clases
- Las variables numéricas se escalan con **StandardScaler**
- Se usa **random_state=42** para reproducibilidad
- El modelo incluye **max_iter=1000** para convergencia

## 🤝 Contribuciones

Para contribuir al proyecto:
1. Fork del repositorio
2. Crear rama para nueva funcionalidad
3. Commit de cambios
4. Pull Request

## 📄 Licencia

Este proyecto está bajo licencia [ESPECIFICAR_LICENCIA]

## 👨‍💻 Autor

[Tu Nombre] - [Fecha]

---

**¡Disfruta usando el modelo de predicción de fallas de frenos! 🚗💨**
