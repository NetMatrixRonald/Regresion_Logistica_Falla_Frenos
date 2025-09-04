# 🚀 Despliegue de la API de Predicción de Fallas de Frenos

## 📋 Descripción

Este proyecto ahora incluye una **versión unificada** que funciona tanto como **script de consola** como **API web**, todo en un solo archivo `main.py`.

## 🌐 Versiones Disponibles

### 1. **`main.py`** - Versión Unificada (Recomendada)
- ✅ **Script de consola**: Ejecuta todo el flujo CRISP-DM
- ✅ **API web**: Endpoints REST con FastAPI
- ✅ **Uso local**: `python main.py`
- ✅ **Uso web**: `uvicorn main:app --host 0.0.0.0 --port 8000`

### 2. **`init_model.py`** - Inicialización del Modelo
- Script para entrenar y guardar el modelo antes del despliegue
- **Uso**: `python init_model.py`

## 🚀 Despliegue en Render.com

### **Paso 1: Preparar el Repositorio**
Asegúrate de que tu repositorio contenga:
```
proyecto/
├── main.py              # ✅ Versión unificada (consola + API)
├── init_model.py        # ✅ Script de inicialización
├── falla_frenos.csv     # ✅ Dataset
├── requirements.txt      # ✅ Dependencias
├── render.yaml          # ✅ Configuración de Render
├── Procfile             # ✅ Comando de inicio
└── README_DEPLOY.md     # ✅ Este archivo
```

### **Paso 2: Conectar con Render**
1. Ve a [render.com](https://render.com)
2. Conecta tu repositorio de GitHub/GitLab
3. Selecciona el repositorio del proyecto

### **Paso 3: Configurar el Servicio**
Render detectará automáticamente la configuración desde `render.yaml`:

```yaml
services:
  - type: web
    name: api-prediccion-frenos
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### **Paso 4: Desplegar**
1. Click en "Create New Service"
2. Selecciona "Web Service"
3. Render usará automáticamente la configuración del `render.yaml`
4. Click en "Create Web Service"

## 🔧 Configuración Manual (Alternativa)

Si prefieres configurar manualmente:

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment**: Python 3.9

## 🌍 Endpoints de la API

Una vez desplegada, tu API tendrá estos endpoints:

### **GET /** - Información de la API
```json
{
  "mensaje": "🚗 API de Predicción de Fallas de Frenos",
  "version": "1.0.0",
  "endpoints": {...}
}
```

### **GET /health** - Estado del Sistema
```json
{
  "status": "healthy",
  "modelo_cargado": true,
  "scaler_cargado": true
}
```

### **POST /predict** - Realizar Predicción
```json
{
  "kms_recorridos": 150000,
  "años_uso": 12,
  "ultima_revision": 18,
  "temperatura_frenos": 95,
  "cambios_pastillas": 5,
  "estilo_conduccion": 1,
  "carga_promedio": 800,
  "luz_alarma_freno": 1
}
```

### **GET /model-info** - Información del Modelo
```json
{
  "tipo_modelo": "Regresión Logística",
  "variables_entrada": 8,
  "variables": [...],
  "intercepto": -0.123
}
```

### **GET /example-prediction** - Ejemplos de Predicción
```json
{
  "ejemplo_alto_riesgo": {...},
  "ejemplo_bajo_riesgo": {...}
}
```

## 📱 Uso de la API

### **Ejemplo con cURL:**
```bash
curl -X POST "https://tu-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "kms_recorridos": 100000,
    "años_uso": 8,
    "ultima_revision": 12,
    "temperatura_frenos": 75,
    "cambios_pastillas": 3,
    "estilo_conduccion": 1,
    "carga_promedio": 600,
    "luz_alarma_freno": 1
  }'
```

### **Ejemplo con Python:**
```python
import requests

url = "https://tu-api.onrender.com/predict"
data = {
    "kms_recorridos": 100000,
    "años_uso": 8,
    "ultima_revision": 12,
    "temperatura_frenos": 75,
    "cambios_pastillas": 3,
    "estilo_conduccion": 1,
    "carga_promedio": 600,
    "luz_alarma_freno": 1
}

response = requests.post(url, json=data)
prediccion = response.json()
print(f"Predicción: {'FALLA' if prediccion['prediccion'] == 1 else 'NO FALLA'}")
print(f"Probabilidad: {prediccion['probabilidad_falla']*100:.1f}%")
```

### **Ejemplo con JavaScript:**
```javascript
fetch('https://tu-api.onrender.com/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    kms_recorridos: 100000,
    años_uso: 8,
    ultima_revision: 12,
    temperatura_frenos: 75,
    cambios_pastillas: 3,
    estilo_conduccion: 1,
    carga_promedio: 600,
    luz_alarma_freno: 1
  })
})
.then(response => response.json())
.then(data => {
  console.log('Predicción:', data.prediccion === 1 ? 'FALLA' : 'NO FALLA');
  console.log('Probabilidad:', (data.probabilidad_falla * 100).toFixed(1) + '%');
});
```

## 🔍 Documentación Automática

Una vez desplegada, tu API incluirá:
- **Swagger UI**: `/docs` - Documentación interactiva
- **ReDoc**: `/redoc` - Documentación alternativa
- **OpenAPI**: `/openapi.json` - Especificación de la API

## 🚨 Solución de Problemas

### **Error: "Attribute 'app' not found in module 'main'"**
- ✅ **SOLUCIONADO**: Ahora `main.py` incluye la aplicación FastAPI
- ✅ Verifica que el start command sea: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### **Error: "Modelo no disponible"**
- ✅ El modelo se entrena automáticamente al ejecutar `main.py`
- ✅ Para uso solo de API, ejecuta primero: `python init_model.py`

### **Error: "uvicorn: command not found"**
- ✅ Asegúrate de que `uvicorn[standard]` esté en `requirements.txt`
- ✅ Verifica que el build command sea `pip install -r requirements.txt`

### **Error: "Module not found"**
- ✅ Verifica que todas las dependencias estén en `requirements.txt`
- ✅ Asegúrate de que el archivo principal sea `main.py`

## 🌟 Ventajas de la Nueva Configuración

1. **Unificación**: Un solo archivo para consola y web
2. **Simplicidad**: Menos archivos para mantener
3. **Flexibilidad**: Funciona en modo consola o API
4. **Compatibilidad**: Funciona con cualquier plataforma de despliegue
5. **Mantenimiento**: Código centralizado y fácil de actualizar

## 🔄 Flujo de Trabajo Recomendado

### **Para Desarrollo Local:**
```bash
# Ejecutar script completo con interfaz interactiva
python main.py
```

### **Para Despliegue Web:**
```bash
# Opción 1: Entrenar modelo primero
python init_model.py
uvicorn main:app --host 0.0.0.0 --port 8000

# Opción 2: Ejecutar directamente (entrena automáticamente)
uvicorn main:app --host 0.0.0.0 --port 8000
```

### **Para Render:**
- Render ejecutará automáticamente: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- El modelo se entrenará automáticamente al iniciar la API

## 📞 Soporte

Si tienes problemas con el despliegue:
1. Revisa los logs en Render
2. Verifica que todos los archivos estén presentes
3. Asegúrate de que las dependencias sean correctas
4. El error de "app not found" ya está solucionado

---

**¡Tu API de predicción de fallas de frenos estará disponible globalmente! 🚗🌍**
