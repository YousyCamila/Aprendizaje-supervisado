# =============================================================================
# MODELO DE APRENDIZAJE SUPERVISADO - ÁRBOL DE DECISIÓN
# Proyecto: Sistema de Transporte Masivo
# Curso: Inteligencia Artificial - Corporación Universitaria Iberoamericana
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
# =============================================================================

print("=" * 65)
print("  MODELO DE PREDICCIÓN DE RETRASOS - TRANSPORTE MASIVO")
print("=" * 65)

df = pd.read_csv("dataset_transporte.csv")

print("\n INFORMACIÓN GENERAL DEL DATASET")
print(f"   Registros     : {df.shape[0]}")
print(f"   Variables      : {df.shape[1]}")
print(f"\n   Columnas: {list(df.columns)}")

print("\nPRIMEROS 5 REGISTROS:")
print(df.head().to_string(index=False))

print("\n ESTADÍSTICAS DESCRIPTIVAS:")
print(df.describe().round(2).to_string())

print("\n  DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (clasificacion_retraso):")
dist = df["clasificacion_retraso"].value_counts()
for clase, count in dist.items():
    pct = count / len(df) * 100
    print(f"   {clase:<10}: {count:>3} registros  ({pct:.1f}%)")

# =============================================================================
# 2. PREPROCESAMIENTO
# =============================================================================

print("\n" + "=" * 65)
print("  PREPROCESAMIENTO DE DATOS")
print("=" * 65)


features = [
    "hora_dia", "dia_semana", "linea", "distancia_km",
    "tiempo_viaje_min", "ocupacion_porc", "temperatura_c",
    "lluvia", "evento_especial"
]
target = "clasificacion_retraso"

X = df[features]
y = df[target]


le = LabelEncoder()
y_encoded = le.fit_transform(y)
clases = le.classes_

print(f"\n   Features usadas: {features}")
print(f"   Variable objetivo: {target}")
print(f"   Clases codificadas: {dict(zip(range(len(clases)), clases))}")


nulos = X.isnull().sum().sum()
print(f"   Valores nulos: {nulos}")

# =============================================================================
# 3. DIVISIÓN TRAIN / TEST
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

print(f"\n   Datos de entrenamiento : {X_train.shape[0]} registros (75%)")
print(f"   Datos de prueba        : {X_test.shape[0]} registros (25%)")

# =============================================================================
# 4. ENTRENAMIENTO DEL MODELO
# =============================================================================

print("\n" + "=" * 65)
print("  ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN")
print("=" * 65)

modelo = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

modelo.fit(X_train, y_train)

print("\n   Parámetros del modelo:")
print(f"   - Criterio       : {modelo.criterion}")
print(f"   - Profundidad máx: {modelo.max_depth}")
print(f"   - Min muestras split: {modelo.min_samples_split}")
print(f"   - Min muestras hoja : {modelo.min_samples_leaf}")
print(f"   - Profundidad real  : {modelo.get_depth()}")
print(f"   - Nodos hoja        : {modelo.get_n_leaves()}")

# =============================================================================
# 5. EVALUACIÓN DEL MODELO
# =============================================================================

print("\n" + "=" * 65)
print("  EVALUACIÓN DEL MODELO")
print("=" * 65)

y_pred = modelo.predict(X_test)
y_pred_train = modelo.predict(X_train)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred)

print(f"\n   Accuracy (entrenamiento): {acc_train:.4f}  ({acc_train*100:.2f}%)")
print(f"   Accuracy (prueba)        : {acc_test:.4f}  ({acc_test*100:.2f}%)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(modelo, X, y_encoded, cv=cv, scoring="accuracy")
print(f"\n   Validación cruzada (5-fold):")
print(f"   - Scores: {[round(s, 4) for s in cv_scores]}")
print(f"   - Media : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n   REPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_pred, target_names=clases))


print("   IMPORTANCIA DE VARIABLES:")
importancias = pd.Series(modelo.feature_importances_, index=features)
importancias_sorted = importancias.sort_values(ascending=False)
for feat, imp in importancias_sorted.items():
    bar = "█" * int(imp * 40)
    print(f"   {feat:<22}: {imp:.4f}  {bar}")

# =============================================================================
# 6. PREDICCIÓN CON CASO NUEVO
# =============================================================================

print("\n" + "=" * 65)
print("  PREDICCIÓN: CASO DE USO REAL")
print("=" * 65)


caso_nuevo = pd.DataFrame([{
    "hora_dia": 17,
    "dia_semana": 1,
    "linea": 1,
    "distancia_km": 5.5,
    "tiempo_viaje_min": 22,
    "ocupacion_porc": 95,
    "temperatura_c": 22,
    "lluvia": 0,
    "evento_especial": 0
}])

pred_encoded = modelo.predict(caso_nuevo)[0]
pred_clase = le.inverse_transform([pred_encoded])[0]
pred_proba = modelo.predict_proba(caso_nuevo)[0]

print("\n   Parámetros del viaje a predecir:")
print(f"   - Hora: 17:00  |  Línea: 1  |  Ocupación: 95%")
print(f"   - Distancia: 5.5 km  |  Lluvia: No  |  Evento especial: No")
print(f"\n   ▶ Clasificación predicha: {pred_clase.upper()}")
print("\n   Probabilidades por clase:")
for clase, proba in zip(clases, pred_proba):
    bar = "█" * int(proba * 30)
    print(f"   {clase:<10}: {proba:.4f}  {bar}")

# =============================================================================
# 7. VISUALIZACIONES
# =============================================================================

print("\n" + "=" * 65)
print("  GENERANDO VISUALIZACIONES...")
print("=" * 65)

colores = {"Alto": "#E74C3C", "Medio": "#F39C12", "Bajo": "#3498DB", "Ninguno": "#2ECC71"}

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "Análisis del Modelo de Predicción de Retrasos\nTransporte Masivo — Árbol de Decisión",
    fontsize=15, fontweight="bold", y=0.98
)

ax1 = axes[0, 0]
clases_dist = df["clasificacion_retraso"].value_counts()
bars = ax1.bar(
    clases_dist.index,
    clases_dist.values,
    color=[colores[c] for c in clases_dist.index],
    edgecolor="white", linewidth=1.5
)
ax1.set_title("Distribución de Clases", fontweight="bold")
ax1.set_xlabel("Clasificación de Retraso")
ax1.set_ylabel("Frecuencia")
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(bar.get_height()), ha="center", fontweight="bold")
ax1.set_facecolor("#f8f9fa")
ax1.grid(axis="y", alpha=0.4)


ax2 = axes[0, 1]
imp_sorted = importancias.sort_values()
colors_imp = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(imp_sorted)))
bars2 = ax2.barh(imp_sorted.index, imp_sorted.values, color=colors_imp, edgecolor="white")
ax2.set_title("Importancia de Variables", fontweight="bold")
ax2.set_xlabel("Importancia (Gini)")
for i, (val, bar) in enumerate(zip(imp_sorted.values, bars2)):
    ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=8)
ax2.set_facecolor("#f8f9fa")
ax2.grid(axis="x", alpha=0.4)


ax3 = axes[0, 2]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
disp.plot(ax=ax3, colorbar=False, cmap="Blues")
ax3.set_title("Matriz de Confusión (Test)", fontweight="bold")

ax4 = axes[1, 0]
for clase in clases_dist.index:
    subset = df[df["clasificacion_retraso"] == clase]["ocupacion_porc"]
    ax4.hist(subset, bins=12, alpha=0.65, label=clase,
             color=colores[clase], edgecolor="white")
ax4.set_title("Ocupación (%) por Tipo de Retraso", fontweight="bold")
ax4.set_xlabel("Ocupación (%)")
ax4.set_ylabel("Frecuencia")
ax4.legend()
ax4.set_facecolor("#f8f9fa")
ax4.grid(alpha=0.3)

# 5. Retrasos por hora
ax5 = axes[1, 1]
hora_clase = df.groupby(["hora_dia", "clasificacion_retraso"]).size().unstack(fill_value=0)
horas = sorted(df["hora_dia"].unique())
bottom = np.zeros(len(hora_clase))
for clase in ["Ninguno", "Bajo", "Medio", "Alto"]:
    if clase in hora_clase.columns:
        ax5.bar(hora_clase.index, hora_clase[clase], bottom=bottom,
                color=colores[clase], label=clase, edgecolor="white")
        bottom += hora_clase[clase].values
ax5.set_title("Distribución de Retrasos por Hora", fontweight="bold")
ax5.set_xlabel("Hora del día")
ax5.set_ylabel("Cantidad de viajes")
ax5.legend(loc="upper left", fontsize=8)
ax5.set_facecolor("#f8f9fa")
ax5.grid(axis="y", alpha=0.3)

# 6. Métricas del modelo
ax6 = axes[1, 2]
ax6.axis("off")
metricas_texto = [
    ("MÉTRICAS DEL MODELO", "", True),
    ("", "", False),
    ("Accuracy (Train)", f"{acc_train*100:.2f}%", False),
    ("Accuracy (Test)", f"{acc_test*100:.2f}%", False),
    ("CV Mean (5-fold)", f"{cv_scores.mean()*100:.2f}%", False),
    ("CV Std", f"± {cv_scores.std()*100:.2f}%", False),
    ("", "", False),
    ("PARÁMETROS", "", True),
    ("Criterio", "Gini", False),
    ("Profundidad máx", "5", False),
    ("Nodos hoja", str(modelo.get_n_leaves()), False),
    ("Profundidad real", str(modelo.get_depth()), False),
    ("", "", False),
    ("DATOS", "", True),
    ("Total registros", "100", False),
    ("Train / Test", "75 / 25", False),
    ("Features", "9", False),
    ("Clases", "4", False),
]
y_pos = 0.97
for label, valor, bold in metricas_texto:
    if bold:
        ax6.text(0.05, y_pos, label, transform=ax6.transAxes,
                 fontsize=10, fontweight="bold", color="#2C3E50")
    elif label:
        ax6.text(0.05, y_pos, f"{label}:", transform=ax6.transAxes,
                 fontsize=9, color="#555")
        ax6.text(0.70, y_pos, valor, transform=ax6.transAxes,
                 fontsize=9, fontweight="bold", color="#E74C3C")
    y_pos -= 0.055

ax6.set_facecolor("#f8f9fa")
ax6.set_title("Resumen del Modelo", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("resultados_modelo.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Guardado: resultados_modelo.png")

# Árbol de decisión visual
fig2, ax_tree = plt.subplots(figsize=(22, 10))
plot_tree(
    modelo,
    feature_names=features,
    class_names=clases,
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=4,
    ax=ax_tree,
    impurity=True,
    proportion=False
)
ax_tree.set_title(
    "Árbol de Decisión — Predicción de Retrasos en Transporte Masivo",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("arbol_decision.png", dpi=120, bbox_inches="tight")
plt.close()
print("    Guardado: arbol_decision.png")

# =============================================================================
# 8. REGLAS DEL ÁRBOL (texto)
# =============================================================================

print("\n" + "=" * 65)
print("  REGLAS DEL ÁRBOL (primeros niveles)")
print("=" * 65)
reglas = export_text(modelo, feature_names=features, max_depth=3)
print(reglas)

print("\n" + "=" * 65)
print("  PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 65)
