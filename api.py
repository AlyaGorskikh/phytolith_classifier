from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional, Dict, Any
from ml_model import MLModel
from database import Database
import numpy as np
import base64
import io
import matplotlib.pyplot as plt

app = FastAPI(title="Phytolith Classification API")

# Инициализация моделей и БД
ml_model = MLModel()
db = Database()


# Pydantic модели
class PhytolithData(BaseModel):
    area: float
    convex_area: float
    perimeter: float
    convex_perimeter: float
    length_feret: float
    fiber_length: float
    width: float
    equivalent_diameter: float
    inscribed_radius: float
    form_factor: float
    roundness: float
    convexity: float
    solidity: float
    compactness: float
    aspect_ratio: float
    elongation: float
    curl: float
    l1: float
    l2: float
    subtribe: Optional[str] = None
    cluster: Optional[int] = None


class PhytolithResponse(BaseModel):
    id: int
    area: float
    convex_area: float
    perimeter: float
    convex_perimeter: float
    length_feret: float
    fiber_length: float
    width: float
    equivalent_diameter: float
    inscribed_radius: float
    form_factor: float
    roundness: float
    convexity: float
    solidity: float
    compactness: float
    aspect_ratio: float
    elongation: float
    curl: float
    l1: float
    l2: float
    subtribe: Optional[str]
    cluster: Optional[int]


class ClusterUpdate(BaseModel):
    clusters: Dict[int, int]


class TrainData(BaseModel):
    X: List[List[float]]
    y: List[str]
    feature_names: Optional[List[str]] = None


# Корневой эндпоинт
@app.get("/")
def read_root():
    return {
        "message": "Phytolith Classification API",
        "status": "running",
        "model_trained": ml_model.is_trained()
    }


# Эндпоинты для работы с БД
@app.get("/phytoliths/", response_model=List[Dict[str, Any]])
def get_all_phytoliths():
    """Получение всех фитолитов"""
    phytoliths = db.get_all_phytoliths()
    return [
        {
            'id': p.id,
            'area': p.area,
            'convex_area': p.convex_area,
            'perimeter': p.perimeter,
            'convex_perimeter': p.convex_perimeter,
            'length_feret': p.length_feret,
            'fiber_length': p.fiber_length,
            'width': p.width,
            'equivalent_diameter': p.equivalent_diameter,
            'inscribed_radius': p.inscribed_radius,
            'form_factor': p.form_factor,
            'roundness': p.roundness,
            'convexity': p.convexity,
            'solidity': p.solidity,
            'compactness': p.compactness,
            'aspect_ratio': p.aspect_ratio,
            'elongation': p.elongation,
            'curl': p.curl,
            'l1': p.l1,
            'l2': p.l2,
            'subtribe': p.subtribe,
            'cluster': p.cluster
        }
        for p in phytoliths
    ]


@app.post("/phytoliths/")
def add_phytolith(data: PhytolithData):
    """Добавление одного фитолита"""
    phytolith_id = db.add_phytolith(**data.dict())
    return {"id": phytolith_id, "success": True}


@app.post("/phytoliths/batch/")
def add_phytolith_batch(data: List[Dict[str, Any]]):
    """Добавление нескольких фитолитов"""
    try:
        # Конвертируем в DataFrame
        df = pd.DataFrame(data)

        # Обрабатываем возможные NaN значения
        df = df.replace({np.nan: None})

        count = db.add_batch_from_df(df)
        return {"count": count, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/phytoliths/update_clusters/")
def update_clusters(data: Dict[int, int]):
    """Обновление кластеров"""
    try:
        db.update_clusters(data)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/phytoliths/")
def clear_table():
    """Очистка таблицы"""
    try:
        db.clear_table()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/phytoliths/dataframe/")
def get_dataframe():
    """Получение DataFrame"""
    try:
        df = db.get_dataframe()

        # Заменяем NaN на None для JSON сериализации
        df = df.replace({np.nan: None})

        return {
            "data": df.to_dict('records'),
            "columns": list(df.columns),
            "shape": df.shape
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Эндпоинты для работы с моделью
@app.get("/model/status/")
def model_status():
    """Статус модели"""
    return {
        "is_trained": ml_model.is_trained(),
        "accuracy": float(ml_model.accuracy) if hasattr(ml_model, 'accuracy') else 0,
        "classes": ml_model.get_class_labels() if ml_model.is_trained() else []
    }


@app.post("/model/load/")
def load_models():
    """Загрузка моделей"""
    success = ml_model.load_models()
    return {"success": success}


@app.post("/model/train/")
def train_model(data: TrainData):
    """Обучение модели"""
    try:
        # Конвертируем данные в DataFrame
        feature_names = data.feature_names if data.feature_names else ml_model.features
        X = pd.DataFrame(data.X, columns=feature_names)
        y = pd.Series(data.y)

        accuracy, X_test, y_test, y_pred = ml_model.train_classifier(X, y)

        if accuracy > 0:
            ml_model.save_models()

            # Получаем отчет классификации
            report = ml_model.get_classification_report()

            return {
                "success": True,
                "accuracy": float(accuracy),
                "n_samples": len(X),
                "n_classes": len(y.unique()),
                "classification_report": report
            }
        else:
            return {
                "success": False,
                "error": "Не удалось обучить модель"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/predict/")
def predict(data: List[Dict[str, float]]):
    """Предсказание для данных"""
    if not ml_model.is_trained():
        raise HTTPException(status_code=400, detail="Model not trained")

    try:
        df = pd.DataFrame(data)
        predictions = ml_model.predict(df)
        probabilities = ml_model.predict_proba(df)

        if predictions is not None:
            return {
                "success": True,
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist() if probabilities is not None else None,
                "classes": ml_model.get_class_labels()
            }
        else:
            return {
                "success": False,
                "error": "Ошибка предсказания"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/cluster/")
def cluster(data: List[Dict[str, float]], n_clusters: int = 3):
    """Кластеризация данных с использованием существующего MLModel"""
    try:
        df = pd.DataFrame(data)

        # Проверяем, обучен ли scaler
        from sklearn.preprocessing import StandardScaler

        if not hasattr(ml_model.scaler, 'mean_'):
            X_scaled = ml_model.scaler.fit_transform(df)
        else:
            X_scaled = ml_model.scaler.transform(df)

        # Создаём кластеризатор с нужным n_clusters
        from sklearn.cluster import AgglomerativeClustering
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        clusters = clusterer.fit_predict(X_scaled)

        # PCA для визуализации
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        return {
            "success": True,
            "clusters": clusters.tolist(),
            "X_pca": X_pca.tolist(),
            "n_clusters": n_clusters,
            "explained_variance": pca.explained_variance_ratio_.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/visualizations/")
def get_visualizations():
    """Получение визуализаций модели"""
    if not ml_model.is_trained():
        return {"success": False, "error": "Model not trained"}

    try:
        # Получаем графики как base64
        visualizations = {}

        # Матрица ошибок
        fig_cm = ml_model.get_confusion_matrix_plot()
        if fig_cm:
            buf = io.BytesIO()
            fig_cm.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            visualizations['confusion_matrix'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig_cm)

        # Важность признаков
        fig_fi = ml_model.get_feature_importance_plot()
        if fig_fi:
            buf = io.BytesIO()
            fig_fi.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            visualizations['feature_importance'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig_fi)

        # Распределение классов
        fig_cd = ml_model.get_class_distribution_plot()
        if fig_cd:
            buf = io.BytesIO()
            fig_cd.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            visualizations['class_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig_cd)

        # Отчет классификации
        report = ml_model.get_classification_report()
        if report:
            visualizations['classification_report'] = report

        visualizations['accuracy'] = float(ml_model.accuracy)
        visualizations['classes'] = ml_model.get_class_labels()
        visualizations['success'] = True

        return visualizations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)