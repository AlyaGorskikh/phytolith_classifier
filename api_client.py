import requests
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Encoder для numpy типов"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)


class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def _handle_response(self, response):
        """Обработка ответа от API"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 422:
                st.error(f"Ошибка формата данных (422): {response.text}")
            elif response.status_code == 500:
                st.error(f"Ошибка сервера (500): {response.text}")
            else:
                st.error(f"HTTP ошибка {response.status_code}: {str(e)}")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка API: {str(e)}")
            return None
        except json.JSONDecodeError:
            st.error("Ошибка декодирования ответа от API")
            return None

    def check_health(self) -> bool:
        """Проверка доступности API"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_all_phytoliths(self) -> Optional[List[Dict]]:
        """Получение всех фитолитов с их ID"""
        response = self.session.get(f"{self.base_url}/phytoliths/")
        return self._handle_response(response)

    def add_phytolith(self, data: Dict) -> Optional[int]:
        """Добавление одного фитолита"""
        required_fields = [
            'area', 'convex_area', 'perimeter', 'convex_perimeter',
            'length_feret', 'fiber_length', 'width', 'equivalent_diameter',
            'inscribed_radius', 'form_factor', 'roundness', 'convexity',
            'solidity', 'compactness', 'aspect_ratio', 'elongation',
            'curl', 'l1', 'l2'
        ]

        for field in required_fields:
            if field not in data:
                data[field] = 0.0

        response = self.session.post(
            f"{self.base_url}/phytoliths/",
            json=data
        )
        result = self._handle_response(response)
        return result.get('id') if result else None

    def add_batch_from_df(self, df: pd.DataFrame) -> Optional[int]:
        """Добавление нескольких фитолитов из DataFrame"""
        try:
            df = df.dropna(how='all')

            column_mapping = {
                'Area u^2': 'area',
                'Convex Area u^2': 'convex_area',
                'Perimeter u': 'perimeter',
                'Convex Perimeter u': 'convex_perimeter',
                'Length (Feret) u': 'length_feret',
                'Fiber Length u': 'fiber_length',
                'Width u': 'width',
                'Equivalent Diameter (ArEquivD) u': 'equivalent_diameter',
                'Inscribed Radius (MinR) u': 'inscribed_radius',
                'Form Factor (Circ)': 'form_factor',
                'Roundness': 'roundness',
                'Convexity': 'convexity',
                'Solidity': 'solidity',
                'Compactness': 'compactness',
                'Aspect Ratio': 'aspect_ratio',
                'Elongation': 'elongation',
                'Curl': 'curl',
                'L1': 'l1',
                'L2': 'l2',
                'Subtribe': 'subtribe',
                'Tribe': 'subtribe',
                'subtribe': 'subtribe'
            }

            df_renamed = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

            records = df_renamed.to_dict('records')

            required_fields = [
                'area', 'convex_area', 'perimeter', 'convex_perimeter',
                'length_feret', 'fiber_length', 'width', 'equivalent_diameter',
                'inscribed_radius', 'form_factor', 'roundness', 'convexity',
                'solidity', 'compactness', 'aspect_ratio', 'elongation',
                'curl', 'l1', 'l2'
            ]

            processed_records = []
            for record in records:
                processed_record = {}
                for key, value in record.items():
                    if key in required_fields:
                        if pd.isna(value) or value == '#ДЕЛ/0!' or value == '':
                            processed_record[key] = 0.0
                        else:
                            try:
                                if isinstance(value, str):
                                    value = value.replace(',', '.')
                                processed_record[key] = float(value)
                            except (ValueError, TypeError):
                                processed_record[key] = 0.0
                    else:
                        if pd.isna(value):
                            processed_record[key] = None
                        else:
                            processed_record[key] = str(value) if value else None

                for field in required_fields:
                    if field not in processed_record:
                        processed_record[field] = 0.0

                if any(processed_record.get(field, 0) != 0 for field in required_fields):
                    processed_records.append(processed_record)

            if not processed_records:
                st.warning("Нет валидных данных для загрузки")
                return 0

            response = self.session.post(
                f"{self.base_url}/phytoliths/batch/",
                json=processed_records
            )
            result = self._handle_response(response)
            return result.get('count') if result else None

        except Exception as e:
            st.error(f"Ошибка при подготовке данных: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def update_clusters(self, clusters_dict: Dict[int, int]) -> bool:
        """Обновление кластеров"""
        response = self.session.post(
            f"{self.base_url}/phytoliths/update_clusters/",
            json=clusters_dict
        )
        result = self._handle_response(response)
        return result.get('success', False) if result else False

    def clear_table(self) -> bool:
        """Очистка таблицы"""
        response = self.session.delete(f"{self.base_url}/phytoliths/")
        result = self._handle_response(response)
        return result.get('success', False) if result else False

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Получение всех данных в виде DataFrame"""
        response = self.session.get(f"{self.base_url}/phytoliths/dataframe/")
        result = self._handle_response(response)
        if result and 'data' in result:
            return pd.DataFrame(result['data'])
        return pd.DataFrame()

    # Методы для работы с моделью

    def train_model(self, X, y) -> Optional[Dict]:
        """Обучение модели"""
        try:
            data = {
                'X': X.values.tolist() if hasattr(X, 'values') else X,
                'y': y.values.tolist() if hasattr(y, 'values') else y,
                'feature_names': list(X.columns) if hasattr(X, 'columns') else None
            }

            response = self.session.post(
                f"{self.base_url}/model/train/",
                json=data
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Ошибка при обучении модели: {str(e)}")
            return None

    def predict(self, data: List[Dict]) -> Optional[Dict]:
        """Предсказание для одного или нескольких образцов"""
        try:
            response = self.session.post(
                f"{self.base_url}/model/predict/",
                json=data
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Ошибка при предсказании: {str(e)}")
            return None

    def cluster(self, data: List[Dict], n_clusters: int = 3) -> Optional[Dict]:
        """Кластеризация данных через API"""
        try:
            response = self.session.post(
                f"{self.base_url}/model/cluster/?n_clusters={n_clusters}",
                json=data
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Ошибка при кластеризации: {str(e)}")
            return None

    def get_model_status(self) -> Optional[Dict]:
        """Получение статуса модели"""
        response = self.session.get(f"{self.base_url}/model/status/")
        return self._handle_response(response)

    def get_model_visualizations(self) -> Optional[Dict]:
        """Получение визуализаций модели"""
        response = self.session.get(f"{self.base_url}/model/visualizations/")
        return self._handle_response(response)

    def load_models(self) -> bool:
        """Загрузка моделей на сервере"""
        response = self.session.post(f"{self.base_url}/model/load/")
        result = self._handle_response(response)
        return result.get('success', False) if result else False