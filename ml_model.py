import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


class MLModel:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.clusterer = AgglomerativeClustering(n_clusters=3, linkage='ward')
        self.scaler = StandardScaler()
        self.features = [
            'area', 'convex_area', 'perimeter', 'convex_perimeter',
            'length_feret', 'fiber_length', 'width', 'equivalent_diameter',
            'inscribed_radius', 'form_factor', 'roundness', 'convexity',
            'solidity', 'compactness', 'aspect_ratio', 'elongation',
            'curl', 'l1', 'l2'
        ]
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        self.y_test = None
        self.y_pred = None
        self.classes = None
        self.X_test = None
        self.classification_report_dict = None
        self.accuracy = 0

    def prepare_data(self, df):
        X = df[self.features].copy()

        valid_indices = X.dropna().index
        X_clean = X.loc[valid_indices]

        y = None
        if 'subtribe' in df.columns:
            y = df.loc[valid_indices, 'subtribe']

        return X_clean, y, valid_indices

    def train_classifier(self, X, y):
        if len(X) == 0 or len(y) == 0:
            self.accuracy = 0
            return 0, None, None, None

        if y.nunique() <= 1:
            self.accuracy = 0
            return 0, None, None, None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)

        self.y_test = y_test
        self.y_pred = y_pred
        self.classes = self.classifier.classes_
        self.X_test = X_test

        self.classification_report_dict = classification_report(y_test, y_pred, output_dict=True)

        print(f"Accuracy: {self.accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return self.accuracy, X_test, y_test, y_pred

    def get_confusion_matrix_plot(self):
        """Создает визуализацию матрицы ошибок"""
        if self.y_test is None or self.y_pred is None:
            return None

        cm = confusion_matrix(self.y_test, self.y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes,
                    yticklabels=self.classes,
                    ax=ax)
        ax.set_xlabel('Предсказанный класс')
        ax.set_ylabel('Истинный класс')
        ax.set_title('Матрица ошибок (Confusion Matrix)')
        plt.tight_layout()

        return fig

    def get_feature_importance_plot(self):
        """Создает график важности признаков"""
        if not hasattr(self.classifier, 'feature_importances_'):
            return None

        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(feature_importance)),
                       feature_importance['importance'])
        ax.set_yticks(range(len(feature_importance)))
        ax.set_yticklabels(feature_importance['feature'])
        ax.set_xlabel('Важность признака')
        ax.set_title('Важность признаков (Feature Importance)')
        ax.invert_yaxis()

        for i, (bar, importance) in enumerate(zip(bars, feature_importance['importance'])):
            ax.text(importance + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{importance:.3f}', va='center')

        plt.tight_layout()
        return fig

    def get_class_distribution_plot(self):
        """Создает график распределения классов"""
        if self.y_test is None or self.y_pred is None:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        true_counts = pd.Series(self.y_test).value_counts()
        ax1.bar(range(len(true_counts)), true_counts.values)
        ax1.set_xticks(range(len(true_counts)))
        ax1.set_xticklabels(true_counts.index, rotation=45, ha='right')
        ax1.set_xlabel('Класс')
        ax1.set_ylabel('Количество')
        ax1.set_title('Распределение истинных классов')

        pred_counts = pd.Series(self.y_pred).value_counts()
        ax2.bar(range(len(pred_counts)), pred_counts.values)
        ax2.set_xticks(range(len(pred_counts)))
        ax2.set_xticklabels(pred_counts.index, rotation=45, ha='right')
        ax2.set_xlabel('Класс')
        ax2.set_ylabel('Количество')
        ax2.set_title('Распределение предсказанных классов')

        plt.tight_layout()
        return fig

    def get_classification_report(self):
        """Возвращает отчет классификации"""
        return self.classification_report_dict

    def get_accuracy(self):
        """Возвращает точность модели"""
        return self.accuracy

    def train_clusterer(self, X, n_clusters=3):
        X_scaled = self.scaler.fit_transform(X)
        self.clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        clusters = self.clusterer.fit_predict(X_scaled)
        return clusters

    def predict(self, X):
        if self.is_trained():
            return self.classifier.predict(X)
        return None

    def predict_proba(self, X):
        """Возвращает вероятности принадлежности к классам"""
        if self.is_trained():
            return self.classifier.predict_proba(X)
        return None

    def cluster(self, X, n_clusters=3):
        X_scaled = self.scaler.transform(X)
        temp_clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        return temp_clusterer.fit_predict(X_scaled)

    def save_models(self):
        joblib.dump(self.classifier, f'{self.models_dir}/classifier.pkl')
        joblib.dump(self.clusterer, f'{self.models_dir}/clusterer.pkl')
        joblib.dump(self.scaler, f'{self.models_dir}/scaler.pkl')

        model_data = {
            'y_test': self.y_test,
            'y_pred': self.y_pred,
            'classes': self.classes,
            'accuracy': self.accuracy,
            'classification_report': self.classification_report_dict
        }
        joblib.dump(model_data, f'{self.models_dir}/model_data.pkl')

    def load_models(self):
        try:
            if os.path.exists(f'{self.models_dir}/classifier.pkl'):
                self.classifier = joblib.load(f'{self.models_dir}/classifier.pkl')

                if os.path.exists(f'{self.models_dir}/model_data.pkl'):
                    model_data = joblib.load(f'{self.models_dir}/model_data.pkl')
                    self.y_test = model_data.get('y_test')
                    self.y_pred = model_data.get('y_pred')
                    self.classes = model_data.get('classes')
                    self.accuracy = model_data.get('accuracy', 0)
                    self.classification_report_dict = model_data.get('classification_report')

                if os.path.exists(f'{self.models_dir}/clusterer.pkl'):
                    self.clusterer = joblib.load(f'{self.models_dir}/clusterer.pkl')
                if os.path.exists(f'{self.models_dir}/scaler.pkl'):
                    self.scaler = joblib.load(f'{self.models_dir}/scaler.pkl')

                print(f"✅ Модель загружена. Точность: {self.accuracy:.2%}")
                return True
        except Exception as e:
            print(f"Ошибка при загрузке моделей: {e}")
        return False

    def is_trained(self):
        try:
            return hasattr(self.classifier, 'classes_') and self.classifier.classes_ is not None
        except:
            return False

    def get_class_labels(self):
        """Возвращает список доступных классов"""
        if self.is_trained():
            return list(self.classifier.classes_)
        return []