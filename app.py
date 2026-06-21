import streamlit as st
import pandas as pd
import numpy as np
from api_client import APIClient
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64

st.set_page_config(
    page_title="Phytolith Classification",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация API клиента
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient("http://localhost:8000")

# Инициализация состояния для хранения результатов
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = {
        'clusters': None,
        'X_pca': None,
        'n_clusters': 3,
        'valid_indices': None,
        'X': None,
        'pca': None
    }

if 'model_status' not in st.session_state:
    st.session_state.model_status = {
        'is_trained': False,
        'accuracy': 0,
        'classes': []
    }

if 'visualizations' not in st.session_state:
    st.session_state.visualizations = {}

api_client = st.session_state.api_client


def check_api_connection():
    """Проверка подключения к API"""
    if not api_client.check_health():
        st.error("❌ Нет подключения к API. Убедитесь, что сервер запущен на http://localhost:8000")
        st.info("Запустите API: uvicorn api:app --reload")
        return False
    return True


def update_model_status():
    """Обновление статуса модели"""
    status = api_client.get_model_status()
    if status:
        st.session_state.model_status = status
    return st.session_state.model_status


def main():
    if not check_api_connection():
        return

    st.title("🌿 Разработка веб-приложения для хранения данных и автоматической классификации фитолитов")

    update_model_status()

    menu = ["Главная", "Загрузка данных", "Классификация", "Визуализация моделей",
            "Ручной ввод", "Кластеризация", "Просмотр данных"]
    choice = st.sidebar.selectbox("Навигация", menu)

    if choice == "Главная":
        show_home()
    elif choice == "Загрузка данных":
        upload_data()
    elif choice == "Классификация":
        classification()
    elif choice == "Визуализация моделей":
        model_visualization()
    elif choice == "Ручной ввод":
        manual_classification()
    elif choice == "Кластеризация":
        clustering()
    elif choice == "Просмотр данных":
        view_data()


def show_home():
    st.header("О проекте")
    st.write("""
    **Веб-приложение для хранения данных и автоматической классификации фитолитов** 
    на основе методов машинного обучения.

    ### Основные функции:
    - **Загрузка CSV файлов** с данными фитолитов
    - **Обучение моделей** классификации (случайный лес) и кластеризации
    - **Классификация новых фитолитов** через ручной ввод или пакетную обработку
    - **Визуализация результатов** обучения моделей
    - **Хранение данных** в базе данных SQLite
    - **Аналитические графики** для исследовательской работы
    """)

    st.subheader("🔄 Статус модели")

    if st.session_state.model_status['is_trained']:
        st.success("✅ Модель классификации загружена и готова к использованию")
        classes = st.session_state.model_status['classes']
        if classes:
            st.info(f"**Доступные классы:** {', '.join(classes[:5])}..." if len(
                classes) > 5 else f"**Доступные классы:** {', '.join(classes)}")

        if st.session_state.model_status['accuracy'] > 0:
            st.metric("Точность модели", f"{st.session_state.model_status['accuracy']:.2%}")
    else:
        st.warning("⚠️ Модель не обучена. Загрузите данные и обучите модель в разделе 'Классификация'")

    if st.session_state.clustering_results['clusters'] is not None:
        st.success(
            f"✅ Результаты кластеризации доступны (кластеров: {st.session_state.clustering_results['n_clusters']})")
    else:
        st.info("ℹ️ Кластеризация не выполнялась. Перейдите в раздел 'Кластеризация' для анализа")


def upload_data():
    st.header("Загрузка данных")

    if st.button("Очистить базу данных"):
        if api_client.clear_table():
            st.session_state.clustering_results = {
                'clusters': None,
                'X_pca': None,
                'n_clusters': 3,
                'valid_indices': None,
                'X': None,
                'pca': None
            }
            st.success("База данных очищена")
        else:
            st.error("Ошибка при очистке базы данных")

    uploaded_file = st.file_uploader("Загрузите CSV файл с данными фитолитов", type=['csv'])

    if uploaded_file is not None:
        try:
            encodings = ['utf-8', 'cp1251', 'latin1', 'iso-8859-1', 'cp1252']
            separators = [';', ',', '\t']
            df = None
            used_encoding = None
            used_sep = None

            for encoding in encodings:
                for sep in separators:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep=sep, encoding=encoding)
                        if len(df.columns) > 1 or (len(df.columns) == 1 and df.iloc[:, 0].str.contains(sep).any()):
                            used_encoding = encoding
                            used_sep = sep
                            break
                    except:
                        continue
                if df is not None:
                    break

            if df is None:
                st.error("Не удалось прочитать файл. Попробуйте сохранить его в формате CSV с разделителем ';' или ','")
                return

            df = df.dropna(how='all')

            st.success(f"Файл успешно прочитан (кодировка: {used_encoding}, разделитель: '{used_sep}')")

            st.write("**Загруженный файл:**")
            st.write(f"Размер: {len(df)} строк, {len(df.columns)} столбцов")
            st.write("**Столбцы в файле:**")
            st.write(list(df.columns))
            st.write("**Предпросмотр данных (первые 5 строк):**")
            st.dataframe(df.head())

            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                st.warning("В данных есть пропущенные значения:")
                st.write(missing_data[missing_data > 0])
            else:
                st.success("Пропущенных значений нет")

            required_columns = [
                'Area u^2', 'Convex Area u^2', 'Perimeter u', 'Convex Perimeter u',
                'Length (Feret) u', 'Fiber Length u', 'Width u',
                'Equivalent Diameter (ArEquivD) u', 'Inscribed Radius (MinR) u',
                'Form Factor (Circ)', 'Roundness', 'Convexity', 'Solidity',
                'Compactness', 'Aspect Ratio', 'Elongation', 'Curl', 'L1', 'L2'
            ]

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"Отсутствуют обязательные столбцы: {missing_cols}")
                st.info("Ожидаемые столбцы:")
                st.write(required_columns)
                return

            st.success("Все обязательные столбцы присутствуют")

            taxon_columns = ['Subtribe', 'Tribe', 'Family', 'subtribe', 'tribe', 'family']
            found_taxon_col = None

            for col in taxon_columns:
                if col in df.columns:
                    found_taxon_col = col
                    break

            if found_taxon_col is None:
                st.warning(
                    "Столбец с таксономической информацией (Subtribe/Tribe/Family) не найден. "
                    "Для обучения модели классификации нужен столбец с названиями фитолитов."
                )

                col_option = st.selectbox("Выберите столбец с названиями фитолитов",
                                          ['Нет такого столбца'] + list(df.columns))

                if col_option != 'Нет такого столбца':
                    # Переименовываем выбранный столбец в 'Subtribe'
                    df = df.rename(columns={col_option: 'Subtribe'})
                    st.success(f"Столбец '{col_option}' переименован в 'Subtribe'")
            else:
                st.success(f"Найден столбец с таксономической информацией: '{found_taxon_col}'")

            if st.button("Сохранить в базу данных", key="save_button"):
                with st.spinner("Сохранение данных..."):
                    try:
                        count = api_client.add_batch_from_df(df)
                        if count:
                            st.success(f"✅ Успешно сохранено {count} записей в базу данных")

                            total_df = api_client.get_dataframe()
                            if total_df is not None and len(total_df) > 0:
                                st.info(f"Всего записей в базе данных: {len(total_df)}")

                            st.session_state.clustering_results = {
                                'clusters': None,
                                'X_pca': None,
                                'n_clusters': 3,
                                'valid_indices': None,
                                'X': None,
                                'pca': None
                            }
                        else:
                            st.error("❌ Не удалось сохранить данные. Проверьте логи сервера.")
                    except Exception as e:
                        st.error(f"❌ Ошибка при сохранении в базу данных: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())

        except Exception as e:
            st.error(f"❌ Ошибка при чтении файла: {str(e)}")
            import traceback
            st.error(traceback.format_exc())


def classification():
    st.header("Классификация фитолитов")

    df = api_client.get_dataframe()

    if df is None or len(df) == 0:
        st.warning("В базе данных нет данных. Загрузите данные в разделе 'Загрузка данных'")
        return

    # Подготовка данных
    from ml_model import MLModel
    temp_model = MLModel()
    X, y, valid_indices = temp_model.prepare_data(df)

    if y is None or len(y) == 0:
        st.warning("В данных отсутствует столбец 'subtribe' или недостаточно данных для обучения модели классификации")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Обучение модели")

        st.info(f"""
        **Информация о данных:**
        - Всего записей: {len(df)}
        - Записей с полными данными: {len(X)}
        - Уникальных классов: {y.nunique()}
        - Примеры классов: {', '.join(y.unique()[:3]) if len(y.unique()) > 3 else ', '.join(y.unique())}
        """)

        if st.button("Обучить модель классификации", type="primary", key="train_button"):
            with st.spinner("Обучение модели..."):
                train_data = {
                    'X': X.values.tolist(),
                    'y': y.values.tolist(),
                    'feature_names': list(X.columns)
                }

                result = api_client.train_model(X, y)

                if result and result.get('success'):
                    update_model_status()

                    st.success(f"✅ Модель обучена с точностью: **{result['accuracy']:.2%}**")
                    st.info("Модель сохранена на сервере")

                    st.subheader("Результаты обучения")
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    with col_metric1:
                        st.metric("Точность", f"{result['accuracy']:.2%}")
                    with col_metric2:
                        st.metric("Записей для обучения", f"{result['n_samples']}")
                    with col_metric3:
                        st.metric("Классов", f"{result['n_classes']}")

                    if st.button("Посмотреть детальные графики обучения", key="show_plots"):
                        st.session_state.current_page = "Визуализация моделей"
                        st.rerun()
                else:
                    st.error("❌ Не удалось обучить модель. Проверьте данные.")

    with col2:
        st.subheader("⚡ Быстрая классификация")
        st.write("Для детального ручного ввода перейдите в раздел **'Ручной ввод'**")

        st.write(f"**Статус модели:** {'✅ Обучена' if st.session_state.model_status['is_trained'] else '❌ Не обучена'}")

        with st.form("quick_test"):
            st.write("**Тестовая классификация:**")

            if len(X) > 0:
                sample_data = X.iloc[0].to_dict()
                st.write(f"Используются параметры первой записи из базы")
            else:
                sample_data = {f: 0.0 for f in temp_model.features}

            submitted = st.form_submit_button("🧪 Протестировать на примере")

            if submitted:
                if st.session_state.model_status['is_trained']:
                    input_data = [sample_data]
                    result = api_client.predict(input_data)

                    if result and result.get('success'):
                        prediction = result['predictions'][0]
                        st.success(f"✅ Тестовый результат: **{prediction}**")

                        if result.get('probabilities'):
                            classes = result['classes']
                            proba = result['probabilities'][0]
                            prob_df = pd.DataFrame({
                                'Класс': classes,
                                'Вероятность': proba
                            }).sort_values('Вероятность', ascending=False)

                            st.write("**Вероятности по классам:**")
                            st.dataframe(prob_df.head(5))
                    else:
                        st.error("❌ Не удалось выполнить классификацию")
                else:
                    st.error("❌ Модель не обучена. Обучите модель сначала.")


def model_visualization():
    st.header("Визуализация моделей")

    if not st.session_state.model_status['is_trained']:
        st.warning("⚠️ Модель не обучена. Сначала обучите модель в разделе 'Классификация'")
        return

    st.success("✅ Модель загружена и готова для визуализации")

    if st.session_state.model_status['accuracy'] > 0:
        st.metric("Точность модели", f"{st.session_state.model_status['accuracy']:.2%}")

    if st.button("Обновить визуализации", key="refresh_viz"):
        with st.spinner("Загрузка визуализаций..."):
            viz_data = api_client.get_model_visualizations()
            if viz_data and viz_data.get('success'):
                st.session_state.visualizations = viz_data
                st.success("Визуализации загружены")
            else:
                st.error("Не удалось загрузить визуализации")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Матрица ошибок",
        "Важность признаков",
        "Отчет классификации",
        "Распределение классов",
        "Кластеризация"
    ])

    with tab1:
        st.subheader("Матрица ошибок (Confusion Matrix)")
        if 'confusion_matrix' in st.session_state.visualizations:
            img_data = base64.b64decode(st.session_state.visualizations['confusion_matrix'])
            img = Image.open(io.BytesIO(img_data))
            st.image(img, use_column_width=True)
            st.caption("""
            **Как читать матрицу ошибок:**
            - Диагональ (слева направо вниз): правильные предсказания
            - Вне диагонали: ошибки классификации
            - Чем светлее цвет на диагонали, тем лучше модель распознает класс
            """)
        else:
            st.info("Нет данных для построения матрицы ошибок. Нажмите 'Обновить визуализации'")

    with tab2:
        st.subheader("Важность признаков (Feature Importance)")
        if 'feature_importance' in st.session_state.visualizations:
            img_data = base64.b64decode(st.session_state.visualizations['feature_importance'])
            img = Image.open(io.BytesIO(img_data))
            st.image(img, use_column_width=True)
            st.caption("""
            **Важность признаков показывает:**
            - Какие характеристики фитолитов наиболее важны для классификации
            - Признаки с наибольшей важностью сильнее влияют на решение модели
            """)
        else:
            st.info("Нет данных о важности признаков. Нажмите 'Обновить визуализации'")

    with tab3:
        st.subheader("Отчет классификации")
        if 'classification_report' in st.session_state.visualizations:
            report = st.session_state.visualizations['classification_report']
            if report:
                report_df = pd.DataFrame(report).transpose()

                if 'accuracy' in report_df.index:
                    accuracy_value = report_df.loc['accuracy', report_df.columns[0]]
                    report_df = report_df.drop('accuracy')

                st.dataframe(report_df)

                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'accuracy' in st.session_state.visualizations:
                        st.metric("Точность (Accuracy)", f"{st.session_state.visualizations['accuracy']:.2%}")
                with col2:
                    if 'weighted avg' in report and 'precision' in report['weighted avg']:
                        st.metric("Средняя точность (Precision)", f"{report['weighted avg']['precision']:.2%}")
                with col3:
                    if 'weighted avg' in report and 'recall' in report['weighted avg']:
                        st.metric("Средняя полнота (Recall)", f"{report['weighted avg']['recall']:.2%}")
        else:
            st.info("Нет данных для отчета классификации. Нажмите 'Обновить визуализации'")

    with tab4:
        st.subheader("Распределение классов")
        if 'class_distribution' in st.session_state.visualizations:
            img_data = base64.b64decode(st.session_state.visualizations['class_distribution'])
            img = Image.open(io.BytesIO(img_data))
            st.image(img, use_column_width=True)
            st.caption("""
            **Сравнение распределения классов:**
            - Левый график: реальное распределение классов в тестовой выборке
            - Правый график: распределение предсказанных классов
            """)
        else:
            st.info("Нет данных для построения распределения классов. Нажмите 'Обновить визуализации'")

    with tab5:
        st.subheader("🔬 Визуализация кластеризации")

        df = api_client.get_dataframe()

        if df is None or len(df) == 0:
            st.warning("В базе данных нет данных для кластеризации.")
            return

        from ml_model import MLModel
        temp_model = MLModel()
        X, _, valid_indices = temp_model.prepare_data(df)

        if len(X) == 0:
            st.error("Нет данных для кластеризации. Проверьте, что все признаки заполнены.")
            return

        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Параметры")
            n_clusters = st.slider("Количество кластеров", min_value=2, max_value=10, value=3, key="viz_clusters")

            if st.button("Выполнить кластеризацию", key="viz_cluster_button"):
                with st.spinner("Выполняется кластеризация..."):
                    input_data = X.to_dict('records')
                    result = api_client.cluster(input_data, n_clusters=n_clusters)

                    if result and result.get('success'):
                        # Используем данные, которые вернул API
                        clusters = result['clusters']
                        X_pca = result['X_pca']  # Берём из ответа API
                        explained_variance = result.get('explained_variance', [0.5, 0.3])

                        # Создаём объект PCA для хранения в сессии
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2)
                        pca.explained_variance_ratio_ = explained_variance  # Сохраняем для подписей

                        st.session_state.clustering_results = {
                            'clusters': clusters,
                            'X_pca': X_pca,  # Используем из API
                            'n_clusters': n_clusters,
                            'valid_indices': valid_indices,
                            'X': X,
                            'pca': pca,
                            'explained_variance': explained_variance  # Сохраняем для подписей
                        }

                        st.success(f"✅ Кластеризация выполнена. Кластеров: {n_clusters}")

        with col2:
            st.subheader("Результаты")

            if st.session_state.clustering_results['clusters'] is not None:
                results = st.session_state.clustering_results
                clusters = results['clusters']
                X_pca = results['X_pca']
                n_clusters = results['n_clusters']
                pca = results['pca']

                cluster_stats = pd.Series(clusters).value_counts().sort_index()

                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Всего объектов", len(clusters))
                with col_stat2:
                    st.metric("Кластеров", n_clusters)
                with col_stat3:
                    st.metric("Min/Max в кластере", f"{cluster_stats.min()}/{cluster_stats.max()}")

                fig = go.Figure()

                for cluster_id in range(n_clusters):
                    cluster_mask = [c == cluster_id for c in clusters]
                    fig.add_trace(go.Scatter(
                        x=X_pca[cluster_mask, 0],
                        y=X_pca[cluster_mask, 1],
                        mode='markers',
                        name=f'Кластер {cluster_id}',
                        marker=dict(size=8, opacity=0.7),
                        text=[f'Кластер: {cluster_id}<br>Объект {i}' for i in range(len(X_pca[cluster_mask]))],
                        hoverinfo='text'
                    ))

                fig.update_layout(
                    title=f'Визуализация кластеризации (PCA) - {n_clusters} кластеров',
                    xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                    yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                    hovermode='closest',
                    width=800,
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                cluster_df = pd.DataFrame({
                    'ID объекта': range(len(clusters)),
                    'Кластер': clusters,
                    'PC1': X_pca[:, 0],
                    'PC2': X_pca[:, 1]
                })

                if 'subtribe' in df.columns and valid_indices is not None:
                    cluster_df['Subtribe'] = df.loc[valid_indices, 'subtribe'].values

                st.dataframe(cluster_df)

                if st.button("💾 Сохранить кластеры в базу данных", key="save_clusters_viz"):
                    try:
                        all_records = api_client.get_all_phytoliths()
                        if all_records:
                            full_clusters = [-1] * len(all_records)
                            for idx, cluster_id in zip(valid_indices, clusters):
                                if idx < len(full_clusters):
                                    full_clusters[idx] = int(cluster_id)

                            clusters_dict = {i: full_clusters[i] for i in range(len(full_clusters))}
                            success = api_client.update_clusters(clusters_dict)

                            if success:
                                st.success(f"✅ Кластеры сохранены в базу данных. Записей с кластерами: {len(clusters)}")
                    except Exception as e:
                        st.error(f"❌ Ошибка при сохранении: {str(e)}")
            else:
                st.info("ℹ️ Нажмите 'Выполнить кластеризацию' для анализа данных")


def manual_classification():
    st.header("✍️ Ручной ввод параметров фитолита")

    if not st.session_state.model_status['is_trained']:
        st.error("❌ Модель не обучена!")
        st.info("Пожалуйста, сначала обучите модель в разделе **'Классификация'**")
        return

    st.success("✅ Модель готова к классификации")

    classes = st.session_state.model_status['classes']
    if classes:
        st.info(f"**Модель обучена на {len(classes)} классах:** {', '.join(classes[:10])}" +
                ("..." if len(classes) > 10 else ""))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📏 Геометрические параметры")

        area = st.number_input("**Площадь (Area u²)**",
                               min_value=0.0, max_value=10000.0,
                               value=610.585, step=10.0,
                               help="Общая площадь фитолита в квадратных единицах")

        convex_area = st.number_input("**Выпуклая площадь (Convex Area u²)**",
                                      min_value=0.0, max_value=10000.0,
                                      value=723.765, step=10.0)

        perimeter = st.number_input("**Периметр (Perimeter u)**",
                                    min_value=0.0, max_value=1000.0,
                                    value=155.650, step=1.0)

        convex_perimeter = st.number_input("**Выпуклый периметр (Convex Perimeter u)**",
                                           min_value=0.0, max_value=1000.0,
                                           value=141.335, step=1.0)

        length_feret = st.number_input("**Длина (Length (Feret) u)**",
                                       min_value=0.0, max_value=500.0,
                                       value=63.584, step=1.0)

        fiber_length = st.number_input("**Длина средней линии (Fiber Length u)**",
                                       min_value=0.0, max_value=500.0,
                                       value=68.083, step=1.0)

        width = st.number_input("**Ширина (Width u)**",
                                min_value=0.0, max_value=100.0,
                                value=12.151, step=0.5)

        equivalent_diameter = st.number_input("**Эквивалентный диаметр (Equivalent Diameter u)**",
                                              min_value=0.0, max_value=100.0,
                                              value=27.882, step=0.5)

        inscribed_radius = st.number_input("**Вписанный радиус (Inscribed Radius u)**",
                                           min_value=0.0, max_value=50.0,
                                           value=4.284, step=0.1)

    with col2:
        st.subheader("📐 Форма и соотношения")

        form_factor = st.number_input("**Фактор формы (Form Factor)**",
                                      min_value=0.0, max_value=1.0,
                                      value=0.317, step=0.01,
                                      help="Мера близости формы к кругу (1 = идеальный круг)")

        roundness = st.number_input("**Закругленность (Roundness)**",
                                    min_value=0.0, max_value=1.0,
                                    value=0.192, step=0.01)

        convexity = st.number_input("**Выпуклость (Convexity)**",
                                    min_value=0.0, max_value=1.0,
                                    value=0.908, step=0.01)

        solidity = st.number_input("**Плотность (Solidity)**",
                                   min_value=0.0, max_value=1.0,
                                   value=0.844, step=0.01)

        compactness = st.number_input("**Компактность (Compactness)**",
                                      min_value=0.0, max_value=1.0,
                                      value=0.439, step=0.01)

        aspect_ratio = st.number_input("**Соотношение сторон (Aspect Ratio)**",
                                       min_value=0.0, max_value=20.0,
                                       value=5.233, step=0.1)

        elongation = st.number_input("**Протяженность (Elongation)**",
                                     min_value=0.0, max_value=20.0,
                                     value=5.603, step=0.1)

        curl = st.number_input("**Извилистость (Curl)**",
                               min_value=0.0, max_value=1.0,
                               value=0.934, step=0.01)

        l1 = st.number_input("**Полуширина между вписанными окружностями**",
                             min_value=0.0, max_value=10.0,
                             value=1.7915, step=0.1)

        l2 = st.number_input("**Коэффициент размещения окружностей по ширине**",
                             min_value=0.0, max_value=10.0,
                             value=1.418184, step=0.1)

    if st.button("🔍 Классифицировать фитолит", type="primary", use_container_width=True, key="classify_button"):
        with st.spinner("Выполняется классификация..."):
            input_data = {
                'area': area,
                'convex_area': convex_area,
                'perimeter': perimeter,
                'convex_perimeter': convex_perimeter,
                'length_feret': length_feret,
                'fiber_length': fiber_length,
                'width': width,
                'equivalent_diameter': equivalent_diameter,
                'inscribed_radius': inscribed_radius,
                'form_factor': form_factor,
                'roundness': roundness,
                'convexity': convexity,
                'solidity': solidity,
                'compactness': compactness,
                'aspect_ratio': aspect_ratio,
                'elongation': elongation,
                'curl': curl,
                'l1': l1,
                'l2': l2
            }

            result = api_client.predict([input_data])

            if result and result.get('success'):
                predicted_class = result['predictions'][0]

                st.success(f"### Результат классификации: **{predicted_class}**")

                col_result1, col_result2 = st.columns(2)

                with col_result1:
                    st.subheader("Вероятности по классам")

                    if result.get('probabilities'):
                        classes = result['classes']
                        proba = result['probabilities'][0]
                        prob_df = pd.DataFrame({
                            'Класс': classes,
                            'Вероятность, %': (np.array(proba) * 100)
                        }).sort_values('Вероятность, %', ascending=False)

                        st.dataframe(prob_df.style.format({'Вероятность, %': '{:.2f}'}))

                with col_result2:
                    st.subheader("Визуализация вероятностей")

                    if result.get('probabilities'):
                        classes = result['classes']
                        proba = result['probabilities'][0]

                        fig, ax = plt.subplots(figsize=(10, 6))
                        prob_df_local = pd.DataFrame({
                            'Класс': classes,
                            'Вероятность, %': (np.array(proba) * 100)
                        }).sort_values('Вероятность, %', ascending=False)

                        top_n = min(10, len(prob_df_local))
                        top_probs = prob_df_local.head(top_n)

                        bars = ax.barh(range(top_n), top_probs['Вероятность, %'])
                        ax.set_yticks(range(top_n))
                        ax.set_yticklabels(top_probs['Класс'])
                        ax.set_xlabel('Вероятность, %')
                        ax.set_title(f'Топ-{top_n} наиболее вероятных классов')
                        ax.invert_yaxis()

                        if predicted_class in top_probs['Класс'].values:
                            idx = top_probs[top_probs['Класс'] == predicted_class].index[0]
                            bars[idx].set_color('green')

                        plt.tight_layout()
                        st.pyplot(fig)

                st.subheader("ℹ️ Интерпретация результата")

                if result.get('probabilities'):
                    max_prob = max(result['probabilities'][0]) * 100
                    if max_prob > 80:
                        st.success(f"Высокая уверенность модели: {max_prob:.1f}%")
                    elif max_prob > 60:
                        st.warning(f"Средняя уверенность модели: {max_prob:.1f}%")
                    else:
                        st.error(f"Низкая уверенность модели: {max_prob:.1f}%")
                        st.info("Рекомендуется проверить введенные параметры или добавить больше обучающих данных")

                if st.button("💾 Сохранить результат в базу данных", key="save_button"):
                    try:
                        input_data['subtribe'] = predicted_class
                        phytolith_id = api_client.add_phytolith(input_data)
                        if phytolith_id:
                            st.success("✅ Результат сохранен в базу данных!")
                    except Exception as e:
                        st.error(f"❌ Ошибка при сохранении: {str(e)}")
            else:
                st.error("❌ Не удалось выполнить классификацию")


def clustering():
    st.header("Кластеризация фитолитов")

    df = api_client.get_dataframe()

    if df is None or len(df) == 0:
        st.warning("В базе данных нет данных.")
        return

    from ml_model import MLModel
    temp_model = MLModel()
    X, y, valid_indices = temp_model.prepare_data(df)

    if len(X) == 0:
        st.error("Нет данных для кластеризации. Проверьте, что все признаки заполнены.")
        return

    st.subheader("Параметры кластеризации")
    n_clusters = st.slider("Количество кластеров", min_value=2, max_value=10, value=3, key="cluster_main")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Выполнить кластеризацию", type="primary", key="cluster_button"):
            with st.spinner("Выполняется кластеризация..."):
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.cluster import AgglomerativeClustering
                    from sklearn.decomposition import PCA

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                    clusters = clusterer.fit_predict(X_scaled)

                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)

                    st.session_state.clustering_results = {
                        'clusters': clusters,
                        'X_pca': X_pca,
                        'n_clusters': n_clusters,
                        'valid_indices': valid_indices,
                        'X': X,
                        'pca': pca,
                        'scaler': scaler  # сохраняем обученный scaler
                    }

                    st.success(f"✅ Кластеризация выполнена. Кластеров: {n_clusters}")

                except Exception as e:
                    st.error(f"❌ Ошибка при кластеризации: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

    with col2:
        st.subheader("Информация")
        st.info(f"""
        **Данные для кластеризации:**
        - Всего записей: {len(df)}
        - Записей с полными данными: {len(X)}
        - Записей с пропусками: {len(df) - len(X)}
        """)

        if st.session_state.clustering_results['clusters'] is not None:
            st.success(
                f"✅ Результаты кластеризации доступны (кластеров: {st.session_state.clustering_results['n_clusters']})")

    if st.session_state.clustering_results['clusters'] is not None:
        results = st.session_state.clustering_results
        clusters = results['clusters']
        X_pca = results['X_pca']
        n_clusters = results['n_clusters']
        pca = results['pca']

        st.subheader("Результаты кластеризации")

        cluster_stats = pd.Series(clusters).value_counts().sort_index()

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Всего объектов", len(clusters))
        with col_stat2:
            st.metric("Кластеров", n_clusters)
        with col_stat3:
            st.metric("Мин. в кластере", cluster_stats.min())
        with col_stat4:
            st.metric("Макс. в кластере", cluster_stats.max())

        st.subheader("Визуализация кластеризации (PCA)")

        fig = go.Figure()

        for cluster_id in range(n_clusters):
            cluster_mask = [c == cluster_id for c in clusters]
            fig.add_trace(go.Scatter(
                x=X_pca[cluster_mask, 0],
                y=X_pca[cluster_mask, 1],
                mode='markers',
                name=f'Кластер {cluster_id}',
                marker=dict(size=8, opacity=0.7),
                text=[f'Кластер: {cluster_id}<br>Объект {i}' for i in range(len(X_pca[cluster_mask]))],
                hoverinfo='text'
            ))

        fig.update_layout(
            title=f'Визуализация кластеризации (PCA) - {n_clusters} кластеров',
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
            hovermode='closest',
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Детальная информация по кластерам")

        cluster_df = pd.DataFrame({
            'ID объекта': range(len(clusters)),
            'Кластер': clusters,
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1]
        })

        if 'subtribe' in df.columns and valid_indices is not None:
            cluster_df['Subtribe'] = df.loc[valid_indices, 'subtribe'].values

        st.dataframe(cluster_df, use_container_width=True)

        col_save1, col_save2 = st.columns(2)

        with col_save1:
            if st.button("💾 Сохранить кластеры в базу данных", type="primary", use_container_width=True):
                try:
                    all_records = api_client.get_all_phytoliths()
                    if all_records:
                        clusters_dict = {}

                        for idx, cluster_id in zip(valid_indices, clusters):
                            if idx < len(all_records):
                                # Находим ID записи в базе
                                record_id = all_records[idx]['id']
                                clusters_dict[record_id] = int(cluster_id)

                        success = api_client.update_clusters(clusters_dict)

                        if success:
                            st.success(f"✅ Кластеры сохранены в базу данных. Записей с кластерами: {len(clusters)}")
                        else:
                            st.error("❌ Ошибка при сохранении кластеров")
                except Exception as e:
                    st.error(f"❌ Ошибка при сохранении: {str(e)}")

        with col_save2:
            csv = cluster_df.to_csv(index=False)
            st.download_button(
                label="📥 Скачать результаты (CSV)",
                data=csv,
                file_name=f"clustering_results_{n_clusters}_clusters.csv",
                mime="text/csv",
                use_container_width=True
            )

        if 'Subtribe' in cluster_df.columns:
            st.subheader("Распределение Subtribe по кластерам")

            pivot_df = pd.crosstab(cluster_df['Subtribe'], cluster_df['Кластер'])
            st.dataframe(pivot_df, use_container_width=True)

            fig_dist = px.bar(
                pivot_df.reset_index().melt(id_vars=['Subtribe'], var_name='Кластер', value_name='Количество'),
                x='Subtribe',
                y='Количество',
                color='Кластер',
                title='Распределение Subtribe по кластерам',
                barmode='group'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("ℹ️ Нажмите 'Выполнить кластеризацию' для анализа данных")


def view_data():
    st.header("Просмотр данных")

    df = api_client.get_dataframe()

    if df is None or len(df) == 0:
        st.warning("В базе данных нет данных.")
        return

    st.write(f"Всего записей: {len(df)}")

    st.subheader("Фильтрация данных")
    col1, col2 = st.columns(2)

    with col1:
        if 'subtribe' in df.columns:
            subtribes = df['subtribe'].unique()
            selected_subtribe = st.selectbox("Выберите Subtribe", ['Все'] + list(subtribes))
            if selected_subtribe != 'Все':
                df = df[df['subtribe'] == selected_subtribe]

    with col2:
        if 'cluster' in df.columns:
            clusters = df['cluster'].unique()
            selected_cluster = st.selectbox("Выберите Кластер", ['Все'] + list(clusters))
            if selected_cluster != 'Все':
                df = df[df['cluster'] == selected_cluster]

    st.dataframe(df, use_container_width=True)

    if st.button("Экспортировать в CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Скачать CSV",
            data=csv,
            file_name="phytolith_data.csv",
            mime="text/csv"
        )

    st.subheader("Статистическая информация")
    if len(df) > 0:
        st.write("Основные статистические показатели:")
        from ml_model import MLModel
        temp_model = MLModel()
        st.write(df[temp_model.features].describe())


if __name__ == "__main__":
    main()