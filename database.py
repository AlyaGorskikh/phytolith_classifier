from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

Base = declarative_base()


class Phytolith(Base):
    __tablename__ = 'phytoliths'

    id = Column(Integer, primary_key=True)
    area = Column(Float)
    convex_area = Column(Float)
    perimeter = Column(Float)
    convex_perimeter = Column(Float)
    length_feret = Column(Float)
    fiber_length = Column(Float)
    width = Column(Float)
    equivalent_diameter = Column(Float)
    inscribed_radius = Column(Float)
    form_factor = Column(Float)
    roundness = Column(Float)
    convexity = Column(Float)
    solidity = Column(Float)
    compactness = Column(Float)
    aspect_ratio = Column(Float)
    elongation = Column(Float)
    curl = Column(Float)
    l1 = Column(Float)
    l2 = Column(Float)
    subtribe = Column(String)
    cluster = Column(Integer, nullable=True)


class Database:
    def __init__(self, db_path='phytolith.db'):
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def add_phytolith(self, **kwargs):
        phytolith = Phytolith(**kwargs)
        self.session.add(phytolith)
        self.session.commit()
        return phytolith.id

    def add_batch_from_df(self, df):
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
            'L2': 'l2'
        }

        taxon_mapping = {
            'Subtribe': 'subtribe',
            'Tribe': 'subtribe',
            'Family': 'subtribe',
            'subtribe': 'subtribe',
            'tribe': 'subtribe',
            'family': 'subtribe'
        }

        full_mapping = {**column_mapping, **taxon_mapping}

        df_renamed = df.rename(columns={k: v for k, v in full_mapping.items() if k in df.columns})

        # Функция для очистки числовых значений от ошибочных данных Excel
        def clean_numeric_value(val):
            if pd.isna(val):
                return None
            if isinstance(val, str):
                val = val.replace(',', '.')
                if val in ['#ДЕЛ/0!', '#DIV/0!', '#NULL!', '#REF!', '#NAME?', '#NUM!', '#VALUE!']:
                    return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        for _, row in df_renamed.iterrows():
            phytolith = Phytolith(
                area=clean_numeric_value(row.get('area')),
                convex_area=clean_numeric_value(row.get('convex_area')),
                perimeter=clean_numeric_value(row.get('perimeter')),
                convex_perimeter=clean_numeric_value(row.get('convex_perimeter')),
                length_feret=clean_numeric_value(row.get('length_feret')),
                fiber_length=clean_numeric_value(row.get('fiber_length')),
                width=clean_numeric_value(row.get('width')),
                equivalent_diameter=clean_numeric_value(row.get('equivalent_diameter')),
                inscribed_radius=clean_numeric_value(row.get('inscribed_radius')),
                form_factor=clean_numeric_value(row.get('form_factor')),
                roundness=clean_numeric_value(row.get('roundness')),
                convexity=clean_numeric_value(row.get('convexity')),
                solidity=clean_numeric_value(row.get('solidity')),
                compactness=clean_numeric_value(row.get('compactness')),
                aspect_ratio=clean_numeric_value(row.get('aspect_ratio')),
                elongation=clean_numeric_value(row.get('elongation')),
                curl=clean_numeric_value(row.get('curl')),
                l1=clean_numeric_value(row.get('l1')),
                l2=clean_numeric_value(row.get('l2')),
                subtribe=row.get('subtribe') if pd.notna(row.get('subtribe')) else None,
                cluster=row.get('cluster', None)
            )
            self.session.add(phytolith)
        self.session.commit()
        return len(df_renamed)

    def get_all_phytoliths(self):
        return self.session.query(Phytolith).order_by(Phytolith.id).all()

    def update_clusters(self, clusters_dict):
        """
        Обновляет кластеры в базе данных
        clusters_dict: словарь {id: cluster_value}
        """
        for phytolith_id, cluster_value in clusters_dict.items():
            phytolith = self.session.query(Phytolith).filter_by(id=phytolith_id).first()
            if phytolith:
                phytolith.cluster = cluster_value
        self.session.commit()

    def get_dataframe(self):
        query = self.session.query(Phytolith)
        df = pd.read_sql(query.statement, self.session.bind)
        return df

    def clear_table(self):
        self.session.query(Phytolith).delete()
        self.session.commit()

    def close(self):
        self.session.close()