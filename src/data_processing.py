import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import chi2_contingency


def engineer_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria as features temporais, dinâmicas de ranking e o alvo binário.
    Respeita estritamente o agrupamento por país e álbum.
    """
    df = df.copy()

    # Padronização de datas
    df['date'] = pd.to_datetime(df['date'])
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Ordenação obrigatória para cálculo correto da semana seguinte
    df = df.sort_values(by=['country', 'uri', 'date']).reset_index(drop=True)

    # Definição da Variável-Alvo: O álbum melhorou de posição (rank menor)?
    df['next_rank'] = df.groupby(['country', 'uri'])['rank'].shift(-1)
    df['target_trend_up'] = (df['next_rank'] < df['rank']).astype(int)

    # Remove o último registro de cada ciclo do álbum (sem semana posterior conhecida)
    df = df.dropna(subset=['next_rank']).copy()
    df = df.drop(columns=['next_rank'])

    # Engenharia de Atributos Explicativos
    release_date_clean = df['release_date'].fillna(df['date'])
    df['album_age_days'] = (df['date'] - release_date_clean).dt.days.clip(lower=0)
    df['dist_from_peak'] = df['rank'] - df['peak_rank']
    df['rank_jump_prev'] = np.where(df['previous_rank'] > 0, df['previous_rank'] - df['rank'], 0)
    df['consecutive_weeks'] = df['consecutive_weeks'].fillna(1).astype(int)
    df['weeks_on_chart'] = df['weeks_on_chart'].fillna(1).astype(int)
    df['is_top_50'] = (df['rank'] <= 50).astype(int)
    df['persistence_ratio'] = df['consecutive_weeks'] / np.maximum(df['weeks_on_chart'], 1)

    # Reordena cronologicamente para uso correto em TimeSeriesSplit e nos cortes D0/D1/D2
    df = df.sort_values(by='date').reset_index(drop=True)

    return df


def split_temporal_data(df: pd.DataFrame,
                        split_date_d0: str = "2024-02-29",
                        split_date_d1: str = "2025-04-30") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide a base cronologicamente em D0 (histórico), D1 (recente) e D2 (futuro).
    Proporção cronológica de ~60% / 20% / 20%.
    """
    df_sorted = df.sort_values(by='date').reset_index(drop=True)

    d0 = df_sorted[df_sorted['date'] <= split_date_d0].copy()
    d1 = df_sorted[(df_sorted['date'] > split_date_d0) & (df_sorted['date'] <= split_date_d1)].copy()
    d2 = df_sorted[df_sorted['date'] > split_date_d1].copy()

    return d0, d1, d2


class TemporalPreprocessor:
    """
    Ajusta escalonadores e codificadores EXCLUSIVAMENTE em D0.
    Aplica transformações em D1 e D2 sem contaminar com informações futuras.
    """
    def __init__(self):
        self.num_cols = [
            'rank', 'peak_rank', 'weeks_on_chart', 'consecutive_weeks',
            'album_age_days', 'dist_from_peak', 'rank_jump_prev', 'persistence_ratio'
        ]
        self.cat_cols = ['country', 'entry_status']
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names = []

    def fit(self, df_d0: pd.DataFrame):
        self.scaler.fit(df_d0[self.num_cols])
        self.encoder.fit(df_d0[self.cat_cols])

        encoded_cat_names = self.encoder.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names = self.num_cols + ['is_top_50'] + encoded_cat_names
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_num = self.scaler.transform(df[self.num_cols])
        X_cat = self.encoder.transform(df[self.cat_cols])
        X_binary = df[['is_top_50']].values

        X = np.hstack([X_num, X_binary, X_cat])
        y = df['target_trend_up'].values
        return X, y


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Calcula o Population Stability Index (PSI) entre duas distribuições numéricas.
    Referência de interpretação:
        PSI < 0.10          -> sem mudança significativa
        0.10 <= PSI < 0.25  -> mudança moderada
        PSI >= 0.25         -> mudança severa (drift relevante)
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    # Garante breakpoints estritamente crescentes mesmo com muitos valores repetidos
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    expected_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, breakpoints)[0] / len(actual)

    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


def categorical_drift_test(expected: pd.Series, actual: pd.Series) -> tuple[float, float]:
    """
    Aplica o teste Qui-Quadrado de independência para verificar se a distribuição
    de uma variável categórica mudou entre dois períodos temporais.

    Retorna: (estatística chi2, p-valor)
    """
    freq_expected = expected.value_counts()
    freq_actual = actual.value_counts()

    categories = sorted(set(freq_expected.index) | set(freq_actual.index))
    contingency = pd.DataFrame({
        'expected': [freq_expected.get(c, 0) for c in categories],
        'actual': [freq_actual.get(c, 0) for c in categories]
    }, index=categories)

    chi2_stat, p_value, _, _ = chi2_contingency(contingency.T)
    return float(chi2_stat), float(p_value)