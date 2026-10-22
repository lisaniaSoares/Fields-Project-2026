from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = (
    PROJECT_ROOT
    / "Dados_Brutos_Oil_Gas"
    / "Processados"
    / "etanol_gasolina_2026_teste_limpo.csv"
)
RESULTS_DIR = PROJECT_ROOT / "baseline_results"
PREDICTIONS_PATH = RESULTS_DIR / "previsoes_gasolina_sao_paulo.csv"
TRAIN_FRACTION = 0.8


def load_weekly_prices() -> pd.Series:
    data = pd.read_csv(
        DATA_PATH,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        parse_dates=["Data da Coleta"],
    )
    required_columns = {
        "Produto",
        "Municipio",
        "Estado - Sigla",
        "Data da Coleta",
        "Valor de Venda",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing_columns)}")

    target = data.loc[
        data["Produto"].astype("string").str.strip().str.upper().eq("GASOLINA")
        & data["Municipio"].astype("string").str.strip().str.upper().eq("SAO PAULO")
        & data["Estado - Sigla"].astype("string").str.strip().str.upper().eq("SP"),
        ["Data da Coleta", "Valor de Venda"],
    ].copy()
    if target.empty:
        raise ValueError("Não há registros de GASOLINA para São Paulo/SP.")

    target["Valor de Venda"] = pd.to_numeric(target["Valor de Venda"], errors="coerce")
    invalid = target["Data da Coleta"].isna() | target["Valor de Venda"].isna()
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} registros-alvo têm data ou preço ausente/inválido."
        )
    if (target["Valor de Venda"] <= 0).any():
        raise ValueError("Há preços não positivos nos registros-alvo.")

    target["Semana_Inicio"] = (
        target["Data da Coleta"]
        - pd.to_timedelta(target["Data da Coleta"].dt.weekday, unit="D")
    )
    weekly_prices = target.groupby("Semana_Inicio")["Valor de Venda"].mean().sort_index()
    complete_weeks = pd.date_range(
        start=weekly_prices.index.min(),
        end=weekly_prices.index.max(),
        freq="7D",
        name="Semana_Inicio",
    )
    weekly_prices = weekly_prices.reindex(complete_weeks)
    if weekly_prices.isna().any():
        absent = weekly_prices.index[weekly_prices.isna()].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Sem observações de preço nas semanas: {absent}")
    weekly_prices.name = "Preco_Medio_Semanal"
    return weekly_prices


def make_features(weekly_prices: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame(index=weekly_prices.index)
    week_number = features.index.isocalendar().week.to_numpy(dtype=float)
    features["tendencia_semanal"] = np.arange(len(weekly_prices), dtype=float)
    features["semana_seno"] = np.sin(2 * np.pi * week_number / 52.0)
    features["semana_cosseno"] = np.cos(2 * np.pi * week_number / 52.0)
    for lag in (1, 2, 4):
        features[f"preco_lag_{lag}"] = weekly_prices.shift(lag)
    features["media_movel_4_semanas"] = weekly_prices.shift(1).rolling(4).mean()
    return features


def main() -> None:
    weekly_prices = load_weekly_prices()
    features = make_features(weekly_prices)
    supervised = features.join(weekly_prices.rename("preco_real")).dropna()

    split_index = int(len(weekly_prices) * TRAIN_FRACTION)
    split_date = weekly_prices.index[split_index]
    train = supervised.loc[supervised.index < split_date]
    test = supervised.loc[supervised.index >= split_date]
    if train.empty or test.empty:
        raise ValueError("A divisão cronológica não produziu treino e teste válidos.")

    train_matrix = xgb.DMatrix(
        train.drop(columns="preco_real"),
        label=train["preco_real"],
    )
    test_matrix = xgb.DMatrix(test.drop(columns="preco_real"))
    model = xgb.train(
        {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 2,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "lambda": 1.0,
            "seed": 42,
            "nthread": 1,
            "tree_method": "hist",
        },
        train_matrix,
        num_boost_round=100,
    )
    predictions = model.predict(test_matrix)
    actual = test["preco_real"].to_numpy()

    mae = float(np.mean(np.abs(actual - predictions)))
    rmse = float(np.sqrt(np.mean(np.square(actual - predictions))))
    forecast = pd.DataFrame(
        {
            "Semana_Inicio": test.index.strftime("%Y-%m-%d"),
            "Preco_Real_Medio_Semanal": actual,
            "Preco_Previsto": predictions,
            "Erro_Absoluto": np.abs(actual - predictions),
        }
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(PREDICTIONS_PATH, index=False, float_format="%.6f")

    print("Baseline XGBoost - GASOLINA em São Paulo/SP")
    print(f"Registros agregados em semanas: {len(weekly_prices)}")
    print(
        f"Semanas supervisionadas usadas: {len(supervised)} "
        f"(treino={len(train)}, teste={len(test)})"
    )
    print(
        "Período completo: "
        f"{weekly_prices.index.min():%Y-%m-%d} a "
        f"{weekly_prices.index.max():%Y-%m-%d}"
    )
    print(
        f"Treino: {train.index.min():%Y-%m-%d} a {train.index.max():%Y-%m-%d}"
    )
    print(f"Teste: {test.index.min():%Y-%m-%d} a {test.index.max():%Y-%m-%d}")
    print(f"MAE: R$ {mae:.4f}/litro")
    print(f"RMSE: R$ {rmse:.4f}/litro")
    print(f"Previsões salvas em: {PREDICTIONS_PATH}")
    print("\nAmostra das previsões:")
    print(forecast.head().to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
