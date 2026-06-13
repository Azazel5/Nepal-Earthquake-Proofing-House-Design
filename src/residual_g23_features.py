"""Residual G2|G3 specialist features — trained only where run_026 predicts G2."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from embed import NUMERIC_COLS, apply_age_sentinel, load_raw_data
from g23_features import FOUNDATION_STRONG, FOUNDATION_WEAK, load_g23_frames
from preprocess import binary_columns

RESIDUAL_CAT_COLS = [
    "foundation_type",
    "plan_configuration",
    "position",
    "roof_type",
    "land_surface_condition",
    "ground_floor_type",
]
WEAK_SUPER_COLS = [
    "has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_bamboo",
]
GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
RATE_KEYS = [
    "foundation_type",
    "geo_level_1_id",
    "geo_level_3_id",
    "plan_configuration",
    "position",
    "roof_type",
]
LAPLACE = 8.0


def load_residual_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_g23_frames()


def _laplace_rate(keys: pd.Series, is_g3: np.ndarray, alpha: float = LAPLACE) -> dict[str, float]:
    mean = float(is_g3.mean()) if len(is_g3) else 0.35
    out: dict[str, float] = {}
    for k in keys.unique():
        m = keys == k
        n = int(m.sum())
        s = int(is_g3[m].sum())
        out[str(k)] = (s + alpha * mean) / (n + alpha)
    return out


def _meta_from_proba(proba: np.ndarray) -> np.ndarray:
    """Only uncertainty signals — not raw probs (those collapse model to base rate)."""
    p = np.asarray(proba, dtype=np.float32)
    mass = p[:, 1] + p[:, 2]
    q26 = np.divide(p[:, 2], mass, out=np.full(len(p), 0.5, dtype=np.float32), where=mass > 1e-9)
    margin = np.abs(p[:, 1] - p[:, 2])
    ent = -np.sum(np.clip(p, 1e-9, 1.0) * np.log(np.clip(p, 1e-9, 1.0)), axis=1)
    return np.column_stack([mass, q26, margin, ent]).astype(np.float32)


class ResidualG23FeatureBuilder:
    """Features for G3 vs G2 among rows where the base model predicts G2."""

    def __init__(self) -> None:
        self.ohe: OneHotEncoder | None = None
        self.bin_cols: list[str] = []
        self.rate_maps: dict[str, dict[str, float]] = {}
        self.combo_maps: dict[str, dict[str, float]] = {}
        self.global_g3: float = 0.35

    def fit(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        *,
        residual_mask: np.ndarray,
    ) -> ResidualG23FeatureBuilder:
        g23 = np.isin(y, [2, 3])
        sub = df.loc[residual_mask & g23]
        sub_y = (y[residual_mask & g23] == 3).astype(np.int8)
        self.global_g3 = float(sub_y.mean()) if len(sub_y) else 0.35
        self.bin_cols = binary_columns(df)

        self.ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        self.ohe.fit(sub[RESIDUAL_CAT_COLS].astype(str))

        self.rate_maps = {}
        for col in RATE_KEYS:
            self.rate_maps[col] = _laplace_rate(sub[col].astype(str), sub_y)

        ft = sub["foundation_type"].astype(str)
        weak = ft.isin(FOUNDATION_WEAK).astype(int)
        geo1 = sub["geo_level_1_id"].astype(str)
        plan = sub["plan_configuration"].astype(str)
        pos = sub["position"].astype(str)

        self.combo_maps = {
            "geo1_found": _laplace_rate(geo1 + "_" + ft, sub_y),
            "geo1_plan": _laplace_rate(geo1 + "_" + plan, sub_y),
            "geo1_pos": _laplace_rate(geo1 + "_" + pos, sub_y),
            "found_plan": _laplace_rate(ft + "_" + plan, sub_y),
            "weak_geo1": _laplace_rate(geo1 + "_w" + weak.astype(str), sub_y),
        }
        return self

    def _map_rate(self, series: pd.Series, mp: dict[str, float]) -> np.ndarray:
        return series.astype(str).map(mp).fillna(self.global_g3).to_numpy(dtype=np.float32)

    def _structural(self, df: pd.DataFrame) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        ft = df["foundation_type"].astype(str)
        weak_f = ft.isin(FOUNDATION_WEAK).astype(np.float32).to_numpy()
        strong_f = ft.isin(FOUNDATION_STRONG).astype(np.float32).to_numpy()

        weak_mat = (
            df[WEAK_SUPER_COLS].max(axis=1).astype(np.float32).to_numpy()
            if all(c in df.columns for c in WEAK_SUPER_COLS)
            else np.zeros(len(df), dtype=np.float32)
        )
        super_cols = [c for c in df.columns if c.startswith("has_superstructure_")]
        n_super = (
            df[super_cols].sum(axis=1).astype(np.float32).to_numpy()
            if super_cols
            else np.zeros(len(df), dtype=np.float32)
        )

        age = df["age"].astype(np.float32).to_numpy()
        height = df["height_percentage"].astype(np.float32).to_numpy()
        floors = df["count_floors_pre_eq"].astype(np.float32).to_numpy()
        area = df["area_percentage"].astype(np.float32).to_numpy()

        geo1 = df["geo_level_1_id"].astype(str)
        plan = df["plan_configuration"].astype(str)
        pos = df["position"].astype(str)

        g1_rate = self._map_rate(geo1, self.rate_maps.get("geo_level_1_id", {}))
        g3_rate = self._map_rate(df["geo_level_3_id"].astype(str), self.rate_maps.get("geo_level_3_id", {}))
        f_rate = self._map_rate(ft, self.rate_maps.get("foundation_type", {}))
        plan_rate = self._map_rate(plan, self.rate_maps.get("plan_configuration", {}))
        pos_rate = self._map_rate(pos, self.rate_maps.get("position", {}))
        roof_rate = self._map_rate(df["roof_type"].astype(str), self.rate_maps.get("roof_type", {}))

        c_geo_found = self._map_rate(geo1 + "_" + ft, self.combo_maps.get("geo1_found", {}))
        c_geo_plan = self._map_rate(geo1 + "_" + plan, self.combo_maps.get("geo1_plan", {}))
        c_geo_pos = self._map_rate(geo1 + "_" + pos, self.combo_maps.get("geo1_pos", {}))
        c_found_plan = self._map_rate(ft + "_" + plan, self.combo_maps.get("found_plan", {}))
        c_weak_geo = self._map_rate(
            geo1 + "_w" + weak_f.astype(int).astype(str),
            self.combo_maps.get("weak_geo1", {}),
        )

        vuln = weak_f * (1.0 + weak_mat) * (1.0 + floors * 0.1) * (1.0 + height * 0.01)
        tall_weak = weak_f * (floors >= 3).astype(np.float32) * height

        return np.column_stack([
            weak_f,
            strong_f,
            weak_mat,
            n_super,
            vuln,
            tall_weak,
            age * weak_f,
            height * weak_f,
            floors * weak_mat,
            area * floors * weak_f,
            g1_rate,
            g3_rate,
            f_rate,
            plan_rate,
            pos_rate,
            roof_rate,
            c_geo_found,
            c_geo_plan,
            c_geo_pos,
            c_found_plan,
            c_weak_geo,
            f_rate - g1_rate,
            c_weak_geo - f_rate,
        ]).astype(np.float32), (
            f_rate, c_weak_geo, g1_rate,
        )

    def transform(self, df: pd.DataFrame, base_proba: np.ndarray) -> np.ndarray:
        assert self.ohe is not None
        cat_ohe = self.ohe.transform(df[RESIDUAL_CAT_COLS].astype(str))
        geo_raw = df[GEO_COLS].to_numpy(dtype=np.float32)
        numeric = df[NUMERIC_COLS].to_numpy(dtype=np.float32)
        binary = df[self.bin_cols].to_numpy(dtype=np.float32)
        struct, (f_rate, c_weak_geo, g1_rate) = self._structural(df)
        meta = _meta_from_proba(base_proba)
        q26 = meta[:, 1]
        extra = np.column_stack([
            f_rate - q26,
            c_weak_geo - q26,
            g1_rate - q26,
            np.maximum(0.0, q26 - f_rate),
        ]).astype(np.float32)

        return np.hstack([cat_ohe, geo_raw, numeric, binary, struct, meta, extra]).astype(np.float32)
