"""Helpers GUI pour règles SkopeRules multi-view (VS + ES)."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from antakia_core.compute.skope_rule.skope_rule import skope_rules
from antakia_core.compute.skope_rule.skope_rule_multiview import (
    MultiviewResultMeta,
    ParetoCandidate,
    skope_rules_multiview,
)
from antakia_core.data_handler.rules import RuleSet
from antakia_core.utils.variable import DataVariables

MultiviewMode = Literal["conjoint", "vs_only", "es_only"]


def multiview_available(x_exp: pd.DataFrame | None) -> bool:
    return x_exp is not None and not x_exp.empty


def find_descriptive_rules(
    selection_mask: pd.Series,
    x_vs: pd.DataFrame,
    x_exp: pd.DataFrame | None,
    variables: DataVariables,
    *,
    multiview: bool,
    mode: MultiviewMode = "conjoint",
    precision: float = 0.7,
    recall: float = 0.7,
) -> tuple[RuleSet, RuleSet, dict, MultiviewResultMeta | None]:
    """
    Retourne (vs_rules, es_rules, score_dict, meta_multiview).

    En mode classique (multiview=False ou ES absent), seules les règles VS
    sont optimisées ; ES reçoit un fit informatif séparé comme avant.
    """
    if multiview and multiview_available(x_exp):
        vs_rules, es_rules, meta = skope_rules_multiview(
            selection_mask,
            x_vs,
            x_exp,
            variables=variables,
            precision=precision,
            recall=recall,
        )
        vs_rules, es_rules, score = _pick_from_pareto(vs_rules, es_rules, meta, mode)
        score_dict = dict(score)
        score_dict["multiview_mode"] = mode
        score_dict["multiview"] = True
        return vs_rules, es_rules, score_dict, meta

    es_rules, _ = skope_rules(
        selection_mask,
        x_exp if multiview_available(x_exp) else x_vs,
        variables=variables,
        precision=precision,
        recall=recall,
    )
    vs_rules, score_dict = skope_rules(
        selection_mask,
        x_vs,
        variables=variables,
        precision=precision,
        recall=recall,
    )
    score_dict["multiview"] = False
    return vs_rules, es_rules, score_dict, None


def apply_multiview_mode(
    default_vs: RuleSet,
    default_es: RuleSet,
    meta: MultiviewResultMeta,
    mode: MultiviewMode,
) -> tuple[RuleSet, RuleSet, dict]:
    return _pick_from_pareto(default_vs, default_es, meta, mode)


def _pick_from_pareto(
    default_vs: RuleSet,
    default_es: RuleSet,
    meta: MultiviewResultMeta,
    mode: MultiviewMode,
) -> tuple[RuleSet, RuleSet, dict]:
    pareto: list[ParetoCandidate] = meta.get("pareto") or []
    for candidate in pareto:
        if candidate["kind"] == mode:
            return (
                candidate["vs"].copy(),
                candidate["es"].copy(),
                dict(candidate["score"]),
            )
    return default_vs, default_es, dict(meta.get("score") or {})


def format_multiview_status(score_dict: dict) -> str:
    if not score_dict.get("multiview"):
        return ""
    mode = score_dict.get("multiview_mode", "conjoint")
    labels = {
        "conjoint": "VS∧ES conjoint",
        "vs_only": "VS seul",
        "es_only": "ES seul",
    }
    p = score_dict.get("precision", "-")
    r = score_dict.get("recall", "-")
    f1 = score_dict.get("f1", "-")
    return f"Multi-view ({labels.get(mode, mode)}) — P={p} R={r} F1={f1}"
