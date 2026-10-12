"""Эксперименты сравнения моделей: Random Forest vs CatBoost, абляция ALS.

Модуль собирает пайплайны и протокол оценки так, чтобы сравнения были честными:
одинаковые признаки, одинаковое разбиение, одинаковая схема кросс-валидации
и одинаковый бюджет подбора гиперпараметров (число триалов и таймаут Optuna).
Всё, что различается между плечами эксперимента, перечислено явно в отчёте
(например, Random Forest исторически использует встроенный отбор признаков,
а CatBoost — нативную обработку категорий без one-hot).

Плечи эксперимента
------------------
``random_forest``
    ``ColumnTransformer(StandardScaler + OneHotEncoder)`` → ``SelectFromModel(RF)``
    → ``RandomForestClassifier(class_weight='balanced')``. Исторический пайплайн
    проекта; отбор признаков можно отключить флагом ``use_feature_selection=False``.
``catboost``
    ``ColumnTransformer(passthrough)`` → ``CatBoostClassifier`` с ``cat_features``
    по именам колонок: категориальные признаки кодируются внутри бустинга
    (ordered target statistics), без one-hot и без масштабирования.


Метрики
-------
``evaluate`` считает полный набор метрик на тесте: macro/weighted precision, recall,
F1, ROC-AUC (OVR macro), per-class PR-AUC с бейзлайном случайного ранжирования
и top-K метрики рекомендаций (`recsys_metrics.top_k_report`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

from recsys_metrics import OTHER_CLASS, top_k_report

# Признак персональной ALS-рекомендации: его наличие/отсутствие — предмет абляции.
ALS_FEATURE = 'recommended_product_id'
MODEL_KINDS = ('random_forest', 'catboost')
MODEL_LABELS = {
    'random_forest': 'Random Forest (OHE + SelectFromModel)',
    'catboost': 'CatBoost (нативные категориальные признаки)',
}
SELECTION_ESTIMATOR_SEED = 0


@dataclass
class StudyResult:
    """Результат подбора гиперпараметров Optuna для одного плеча эксперимента."""

    model_kind: str
    best_params: Dict[str, Any]
    best_value: float
    cv_std: float
    n_trials: int
    elapsed_sec: float
    study: Any = field(repr=False, default=None)

    def as_row(self) -> Dict[str, Any]:
        return {
            'model': self.model_kind,
            'model_label': MODEL_LABELS.get(self.model_kind, self.model_kind),
            'cv_roc_auc_macro': self.best_value,
            'cv_roc_auc_std': self.cv_std,
            'optuna_trials': self.n_trials,
            'optuna_timeout': self.elapsed_sec,
            'best_params': self.best_params,
        }


@dataclass
class EvaluationResult:
    """Полный результат оценки модели на тестовой выборке."""

    metrics: Dict[str, float]
    per_class: pd.DataFrame
    top_k: pd.DataFrame
    y_pred: np.ndarray = field(repr=False, default=None)
    y_proba: np.ndarray = field(repr=False, default=None)

    def summary(self, name: str) -> Dict[str, Any]:
        """Одна строка для таблиц сравнения моделей и абляций."""
        row: Dict[str, Any] = {'variant': name}
        row.update(self.metrics)
        for scope in ('shortlist_purchasers', 'purchasers_any', 'all_rows'):
            scoped = self.top_k[(self.top_k['scope'] == scope) & (self.top_k['k'] == 5)]
            for column in ('precision_at_k', 'recall_at_k', 'ndcg_at_k', 'hit_rate_at_k'):
                if not scoped.empty:
                    row[f'{scope}_{column}'] = float(scoped[column].iloc[0])
        return row


def build_preprocessor(model_kind: str, numeric_features: Sequence[str],
                       categorical_features: Sequence[str]) -> ColumnTransformer:
    """Препроцессор под конкретную модель.

    Random Forest не умеет категории — получает масштабированные числа и one-hot.
    CatBoost кодирует категории сам, поэтому для него колонки передаются как есть,
    а имена сохраняются (``set_output(transform='pandas')``) — по ним классификатору
    сообщается список ``cat_features``.
    """
    if model_kind == 'random_forest':
        return ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), list(numeric_features)),
                ('cat', OneHotEncoder(handle_unknown='ignore'), list(categorical_features)),
            ])
    if model_kind == 'catboost':
        # verbose_feature_names_out=False: имена колонок остаются исходными — по ним
        # классификатору сообщается список cat_features, имена с префиксами ему не подходят.
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', list(numeric_features)),
                ('cat', 'passthrough', list(categorical_features)),
            ],
            verbose_feature_names_out=False)
        preprocessor.set_output(transform='pandas')
        return preprocessor
    raise ValueError(f'неизвестная модель: {model_kind!r}; ожидается одна из {MODEL_KINDS}')


def build_classifier(model_kind: str, *, seed: int = 0,
                     categorical_features: Sequence[str] = ()):
    """Классификатор плеча в конфигурации по умолчанию (гиперпараметры — через pipeline)."""
    if model_kind == 'random_forest':
        return RandomForestClassifier(class_weight='balanced', random_state=seed, n_jobs=-1)
    if model_kind == 'catboost':
        from catboost import CatBoostClassifier  # импорт внутри: модуль нужен только этому плечу

        return CatBoostClassifier(
            loss_function='MultiClass',
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
            cat_features=list(categorical_features))
    raise ValueError(f'неизвестная модель: {model_kind!r}; ожидается одна из {MODEL_KINDS}')


def build_pipeline(model_kind: str, numeric_features: Sequence[str],
                   categorical_features: Sequence[str],
                   params: Optional[Mapping[str, Any]] = None, *, seed: int = 0,
                   use_feature_selection: bool = True) -> Pipeline:
    """Пайплайн «препроцессор → (отбор признаков) → классификатор» под плечо эксперимента.

    ``params`` — гиперпараметры в нотации sklearn Pipeline (``classifier__iterations``
    и т.п.): ровно в таком виде их предлагает Optuna и логирует MLflow, поэтому
    в проекте нет второй системы имён.
    """
    steps: List[Tuple[str, Any]] = [
        ('preprocessor', build_preprocessor(model_kind, numeric_features, categorical_features)),
    ]
    if use_feature_selection and model_kind == 'random_forest':
        steps.append(('feature_selection', SelectFromModel(RandomForestClassifier(
            n_estimators=50, class_weight='balanced',
            random_state=SELECTION_ESTIMATOR_SEED, n_jobs=-1))))
    steps.append(('classifier', build_classifier(
        model_kind, seed=seed, categorical_features=categorical_features)))
    pipeline = Pipeline(steps)
    if params:
        pipeline.set_params(**dict(params))
    return pipeline


def suggest_params(trial: 'optuna.Trial', model_kind: str) -> Dict[str, Any]:
    """Пространство поиска гиперпараметров для плеча.

    Random Forest — историческое пространство проекта (совместимо с прошлыми
    прогонами); CatBoost — компактное пространство с ограничением по числу
    итераций, чтобы бюджет подбора оставался сопоставимым по времени.
    """
    if model_kind == 'random_forest':
        return {
            'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 5, 50),
            'classifier__max_depth': trial.suggest_categorical(
                'classifier__max_depth', [1, 5, 10, 30, None]),
            'classifier__min_samples_split': trial.suggest_int(
                'classifier__min_samples_split', 2, 20),
        }
    if model_kind == 'catboost':
        return {
            'classifier__iterations': trial.suggest_int('classifier__iterations', 50, 400),
            'classifier__depth': trial.suggest_int('classifier__depth', 3, 8),
            'classifier__learning_rate': trial.suggest_float(
                'classifier__learning_rate', 0.03, 0.3, log=True),
            'classifier__l2_leaf_reg': trial.suggest_float(
                'classifier__l2_leaf_reg', 1.0, 10.0),
        }
    raise ValueError(f'неизвестная модель: {model_kind!r}; ожидается одна из {MODEL_KINDS}')


def macro_roc_auc(y_true: Sequence[int], y_proba: np.ndarray, classes: Sequence[int]) -> float:
    """macro ROC-AUC (OVR) с защитой от классов, которых нет в модели, и ренормировкой."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    classes = list(classes)

    row_sums = y_proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_proba = y_proba / row_sums

    present = [cls for cls in classes if (y_true == cls).any()]
    if not present:
        return float('nan')

    mask = np.isin(y_true, present)
    columns = [classes.index(cls) for cls in present]
    y_proba_sub = y_proba[mask][:, columns]
    y_true_sub = np.array([present.index(value) for value in y_true[mask]])

    if len(present) == 2:
        return float(roc_auc_score(y_true_sub, y_proba_sub[:, 1]))
    return float(roc_auc_score(
        label_binarize(y_true_sub, classes=list(range(len(present)))),
        y_proba_sub,
        multi_class='ovr',
        average='macro',
    ))


def cross_val_macro_roc_auc(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                            cv) -> np.ndarray:
    """ROC-AUC по фолдам кросс-валидации — тестовая выборка не используется."""
    scores = []
    for train_index, valid_index in cv.split(X_train, y_train):
        fold_model = clone(pipeline).fit(X_train.iloc[train_index], y_train.iloc[train_index])
        scores.append(macro_roc_auc(
            y_train.iloc[valid_index].to_numpy(),
            fold_model.predict_proba(X_train.iloc[valid_index]),
            list(fold_model.named_steps['classifier'].classes_),
        ))
    return np.asarray(scores, dtype=float)


def make_objective(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv,
                   model_kind: str):
    """Objective для Optuna: ROC-AUC по кросс-валидации на train.

    Тестовая выборка в подборе гиперпараметров не участвует: иначе метрика на тесте —
    это максимум из перебора, а не честная оценка.
    """

    def objective(trial: 'optuna.Trial') -> float:
        pipeline.set_params(**suggest_params(trial, model_kind))
        scores = cross_val_macro_roc_auc(pipeline, X_train, y_train, cv)
        trial.set_user_attr('cv_std', float(np.nanstd(scores)))
        return float(np.nanmean(scores))

    return objective


def run_study(model_kind: str, X_train: pd.DataFrame, y_train: pd.Series,
              numeric_features: Sequence[str], categorical_features: Sequence[str], *,
              n_trials: int = 20, timeout: int = 1200, n_splits: int = 3,
              seed: int = 42, use_feature_selection: bool = True,
              sampler_seed: int = 42) -> StudyResult:
    """Подбор гиперпараметров плеча с фиксированным бюджетом (trials + timeout)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipeline = build_pipeline(model_kind, numeric_features, categorical_features,
                              seed=seed, use_feature_selection=use_feature_selection)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=sampler_seed))
    started = time.perf_counter()
    study.optimize(make_objective(pipeline, X_train, y_train, cv, model_kind),
                   n_trials=n_trials, timeout=timeout)
    elapsed = time.perf_counter() - started

    completed = [trial for trial in study.trials
                 if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise ValueError(f'Optuna не завершила ни одного триала для {model_kind} '
                         f'за {timeout} с — увеличьте timeout или упростите поиск')

    return StudyResult(
        model_kind=model_kind,
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        cv_std=float(study.best_trial.user_attrs.get('cv_std', float('nan'))),
        n_trials=len(study.trials),
        elapsed_sec=elapsed,
        study=study,
    )


def build_model(result: StudyResult, numeric_features: Sequence[str],
                categorical_features: Sequence[str], *, seed: int = 0,
                use_feature_selection: bool = True) -> Pipeline:
    """Пайплайн плеча с лучшими гиперпараметрами (ещё не обученный)."""
    return build_pipeline(result.model_kind, numeric_features, categorical_features,
                          params=result.best_params, seed=seed,
                          use_feature_selection=use_feature_selection)


def multiclass_metrics(y_true: Sequence[int], y_pred: np.ndarray, y_proba: np.ndarray,
                       classes: Sequence[int],
                       label_map: Optional[Mapping[int, str]] = None) -> Tuple[Dict[str, float],
                                                                             pd.DataFrame]:
    """macro/weighted метрики классификации и per-class PR-AUC с random-бейзлайном."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    label_map = dict(label_map or {int(cls): str(cls) for cls in classes})

    metrics = {
        'roc_auc_ovr_macro': macro_roc_auc(y_true, y_proba, classes),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

    rows = []
    for index, cls in enumerate(classes):
        y_binary = (y_true == int(cls)).astype(int)
        if y_binary.sum() == 0:
            continue
        rows.append({
            'class': int(cls),
            'product': label_map.get(int(cls), str(cls)),
            'support': int(y_binary.sum()),
            'share': float(y_binary.mean()),
            'pr_auc': float(average_precision_score(y_binary, y_proba[:, index])),
            'roc_auc_ovr': float(roc_auc_score(y_binary, y_proba[:, index])),
        })
    per_class = pd.DataFrame(rows).sort_values('share', ascending=False)
    metrics['pr_auc_macro'] = float(per_class['pr_auc'].mean())
    return metrics, per_class


def baseline_metrics(per_class: pd.DataFrame, metrics: Mapping[str, float]) -> Dict[str, float]:
    """Бейзлайны интерпретации: accuracy «always no_purchase» и lift PR-AUC по продуктам."""
    no_purchase = per_class[per_class['class'] == 0]
    no_purchase_share = (float(no_purchase['share'].iloc[0])
                         if not no_purchase.empty else float('nan'))

    product_classes = per_class[(~per_class['class'].isin([0, OTHER_CLASS]))
                                & (per_class['support'] > 0)]
    product_mean_pr_auc = float(product_classes['pr_auc'].mean()) if not product_classes.empty \
        else float('nan')
    product_random_pr_auc = float(product_classes['share'].mean()) if not product_classes.empty \
        else float('nan')
    return {
        'no_purchase_accuracy': no_purchase_share,
        'accuracy_lift_pp': (float(metrics['accuracy']) - no_purchase_share) * 100,
        'product_classes_mean_pr_auc': product_mean_pr_auc,
        'product_classes_mean_random_pr_auc': product_random_pr_auc,
        'product_classes_pr_auc_lift': product_mean_pr_auc / product_random_pr_auc
        if product_random_pr_auc else float('nan'),
    }


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: Sequence[int], *,
             candidate_classes: Sequence[int], label_map: Optional[Mapping[int, str]] = None,
             ks: Sequence[int] = (1, 3, 5, 10),
             popularity: Optional[Mapping[int, float]] = None) -> EvaluationResult:
    """Единая оценка любого плеча: классификационные метрики + top-K рекомендаций.

    И препроцессор, и классификатор живут внутри пайплайна, поэтому оценка не зависит
    от плеча: сравнение Random Forest и CatBoost идёт на одном и том же тесте
    одной и той же функцией.
    """
    y_pred = np.asarray(model.predict(X_test))
    y_proba = np.asarray(model.predict_proba(X_test), dtype=np.float64)
    classes = [int(cls) for cls in model.named_steps['classifier'].classes_]

    metrics, per_class = multiclass_metrics(y_test, y_pred, y_proba, classes, label_map)
    top_k = top_k_report(np.asarray(y_test), y_proba, classes, candidate_classes,
                         ks=ks, popularity=dict(popularity) if popularity else None)
    return EvaluationResult(metrics=metrics, per_class=per_class, top_k=top_k,
                            y_pred=y_pred, y_proba=y_proba)


def drop_feature(features: Sequence[str], feature: str) -> List[str]:
    """Список признаков без одного столбца (для абляций)."""
    remaining = [col for col in features if col != feature]
    if len(remaining) == len(features):
        raise ValueError(f'признака {feature!r} нет среди {list(features)}')
    return remaining


def ablation_rows(variants: Mapping[str, EvaluationResult]) -> pd.DataFrame:
    """Таблица абляции: по строке на вариант, метрики в одном наборе колонок."""
    rows = [result.summary(name) for name, result in variants.items()]
    table = pd.DataFrame(rows)
    ordered = ['variant', 'roc_auc_ovr_macro', 'precision_macro', 'recall_macro',
               'f1_macro', 'pr_auc_macro']
    ordered += [column for column in table.columns
                if column.startswith(('shortlist_purchasers_', 'all_rows_',
                                      'purchasers_any_'))]
    ordered = [column for column in ordered if column in table.columns]
    return table[ordered] if ordered else table


def comparison_table(studies: Sequence[StudyResult],
                     evaluations: Mapping[str, EvaluationResult]) -> pd.DataFrame:
    """Таблица сравнения моделей: CV-метрики подбора + метрики на общем тесте."""
    rows = []
    for study in studies:
        row = study.as_row()
        row['best_params'] = str(row['best_params'])
        evaluation = evaluations.get(study.model_kind)
        if evaluation is not None:
            row.update({key: value for key, value in evaluation.metrics.items()})
            for scope in ('shortlist_purchasers', 'all_rows'):
                scoped = evaluation.top_k[(evaluation.top_k['scope'] == scope)
                                          & (evaluation.top_k['k'] == 5)]
                for column in ('precision_at_k', 'recall_at_k', 'ndcg_at_k'):
                    if not scoped.empty:
                        row[f'{scope}_{column}'] = float(scoped[column].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)
