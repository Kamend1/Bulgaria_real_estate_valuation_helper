"""
Unit tests for utils/ml/avm_features.py::clean_shap_feature_name (Phase 14
Tier 2.3) -- pure string logic, no model/DB involved. Verified live against
real ColumnTransformer.get_feature_names_out() output from a trained
hospitality pipeline before writing these (see session notes); these lock
in that exact naming shape.
"""
from utils.ml.avm_features import clean_shap_feature_name


def test_numeric_column_gets_bulgarian_label():
    assert clean_shap_feature_name("num__area_sqm") == "Площ"
    assert clean_shap_feature_name("num__views") == "Брой прегледи"


def test_tfidf_text_component_gets_readable_label():
    assert clean_shap_feature_name("num__txt_tfidf_4") == "Текстово описание (компонента 4)"


def test_categorical_onehot_column_splits_base_and_category():
    assert clean_shap_feature_name("cat__geo_category_small_city") == "Гео-категория: small_city"
    assert clean_shap_feature_name("cat__property_type_raw_ХОТЕЛ") == "Тип имот: ХОТЕЛ"


def test_categorical_column_prefers_longer_match_over_shorter_substring():
    # "geo_category" is itself a substring check target -- must not shadow
    # a genuinely different, unrelated categorical column that happens to
    # share a prefix in theory. Regression guard for the sort-by-length step.
    assert clean_shap_feature_name("cat__title_geo_2_model_Лозенец") == "Квартал: Лозенец"


def test_binary_passthrough_column_strips_prefix():
    assert clean_shap_feature_name("bin__feature_Асансьор") == "Асансьор"


def test_unknown_prefix_returned_unchanged():
    assert clean_shap_feature_name("unrecognized_raw_name") == "unrecognized_raw_name"
