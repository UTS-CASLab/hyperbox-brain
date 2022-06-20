"""
The :mod:`hbbrain.utils` module implements a variety of utility functions for
hyperbox-based classifiers.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0


from hbbrain.utils.membership_calc import (
    membership_func_gfmm,
    asym_similarity_val_one_many_hyperboxes,
    get_membership_gfmm_all_classes,
    membership_func_fmnn,
    get_membership_fmnn_all_classes,
    membership_func_onehot_gfmm,
    get_membership_onehot_gfmm_all_classes,
    membership_func_freq_cat_gfmm,
    get_membership_freq_cat_gfmm_all_classes,
    membership_func_extended_iol_gfmm,
    get_membership_extended_iol_gfmm_all_classes,
    membership_func_free_range_gfmm,
    get_membership_free_range_gfmm_all_classes
)
from hbbrain.utils.dist_metrics import (
    manhattan_distance,
    manhattan_distance_with_missing_val
)
from hbbrain.utils.model_storage import (
    load_multi_models,
    load_model,
    store_model,
)
from hbbrain.utils.data_editing import (
    data_editing_leave_one_out,
    data_editing_two_fold_cv,
    data_editing_two_fold_cv_with_probability,
)
from hbbrain.utils.adjust_hyperbox import (
    is_overlap_one_many_hyperboxes_num_data_general,
    is_overlap_one_many_diff_label_hyperboxes_num_data_general,
    is_two_hyperboxes_overlap_num_data_general,
    overlap_resolving_num_data,
    hyperbox_overlap_test_fmnn,
    hyperbox_contraction_fmnn,
    hyperbox_overlap_test_efmnn,
    hyperbox_contraction_efmnn,
    is_overlap_diff_labels_num_data_rfmnn,
    hyperbox_contraction_rfmnn,
    hyperbox_overlap_test_freq_cat_gfmm,
    hyperbox_contraction_freq_cat_gfmm,
    is_overlap_one_many_diff_label_hyperboxes_mixed_data_general,
    is_two_hyperboxes_overlap_num_data_free_range_general,
    overlap_resolving_num_data_free_range
)
__all__ = [
    "membership_func_gfmm",
    "asym_similarity_val_one_many_hyperboxes",
    "get_membership_gfmm_all_classes",
    "membership_func_fmnn",
    "get_membership_fmnn_all_classes",
    "membership_func_onehot_gfmm",
    "get_membership_onehot_gfmm_all_classes",
    "membership_func_freq_cat_gfmm",
    "get_membership_freq_cat_gfmm_all_classes",
    "membership_func_extended_iol_gfmm",
    "get_membership_extended_iol_gfmm_all_classes",
    "membership_func_free_range_gfmm",
    "get_membership_free_range_gfmm_all_classes",
    "manhattan_distance",
    "manhattan_distance_with_missing_val",
    "load_multi_models",
    "load_model",
    "store_model",
    "is_overlap_one_many_hyperboxes_num_data_general",
    "is_overlap_one_many_diff_label_hyperboxes_num_data_general",
    "is_two_hyperboxes_overlap_num_data_general",
    "overlap_resolving_num_data",
    "hyperbox_overlap_test_fmnn",
    "hyperbox_contraction_fmnn",
    "hyperbox_overlap_test_efmnn",
    "hyperbox_contraction_efmnn",
    "is_overlap_diff_labels_num_data_rfmnn",
    "hyperbox_contraction_rfmnn",
    "hyperbox_overlap_test_freq_cat_gfmm",
    "hyperbox_contraction_freq_cat_gfmm",
    "is_overlap_one_many_diff_label_hyperboxes_mixed_data_general",
    "is_two_hyperboxes_overlap_num_data_free_range_general",
    "overlap_resolving_num_data_free_range",
    "data_editing_leave_one_out",
    "data_editing_two_fold_cv",
    "data_editing_two_fold_cv_with_probability"
]
