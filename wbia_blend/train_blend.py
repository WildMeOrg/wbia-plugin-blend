# -*- coding: utf-8 -*-

import numpy as np
import wbia  # NOQA

QUERY_CONFIG_DICT_DICT = {
    'PIE v2': {'pipeline_root': 'PieTwo'},
    'HotSpotter': {'sv_on': True},
}


def train(ibs, aid_list, query_config_dict_dict):
    """
    End-to-end blend fitting of algos specified by query_config_dict_dict on aid_list.
    My use has been to embed and work through this func interactively.
    Args:
        ibs (IBEISController): IBEIS / WBIA controller object
        aid_list: list of annotation ids
        query_config_dict_dict: dict with algo name keys and query config dict values
    """
    score_matrix_dict = {}
    algo_names = list(query_config_dict_dict.keys())
    for algo_name in algo_names:
        score_matrix = compute_score_matrix(
            ibs, aid_list, query_config_dict_dict[algo_name]
        )
        score_matrix_dict[algo_name] = score_matrix

    names = ibs.get_annot_names(aid_list)
    truth_matrix = np.array(
        [
            [names[i] == names[j] for j in range(len(aid_list))]
            for i in range(len(aid_list))
        ]
    )

    topk_dict = {
        algo_name: score_matrix_to_topk(score_matrix_dict[algo_name], truth_matrix)
        for algo_name in algo_names
    }
    for algo_name in score_matrix_dict.keys():
        print(f'{algo_name} top_k accuracies: {topk_dict[algo_name]}')

    algo_names = list(score_matrix_dict.keys())
    score_matrices = [score_matrix_dict[algo_name] for algo_name in algo_names]

    optimum_alg1_weight = optimize_weights(score_matrices, truth_matrix)
    print(f'alg1 is {algo_names[0]}')

    return optimum_alg1_weight


def compute_score_matrix(ibs, aid_list, query_config_dict, no_score_val=0.0):
    r"""
    Generate embeddings using the Pose-Invariant Embedding (PIE)
    Args:
        ibs (IBEISController): IBEIS / WBIA controller object
        aid_list  (int): annot ids specifying the input
        query_config_dict: ID configuration dict passed to ibs.query_chips_graph
    Example:
        >>> # ENABLE_DOCTEST
        >>> # Note that this is an expensive test that loads an 1879-image dataset the first time it's run
        >>> import wbia_blend
        >>> # from wbia_blend import QUERY_CONFIG_DICT_DICT
        >>> # use PIE sample data
        >>> import wbia_pie_v2
        >>> from wbia_pie_v2._plugin import DEMOS, CONFIGS, MODELS
        >>> species = 'whale_grey'
        >>> test_ibs = wbia_pie_v2._plugin.wbia_pie_v2_test_ibs(DEMOS[species], species, 'test2021')
        >>> aid_list = test_ibs.get_valid_aids(species=species)
        >>> names = test_ibs.get_annot_names(aid_list)
        >>> test_names = ['33dfd0d7-ce13-43b7-9c53-a4a1e4fa69d1', '740b1b08-5f33-4810-9fdf-26b0b27260cf']
        >>> test_aids = [aid for aid, name in zip(aid_list, names) if name in test_names]
        >>> config = QUERY_CONFIG_DICT_DICT['PIE v2']
        >>> score_matrix = compute_score_matrix(test_ibs, test_aids, config)
        >>> # no good way to format this; at least this is readable if long
        >>> expected_matrix = np.array([
        >>>     [0.00000000e+00, 5.86341648e-04, 5.86341648e-04, 5.86341648e-04, 5.86341648e-04, 5.86341648e-04, 3.10301033e-05, 3.10301033e-05, 3.10301033e-05],
        >>>     [1.36292574e-03, 0.00000000e+00, 1.36292574e-03, 1.36292574e-03, 1.36292574e-03, 1.36292574e-03, 2.65372525e-05, 2.65372525e-05, 2.65372525e-05],
        >>>     [1.84212429e-03, 1.84212429e-03, 0.00000000e+00, 1.84212429e-03, 1.84212429e-03, 1.84212429e-03, 4.21024098e-05, 4.21024098e-05, 4.21024098e-05],
        >>>     [1.84212429e-03, 1.84212429e-03, 1.84212429e-03, 0.00000000e+00, 1.84212429e-03, 1.84212429e-03, 4.14823763e-05, 4.14823763e-05, 4.14823763e-05],
        >>>     [4.35253874e-04, 4.35253874e-04, 4.35253874e-04, 4.35253874e-04, 0.00000000e+00, 4.35253874e-04, 2.72582528e-05, 2.72582528e-05, 2.72582528e-05],
        >>>     [5.70367500e-04, 5.70367500e-04, 5.70367500e-04, 5.70367500e-04, 5.70367500e-04, 0.00000000e+00, 5.10863716e-05, 5.10863716e-05, 5.10863716e-05],
        >>>     [1.09777973e-05, 1.09777973e-05, 1.09777973e-05, 1.09777973e-05, 1.09777973e-05, 1.09777973e-05, 0.00000000e+00, 2.93286728e-04, 2.93286728e-04],
        >>>     [2.55431858e-05, 2.55431858e-05, 2.55431858e-05, 2.55431858e-05, 2.55431858e-05, 2.55431858e-05, 2.96477435e-04, 0.00000000e+00, 2.96477435e-04],
        >>>     [9.37124476e-06, 9.37124476e-06, 9.37124476e-06, 9.37124476e-06, 9.37124476e-06, 9.37124476e-06, 2.96477435e-04, 2.96477435e-04, 0.00000000e+00]
        >>> ])
        >>> diff = abs(expected_matrix - score_matrix)
        >>> assert (diff < 1e-8).all()
    """

    n = len(aid_list)
    score_matrix = np.full((n, n), no_score_val)
    auuid_list = ibs.get_annot_uuids(aid_list)
    for query_index, (qaid, qauuid) in enumerate(zip(aid_list, auuid_list)):
        result = ibs.query_chips_graph(
            qaid_list=[qaid],
            daid_list=aid_list,
            query_config_dict=query_config_dict,
            echo_query_params=False,
            cache_images=False,
            n=1,
        )
        score_list = get_score_array(result, qauuid, auuid_list, no_score_val)
        score_matrix[query_index, :] = score_list

    # replace any -inf score with no_score_val
    score_matrix[np.isneginf(score_matrix)] = no_score_val
    return score_matrix


# # result_dict is result_list['cm_dict'][qauuid]
# def get_score_list(query_result, qauuid, no_score_val=0.0):
#     result_dict = query_result['cm_dict'][str(qauuid)]
#     result_dauuids = result_dict['dauuid_list']
#     # TDOD: confirm if we want annot_score_list or score_list below
#     result_scores = result_dict['annot_score_list']
#     assert len(result_dauuids) == len(result_scores)
#     dauid_scores = {dauuid: score for dauuid, score in zip(result_dauuids, result_scores)}
#     score_list = [dauid_scores.get(dauuid, no_score_val) for dauuid in dauuid_list]
#     return score_list

# TODO: remove duplicate func and import from train_blend
def get_score_array(query_result, qauuid, dauuid_list, no_score_val=0.0):
    r"""
    Generate embeddings using the Pose-Invariant Embedding (PIE)
    Args:
        ibs (IBEISController): IBEIS / WBIA controller object
        aid_list  (int): annot ids specifying the input
        query_config_dict: ID configuration dict passed to ibs.query_chips_graph
    Example:
        >>> # ENABLE_DOCTEST
        >>> # Note that this is a very expensive test that loads an 1879-image dataset
        >>> import wbia_blend
        >>> from wbia_blend import QUERY_CONFIG_DICT_DICT
        >>> # use PIE sample data
        >>> import wbia_pie_v2
        >>> from wbia_pie_v2._plugin import DEMOS, CONFIGS, MODELS
        >>> species = 'whale_grey'
        >>> test_ibs = wbia_pie_v2._plugin.wbia_pie_v2_test_ibs(DEMOS[species], species, 'test2021')
        >>> aid_list = test_ibs.get_valid_aids(species=species)
        >>> test_names = ['33dfd0d7-ce13-43b7-9c53-a4a1e4fa69d1', '740b1b08-5f33-4810-9fdf-26b0b27260cf']
        >>> test_aids = [aid for aid, name in zip(aid_list, names) if name in test_names]
        >>> test_auuids = test_ibs.get_annot_uuids(test_aids)
        >>> config = QUERY_CONFIG_DICT_DICT['PIE v2']
        >>> query_result = test_ibs.query_chips_graph(
        >>>        qaid_list=[test_aids[0]],
        >>>        daid_list=test_aids,
        >>>        query_config_dict=config,
        >>>        echo_query_params=False,
        >>>        cache_images=False,
        >>>        n=1,
        >>>    )
        >>> score_array = get_score_array(query_result, test_auuids[0], test_auuids)
        >>> expected_array = np.array([
        >>>     0.00000000e+00, 5.86341648e-04, 5.86341648e-04, 5.86341648e-04, 5.86341648e-04,
        >>>     5.86341648e-04, 3.10301033e-05, 3.10301033e-05, 3.10301033e-05
        >>> ])
        >>> assert all(abs(expected_array - score_array) < 1e-7)
    """
    result_dict = query_result['cm_dict'][str(qauuid)]
    result_dauuids = result_dict['dannot_uuid_list']
    result_scores = result_dict['annot_score_list']
    assert len(result_dauuids) == len(result_scores)
    dauid_scores = {dauuid: score for dauuid, score in zip(result_dauuids, result_scores)}
    score_array = [dauid_scores.get(dauuid, no_score_val) for dauuid in dauuid_list]
    score_array = np.array(score_array)
    score_array[np.isneginf(score_array)] = 0.0  # replace missing scores with zero

    return score_array


def score_matrix_to_topk(score_matrix, truth_matrix):
    accuracy_lists = []
    for i in range(len(score_matrix)):
        score_list = score_matrix[i]
        truth_list = truth_matrix[i]
        accuracy_lists.append(score_list_to_accuracy_list(score_list, truth_list))

    accuracy_lists = np.array(accuracy_lists)
    rank_ks = np.sum(accuracy_lists, axis=0)
    accuracy_k = rank_ks / len(rank_ks)

    return accuracy_k


def score_list_to_accuracy_list(score_list, truth_list, max_k=20):
    if max_k is None:
        max_k = len(score_list)
    # position k of accuracy list is 0 if no match, 1 if match exists at rank <= k
    score_truth_list = sorted(zip(score_list, truth_list), reverse=True)
    accuracy_list = [score_truth_list[0][1]]
    for i in range(1, max_k):
        accuracy_list.append(accuracy_list[i - 1] or score_truth_list[i][1])

    accuracy_list = [int(x) for x in accuracy_list]
    return accuracy_list


def top_1_acc(score_matrices, alg_1_weight, truth_matrix):
    """
    Calculate the top-1 accuracy of a weighted average of two score matrices
    Args:
        score_matrices: list of two n x n score matrices from matching n annotations against each other
        alg_1_weight  (float): weight for the first set of scores; alg_2_weight = 1 - alg_1_weight
        truth_matrix: n x n boolean matrix labeling when the ground truth is a match

    """
    assert len(score_matrices) == 2, 'Can only optimize weights between two algorithms'
    top_k_acc = combined_weighted_accuracy(
        score_matrices, [alg_1_weight, 1 - alg_1_weight], truth_matrix
    )
    return top_k_acc[0]


def combined_weighted_accuracy(score_matrices, weights, truth_matrix):
    """
    score_matrices: list of n x n matrices
    weights: list of weights
    truth_matrix: n x n matrix
    """
    assert len(score_matrices) == len(weights), 'len(score_matrices) != len(weights)'
    # TODO: numpythonic / linear algebraic way to do this. it's like matrix mult but score_matrices is a tensor [3-indexed matrix]
    weighted_matrices = [
        score_matrix * weight for score_matrix, weight in zip(score_matrices, weights)
    ]
    combined_matrix = np.array(weighted_matrices).sum(axis=0)
    top_k = score_matrix_to_topk(combined_matrix, truth_matrix)
    return top_k


def optimize_weights(score_matrices, truth_matrix):
    """
    Fine the weights that generate the best blended accuracy for score_matrices
    Args:
        score_matrices: list of n x n score matrices from matching n annotations against each other
        truth_matrix: n x n boolean matrix labeling when the ground truth is a match
    """
    assert len(score_matrices) == 2, 'Can only optimize weights between two algorithms'
    candidate_weights = np.linspace(0, 1, num=50)
    top_1_accs = [
        top_1_acc(score_matrices, weight, truth_matrix) for weight in candidate_weights
    ]
    print('top_1_accs:', top_1_accs)
    best_weight = candidate_weights[np.argmax(top_1_accs)]
    print(f'best accuracy: {np.max(top_1_accs)} with alg_1 weight: {best_weight}')
    return best_weight
