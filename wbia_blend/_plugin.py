# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import utool as ut
import vtool as vt
import wbia
from wbia import dtool as dt
from wbia.constants import ANNOTATION_TABLE, CONTAINERIZED, PRODUCTION, UNKNOWN  # NOQA
from wbia.control import controller_inject

from .train_blend import get_score_array

(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)

register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']


class PieTwoHotSpotterConfig(dt.Config):  # NOQA
    def get_param_info_list(self):
        return [
            ut.ParamInfo('pie_weight', 0.9795918367346939),
        ]


# windowdressing for ID api, copied from PIE v2
def get_match_results(depc, qaid_list, daid_list, score_list, config):
    """ converts table results into format for ipython notebook """
    # qaid_list, daid_list = request.get_parent_rowids()
    # score_list = request.score_list
    # config = request.config

    unique_qaids, groupxs = ut.group_indices(qaid_list)
    # grouped_qaids_list = ut.apply_grouping(qaid_list, groupxs)
    grouped_daids = ut.apply_grouping(daid_list, groupxs)
    grouped_scores = ut.apply_grouping(score_list, groupxs)

    ibs = depc.controller
    unique_qnids = ibs.get_annot_nids(unique_qaids)

    # scores
    _iter = zip(unique_qaids, unique_qnids, grouped_daids, grouped_scores)
    for qaid, qnid, daids, scores in _iter:
        dnids = ibs.get_annot_nids(daids)

        # Remove distance to self
        annot_scores = np.array(scores)
        daid_list_ = np.array(daids)
        dnid_list_ = np.array(dnids)

        is_valid = daid_list_ != qaid
        daid_list_ = daid_list_.compress(is_valid)
        dnid_list_ = dnid_list_.compress(is_valid)
        annot_scores = annot_scores.compress(is_valid)

        # Hacked in version of creating an annot match object
        match_result = wbia.AnnotMatch()
        match_result.qaid = qaid
        match_result.qnid = qnid
        match_result.daid_list = daid_list_
        match_result.dnid_list = dnid_list_
        match_result._update_daid_index()
        match_result._update_unique_nid_index()

        grouped_annot_scores = vt.apply_grouping(annot_scores, match_result.name_groupxs)
        name_scores = np.array([np.sum(dists) for dists in grouped_annot_scores])
        match_result.set_cannonical_name_score(annot_scores, name_scores)
        yield match_result


class PieTwoHotspotterRequest(dt.base.VsOneSimilarityRequest):
    _symmetric = False
    _tablename = 'PieTwoHotSpotter'

    @ut.accepts_scalar_input
    def get_fmatch_overlayed_chip(request, aid_list, overlay=True, config=None):
        depc = request.depc
        ibs = depc.controller
        chips = ibs.get_annot_chips(aid_list)
        return chips

    def render_single_result(request, cm, aid, **kwargs):
        # HACK FOR WEB VIEWER
        overlay = kwargs.get('draw_fmatches')
        chips = request.get_fmatch_overlayed_chip(
            [cm.qaid, aid], overlay=overlay, config=request.config
        )
        out_image = vt.stack_image_list(chips)
        return out_image

    def postprocess_execute(request, table, parent_rowids, rowids, result_list):
        qaid_list, daid_list = list(zip(*parent_rowids))
        score_list = ut.take_column(result_list, 0)
        depc = request.depc
        config = request.config
        cm_list = list(get_match_results(depc, qaid_list, daid_list, score_list, config))
        table.delete_rows(rowids)
        return cm_list

    def execute(request, *args, **kwargs):
        # kwargs['use_cache'] = False
        result_list = super(PieTwoHotspotterRequest, request).execute(*args, **kwargs)
        qaids = kwargs.pop('qaids', None)
        if qaids is not None:
            result_list = [result for result in result_list if result.qaid in qaids]
        return result_list


@register_preproc_annot(
    tablename='PieTwoHotSpotter',
    parents=[ANNOTATION_TABLE, ANNOTATION_TABLE],
    colnames=['score'],
    coltypes=[float],
    configclass=PieTwoHotSpotterConfig,
    requestclass=PieTwoHotspotterRequest,
    fname='pie_v2_hotspotter',
    rm_extern_on_delete=True,
    chunksize=None,
)
def wbia_plugin_pie_hotspotter_blend(depc, qaid_list, daid_list, config):
    """
    Fine the weights that generate the best blended accuracy for score_matrices
    Args:
        score_matrices: list of n x n score matrices from matching n annotations against each other
        truth_matrix: n x n boolean matrix labeling when the ground truth is a match
    """
    ibs = depc.controller
    pie_weight = config.get('pie_weight')
    qaids = list(set(qaid_list))
    daids = list(set(daid_list))
    assert len(qaids) == 1, 'Does not support multi-query matching'
    qauuid = ibs.get_annot_uuids(qaids)[0]
    dauuid_list = ibs.get_annot_uuids(daids)

    print('PIE-HS blend calculating PIE scores')
    try:
        result = ibs.query_chips_graph(
            qaid_list=qaids,
            daid_list=daids,
            query_config_dict={'pipeline_root': 'PieTwo'},
            echo_query_params=False,
            cache_images=False,
            # n=1,
            n=1,
        )
        # each entry out of depc.get is a tuple
        pie_scores = get_score_array(result, qauuid, dauuid_list)
    except KeyError:
        raise Exception('Pie-HotSpotter Blend called without Pie v2 enabled on wbia')

    print('PIE-HS blend Getting HS scores')
    try:
        result = ibs.query_chips_graph(
            qaid_list=qaids,
            daid_list=daids,
            query_config_dict={'sv_on': True},
            echo_query_params=False,
            cache_images=False,
            n=1,
        )
        hotspotter_scores = get_score_array(result, qauuid, dauuid_list)
    except Exception:
        raise Exception('Pie-HotSpotter Blend encountered error on HS scores')

    print('PIE-HS blend Blending scores')
    hotspotter_weight = 1 - pie_weight
    blended_scores = pie_weight * pie_scores + hotspotter_weight * hotspotter_scores
    # This just makes scores more readable
    blended_scores = 1000 * blended_scores

    for daid, blended_score in zip(daid_list, list(blended_scores)):
        yield (blended_score,)
