import logging
import numpy as np


def compute_ap_cmc(index, good_index, junk_index):
    """ Compute AP and CMC for each sample
    """
    ap = 0
    cmc = np.zeros(len(index)) 
    
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        ap = ap + d_recall*precision

    return ap, cmc


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids):
    """ Compute CMC and mAP

    Args:
        distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
    """
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        logger = logging.getLogger('reid.evaluate')
        logger.info("{} query samples do not have groundtruth.".format(num_no_gt))

    CMC = CMC / (num_q - num_no_gt)
    mAP = AP / (num_q - num_no_gt)

    return CMC, mAP


def evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, mode='CC'):
    """ Compute CMC and mAP with clothes

    Args:
        distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
        q_clothids (numpy array): clothes IDs for query samples.
        g_clothids (numpy array): clothes IDs for gallery samples.
        mode: 'CC' for clothes-changing; 'SC' for the same clothes.
    """
    assert mode in ['CC', 'SC']
    
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        cloth_index = np.argwhere(g_clothids==q_clothids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if mode == 'CC':
            good_index = np.setdiff1d(good_index, cloth_index, assume_unique=True)
            # remove gallery samples that have the same (pid, camid) or (pid, clothid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.intersect1d(query_index, cloth_index)
            junk_index = np.union1d(junk_index1, junk_index2)
        else:
            good_index = np.intersect1d(good_index, cloth_index)
            # remove gallery samples that have the same (pid, camid) or 
            # (the same pid and different clothid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.setdiff1d(query_index, cloth_index)
            junk_index = np.union1d(junk_index1, junk_index2)

        if good_index.size == 0:
            num_no_gt += 1
            continue
    
        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        logger = logging.getLogger('reid.evaluate')
        logger.info("{} query samples do not have groundtruth.".format(num_no_gt))

    if (num_q - num_no_gt) != 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0

    return CMC, mAP