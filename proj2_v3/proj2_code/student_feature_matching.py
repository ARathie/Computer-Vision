import numpy as np


def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    len1 = features1.shape[0]
    len2 = features2.shape[0]
    num_feats = features1.shape[1]
    dists = np.zeros((len1, len2))
    for m in range(len1):
        for n in range(len2):
            channel1_result = (features1[m][0] - features2[n][0])**2
            channel2_result = (features1[m][1] - features2[n][1])**2
            dists[m][n] = np.sqrt(channel1_result + channel2_result)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)
    - x1: A numpy array of shape (n,) containing the x-locations of features1
    - y1: A numpy array of shape (n,) containing the y-locations of features1
    - x2: A numpy array of shape (m,) containing the x-locations of features2
    - y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    matches = []
    confidences = []

    for inds1, fts1 in enumerate(features1):

        normed1 = np.linalg.norm(fts1)
        adj_fts1 = fts1 / (normed1 + 0.0001)

        match_list_temp = []

        for inds2, fts2 in enumerate(features2):
            normed2 = np.linalg.norm(fts2)
            adj_fts2 = fts2 / (normed2 + 0.0001)

            dist = np.sum(abs(adj_fts2 - adj_fts1))

            inds_concat = [inds1, inds2, dist]
            match_list_temp.append(inds_concat)

        match_list_temp = np.asarray(match_list_temp)
        numparr = match_list_temp[match_list_temp[:, 2].argsort()]
        if (3/4) * numparr[1, 2] > numparr[0, 2] :
            matches.append(numparr[0, 0:2])
            confidences.append(numparr[0, 2])

    temp_match = np.asarray(matches)
    temp_conf = np.asarray(confidences)

    temp_conf = np.expand_dims(temp_conf, axis=1)
    combined_results = np.concatenate((temp_match, temp_conf), axis=1)

    combined_results = combined_results[combined_results[:, 2].argsort()]

    matches = combined_results[:, 0:2].astype(int)
    confidences = combined_results[:, 2]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
