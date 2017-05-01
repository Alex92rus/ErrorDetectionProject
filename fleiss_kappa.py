def fleiss_kappa(ratings, n):
    '''
    Computes the Fleiss' kappa measure for assessing the reliability of
    agreement between a fixed number n of raters when assigning categorical
    ratings to a number of items.

    Args:
        ratings: a list of (item, category)-ratings
        n: number of raters
    Returns:
        the Fleiss' kappa score

    Refactored implementation from:
        https://gist.github.com/ShinNoNoir/4749548

    See also:
        http://en.wikipedia.org/wiki/Fleiss'_kappa
    '''
    items = set()
    categories = set()
    n_ij = {}

    for i, c in ratings:
        items.add(i)
        categories.add(c)
        n_ij[(i, c)] = n_ij.get((i, c), 0) + 1

    N = len(items)

    p_j = {}
    for c in categories:
        p_j[c] = sum(n_ij.get((i, c), 0) for i in items) / (1.0 * n * N)

    P_i = {}
    for i in items:
        P_i[i] = (sum(n_ij.get((i, c), 0) ** 2 for c in categories) - n) / (n * (n - 1.0))

    P_bar = sum(iter(P_i.values())) / (1.0 * N)
    P_e_bar = sum(p_j[c] ** 2 for c in categories)

    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

    return kappa


example = ([(1, 5)] * 14 +
           [(2, 2)] * 2 + [(2, 3)] * 6 + [(2, 4)] * 4 + [(2, 5)] * 2 +
           [(3, 3)] * 3 + [(3, 4)] * 5 + [(3, 5)] * 6 +
           [(4, 2)] * 3 + [(4, 3)] * 9 + [(4, 4)] * 2 +
           [(5, 1)] * 2 + [(5, 2)] * 2 + [(5, 3)] * 8 + [(5, 4)] * 1 + [(5, 5)] * 1 +
           [(6, 1)] * 7 + [(6, 2)] * 7 +
           [(7, 1)] * 3 + [(7, 2)] * 2 + [(7, 3)] * 6 + [(7, 4)] * 3 +
           [(8, 1)] * 2 + [(8, 2)] * 5 + [(8, 3)] * 3 + [(8, 4)] * 2 + [(8, 5)] * 2 +
           [(9, 1)] * 6 + [(9, 2)] * 5 + [(9, 3)] * 2 + [(9, 4)] * 1 +
           [(10, 2)] * 2 + [(10, 3)] * 2 + [(10, 4)] * 3 + [(10, 5)] * 7)

