import itertools as it
import pytest


@pytest.mark.parametrize('directed, within', it.product(*[[True, False]] * 2))
def test_combinatorics(directed, within):
    """
    This test does not test any code in the repository but runs numerical checks on the
    combinatorics of counting contributions to the variance of aggregate relational data.
    """
    n1 = 11
    idx1 = range(n1)

    if within:
        idx1 = idx2 = range(n1)
    else:
        n2 = 13
        idx2 = range(n1, n1 + n2)

    counts = {}
    for indices in it.product(idx1, idx2, idx1, idx2):
        i, j, k, l = indices  # noqa: E741
        # Skip if necessary for directed and undirected graphs.
        if directed and (i == j or k == l):
            continue
        if not directed and (i >= j or k >= l):
            continue

        # Construct an index identifier reusing indices if they're already used.
        lookup = {
            i: 'i',
            j: 'j',
        }
        key = ''.join([lookup.get(v, k) for k, v in zip('ijkl', indices)])
        counts[key] = counts.get(key, 0) + 1

    if within:
        total = (n1 * (n1 - 1)) ** 2
        if directed:
            expected = {
                'ijij': n1 * (n1 - 1),
                'ijkl': n1 * (n1 - 1) * (n1 - 2) * (n1 - 3),
                'ijji': n1 * (n1 - 1),
            }
            for key in ['ijil', 'ijki', 'ijjl', 'ijkj']:
                expected[key] = n1 * (n1 - 1) * (n1 - 2)
        else:
            total /= 4
            expected = {
                'ijij': n1 * (n1 - 1) / 2,
                'ijkl': n1 * (n1 - 1) * (n1 - 2) * (n1 - 3) / 4,
                # the 'ijji' term does not appear because we require i<j and k<l.
            }
            # For the 'ijil' option, we have n choices for i, n - i choices for j, and n - i - 1
            # choices for l. Evaluating Factor[Sum[(n - i - 1) (n - i), {i, 1, n}]] gives.
            expected['ijil'] = expected['ijkj'] = n1 * (n1 - 1) * (n1 - 2) / 3
            # For the 'ijjl' option we have half that number of choices.
            expected['ijjl'] = expected['ijki'] = n1 * (n1 - 1) * (n1 - 2) / 6
    else:
        total = (n1 * n2) ** 2
        expected = {
            'ijij': n1 * n2,
            'ijil': n1 * n2 * (n2 - 1),
            'ijkj': n1 * (n1 - 1) * n2,
            'ijkl': n1 * (n1 - 1) * n2 * (n2 - 1),
        }

    assert counts == expected
    assert total == sum(counts.values())
