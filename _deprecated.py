def cosine_distance(row_values, col_values):
    """Calculate the cosine distance between two vectors. Also
    accepts matrices and 2-d arrays, and calculates the
    distances over the cross product of rows and columns.
    """
    verr_msg = '`cosine_distance` is not defined for {}-dimensional arrays.'
    if len(row_values.shape) == 1:
        row_values = row_values[None,:]
    elif len(row_values.shape) != 2:
        raise ValueError(verr_msg.format(len(row_values.shape)))

    if len(col_values.shape) == 1:
        col_values = col_values[:,None]
    elif len(col_values.shape) != 2:
        raise ValueError(verr_msg.format(len(col_values.shape)))

    row_norm = (row_values * row_values).sum(axis=1) ** 0.5
    row_norm = row_norm[:,None]

    col_norm = (col_values * col_values).sum(axis=0) ** 0.5
    col_norm = col_norm[None,:]

    result = row_values @ col_values
    result /= row_norm
    result /= col_norm
    return 1 - result

def make_match_strata(records, record_structure, num_strata, max_threshold):
    combined_ix = record_structure['fields'].index('BEST_COMBINED_DISTANCE')
    low = [i / num_strata * max_threshold
           for i in range(0, num_strata)]
    high = [i / num_strata * max_threshold
            for i in range(1, num_strata + 1)]
    ranges = zip(low, high)

    return [[r for r in records[1:]
             if r[combined_ix] >= low and r[combined_ix] < high]
            for low, high in ranges]

def label_match_strata(num_strata, max_threshold):
    high = [i / num_strata * max_threshold
            for i in range(1, num_strata + 1)]
    return ['Number of matches below threshold {:.2}'.format(h)
            for h in high]

def chart_match_strata(records,
                       num_strata=5, max_threshold=1,
                       start=1, end=None,
                       figsize=(15, 10),
                       colormap='plasma',
                       legend=True):
    match_strata = make_match_strata(records, new_record_structure, num_strata, max_threshold)

    cumulative_strata = [match_strata[0:i] for i in
                         range(len(match_strata), 0, -1)]
    match_counters = [Counter(row[4] for matches in strata for row in matches)
                      for strata in cumulative_strata]
    maxn = max(max(mc) for mc in match_counters if mc)
    match_cols = [[mc[n] for mc in match_counters]
                  for n in range(maxn + 1)]

    col_names = label_match_strata(num_strata, max_threshold)
    col_names.reverse()
    df = pd.DataFrame(match_cols,
                          index = range(maxn + 1),
                          columns=col_names)
    df.index.name = 'Word index in original script'
    df = df.loc[start:end]
    df.plot(figsize=figsize, colormap=colormap, legend=legend)

def most_frequent_matches(records, n_matches, threshold):
    ct = Counter(r[3] for r in records if r[-1] < threshold)
    ix_to_context = {r[3]: r[4] for r in records}
    matches = ct.most_common(n_matches)
    return [(i, c, ix_to_context[i])
            for i, c in matches]
    return matches
