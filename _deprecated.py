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

# ----------------
# matrix functions
# ----------------

def add_matrix_subparser(subparsers):
    # Create n-gram matrices (deprecated)
    matrix_parser = subparsers.add_parser('matrix', help='deduplicates and builds matrix for best n-gram matches')
    matrix_parser.add_argument('i', action='store', help='input csv file')
    matrix_parser.add_argument('m', action = 'store', help='fandom/movie name for output file prefix')
    matrix_parser.add_argument('-n', action='store', default=6, help='n-gram size, default is 6-grams')
    matrix_parser.set_defaults(func=process)

class StrictNgramDedupe(object):
    def __init__(self, data_path, ngram_size):
        self.ngram_size = ngram_size

        with open(data_path, encoding='UTF8') as ip:
            rows = list(csv.DictReader(ip))
        self.data = rows
        self.work_matches = collections.defaultdict(list)

        for r in rows:
            self.work_matches[r['FAN_WORK_FILENAME']].append(r)

        # Use n-gram starting index as a unique identifier.
        self.starts_counter = collections.Counter(
            start
            for matches in self.work_matches.values()
            for start in self.to_ngram_starts(self.segment_full(matches))
        )

        filtered_matches = [self.top_ngram(span)
                            for matches in self.work_matches.values()
                            for span in self.segment_full(matches)]

        self.filtered_matches = [ng for ng in filtered_matches
                                 if self.no_better_match(ng)]

    def num_ngrams(self):
        return len(set(int(ng[0]['ORIGINAL_SCRIPT_WORD_INDEX'])
                       for ng in self.filtered_matches))

    def match_to_phrase(self, match):
        return ' '.join(m['ORIGINAL_SCRIPT_WORD'].lower() for m in match)

    def write_match_work_count_matrix(self, out_filename):
        ngrams = {}
        works = set()
        cells = collections.defaultdict(int)
        for m in self.filtered_matches:
            phrase = self.match_to_phrase(m)
            ix = int(m[0]['ORIGINAL_SCRIPT_WORD_INDEX'])
            filename = m[0]['FAN_WORK_FILENAME']

            ngrams[phrase] = ix
            works.add(filename)
            cells[(filename, phrase)] += 1

        ngrams = sorted(ngrams, key=ngrams.get)
        works = sorted(works)
        rows = [[cells[(fn, ng)] for ng in ngrams]
                for fn in works]
        totals = [sum(r[col] for r in rows) for col in range(len(rows[0]))]

        header = ['FILENAME'] + ngrams
        totals = ['(total)'] + totals
        rows = [[fn] + r for fn, r in zip(works, rows)]
        rows = [header, totals] + rows

        with open(out_filename, 'w', encoding='utf-8') as op:
            csv.writer(op).writerows(rows)

    def write_match_sentiment(self, out_filename):
        phrases = {}
        for m in self.filtered_matches:
            phrase = self.match_to_phrase(m)
            ix = int(m[0]['ORIGINAL_SCRIPT_WORD_INDEX'])
            phrases[phrase] = ix
        sorted_phrases = sorted(phrases, key=phrases.get)

        phrase_indices = [phrases[p] for p in sorted_phrases]
        phrases = sorted_phrases

        if emolex:
            emo_count = [emolex.lex_count(p) for p in phrases]
            emo_sent_count = self.project_sentiment_keys(emo_count,
                                                         ['NEGATIVE', 'POSITIVE'])
            emo_emo_count = self.project_sentiment_keys(emo_count,
                                                        ['ANTICIPATION',
                                                         'ANGER',
                                                         'TRUST',
                                                         'SADNESS',
                                                         'DISGUST',
                                                         'SURPRISE',
                                                         'FEAR',
                                                         'JOY'])
        if bing:
            bing_count = [bing.lex_count(p) for p in phrases]
            bing_count = self.project_sentiment_keys(bing_count,
                                                     ['NEGATIVE', 'POSITIVE'])

        if liwc:
            liwc_count = [liwc.lex_count(p) for p in phrases]
            liwc_sent_count = self.project_sentiment_keys(liwc_count,
                                                          ['POSEMO', 'NEGEMO'])
            liwc_other_keys = set(k for ct in liwc_count for k in ct.keys())
            liwc_other_keys -= set(['POSEMO', 'NEGEMO'])
            liwc_other_count = self.project_sentiment_keys(liwc_count,
                                                           liwc_other_keys)

        counts = []
        count_labels = []

        if emolex:
            counts.append(emo_emo_count)
            counts.append(emo_sent_count)
            count_labels.append('NRC_EMOTION_')
            count_labels.append('NRC_SENTIMENT_')

        counts.append(bing_count)
        count_labels.append('BING_SENTIMENT_')

        if liwc:
            counts.append(liwc_sent_count)
            counts.append(liwc_other_count)
            count_labels.append('LIWC_SENTIMENT_')
            count_labels.append('LIWC_ALL_OTHER_')

        rows = self.compile_sentiment_groups(counts, count_labels)

        for r, p, i in zip(rows, phrases, phrase_indices):
            r['{}-GRAM'.format(self.ngram_size)] = p
            r['{}-GRAM_START_INDEX'.format(self.ngram_size)] = i

        fieldnames = sorted(set(k for r in rows for k in r.keys()))
        totals = collections.defaultdict(int)
        skipkeys = ['{}-GRAM_START_INDEX'.format(self.ngram_size),
                    '{}-GRAM'.format(self.ngram_size)]
        totals[skipkeys[0]] = 0
        totals[skipkeys[1]] = '(total)'
        for r in rows:
            for k in r:
                if k not in skipkeys:
                    totals[k] += r[k]
        rows = [totals] + rows

        with open(out_filename, 'w', encoding='utf-8') as op:
            wr = csv.DictWriter(op, fieldnames=fieldnames)
            wr.writeheader()
            wr.writerows(rows)

    def project_sentiment_keys(self, counts, keys):
        counts = [{k: ct.get(k, 0) for k in keys}
                  for ct in counts]
        for ct in counts:
            if sum(ct.values()) == 0:
                ct['UNDETERMINED'] = 1
            else:
                ct['UNDETERMINED'] = 0

        return counts

    def compile_sentiment_groups(self, groups, prefixes):
        new_rows = []
        for group_row in zip(*groups):
            new_row = {}
            for gr, pf in zip(group_row, prefixes):
                for k, v in gr.items():
                    new_row[pf + k] = v
            new_rows.append(new_row)
        return new_rows

    def get_spans(self, indices):
        starts = [0]
        ends = []
        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:
                starts.append(i)
                ends.append(i)
        ends.append(len(indices))
        return list(zip(starts, ends))

    def segment_matches(self, matches, key):
        matches = sorted(matches, key=lambda m: int(m[key]))
        indices = [int(m[key]) for m in matches]
        return [[matches[i] for i in range(start, end)]
                for start, end in self.get_spans(indices)]

    def segment_fan_matches(self, matches):
        return self.segment_matches(matches, 'FAN_WORK_WORD_INDEX')

    def segment_orig_matches(self, matches):
        return self.segment_matches(matches, 'ORIGINAL_SCRIPT_WORD_INDEX')

    def segment_full(self, matches):
        return [orig_m
                for fan_m in self.segment_fan_matches(matches)
                for orig_m in self.segment_orig_matches(fan_m)
                if len(orig_m) >= self.ngram_size]

    def to_ngram_starts(self, match_spans):
        return [int(ms[i]['ORIGINAL_SCRIPT_WORD_INDEX'])
                for ms in match_spans
                for i in range(len(ms) - self.ngram_size + 1)]

    def start_count_key(self, span):
        def key(i):
            script_ix = int(span[i]['ORIGINAL_SCRIPT_WORD_INDEX'])
            return self.starts_counter.get(script_ix, 0)
        return key

    def no_better_match(self, ng):
        start = int(ng[0]['ORIGINAL_SCRIPT_WORD_INDEX'])
        best_start = max(range(start - self.ngram_size + 1,
                               start + self.ngram_size),
                         key=self.starts_counter.__getitem__)
        return start == best_start

    def top_ngram(self, span):
        start = max(
            range(len(span) - self.ngram_size + 1),
            key=self.start_count_key(span)
        )
        return span[start: start + self.ngram_size]


def process(inputs):
    ngram_size = inputs['n']
    in_file = inputs['i']
    out_prefix = inputs['m']

    matrix_out = '{}-most-common-perfect-matches-no-overlap-{}-gram-match-matrix.csv'.format(out_prefix, ngram_size)
    sentiment_out = '{}-most-common-perfect-matches-no-overlap-{}-gram-sentiment.csv'.format(out_prefix, ngram_size)

    dd = StrictNgramDedupe(in_file, ngram_size=ngram_size)
    #print(dd.num_ngrams())

    dd.write_match_work_count_matrix(matrix_out)
