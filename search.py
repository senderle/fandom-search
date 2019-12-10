import multiprocessing
import datetime
import csv
import os
import re
import sys
import random
from operator import itemgetter
from collections import defaultdict

import numpy
import nearpy
import spacy
from Levenshtein import distance as lev_distance

_SPACY_MODEL = None

# Approximate nearest neighbors search settings:

new_record_structure = {
    'fields': ['FAN_WORK_FILENAME',
               'FAN_WORK_WORD_INDEX',
               'FAN_WORK_WORD',
               'FAN_WORK_ORTH_ID',
               'ORIGINAL_SCRIPT_WORD_INDEX',
               'ORIGINAL_SCRIPT_WORD',
               'ORIGINAL_SCRIPT_ORTH_ID',
               'ORIGINAL_SCRIPT_CHARACTER',
               'ORIGINAL_SCRIPT_SCENE',
               'BEST_MATCH_DISTANCE',
               'BEST_LEVENSHTEIN_DISTANCE',
               'BEST_COMBINED_DISTANCE',
               ],
    'types': [str, int, str, int, int, str,
              int, str, int, float, int, float
              ]
}


def get_spacy_model():
    global _SPACY_MODEL
    if _SPACY_MODEL is None:
        _SPACY_MODEL = spacy.load('en_core_web_md',
                                  disable=['parser', 'tagger', 'ner'])
    return _SPACY_MODEL

def sp_parse_chunks(txt, size=100000):
    spacy_model = get_spacy_model()

    start = 0
    if len(txt) < 100000:
        yield spacy_model(txt)
        return

    while start < len(txt):
        end = start + 100000
        if end > len(txt):
            end = len(txt)
        else:
            while txt[end] != ' ':
                end -= 1
        yield spacy_model(txt[start: end])
        start = end + 1

def mk_vectors(sp_txt):
    # Given a text, parse it into `spacy`'s native format,
    # and produce a sequence of vectors, one per token.

    rows = len(sp_txt)
    cols = len(sp_txt[0].vector) if rows else 0

    vectors = numpy.empty((rows, cols), dtype=float)
    for i, word in enumerate(sp_txt):
        if word.has_vector:
            vectors[i] = word.vector
        else:
            # `spacy` doesn't have a pre-trained vector for this word,
            # so give it a unique random vector.
            w_str = str(word)
            vectors[i] = 0
            vectors[i][hash(w_str) % cols] = 1.0
            vectors[i][hash(w_str * 2) % cols] = 1.0
            vectors[i][hash(w_str * 3) % cols] = 1.0
    return vectors

def build_lsh_engine(orig, window_size, number_of_hashes, hash_dimensions):
    # Build the ngram vectors using rolling windows.
    # Variables named `*_win_vectors` contain vectors for
    # the given input, such that each row is the vector
    # for a single window. Successive windows overlap
    # at all words except for the first and last.

    orig_vectors = mk_vectors(orig)
    orig_win_vectors = numpy.array([orig_vectors[i:i + window_size, :].ravel()
                                   for i in range(orig_vectors.shape[0] - window_size + 1)])

    # Initialize the approximate nearest neighbor search algorithm.
    # This creates the search "engine" and populates its index with
    # the window-vectors from the original script. We can then pass
    # over the window-vectors from a fan work, taking each vector
    # and searching for good matches in the engine's index of script
    # text.

    # We could do the search in the opposite direction, storing
    # fan text in the engine's index, and passing over window-
    # vectors from the original script, searching for matches in
    # the index of fan text. Unfortuantely, the quality of the
    # matches found goes down when you add too many values to the
    # engine's index.
    vector_dim = orig_win_vectors.shape[1]

    hashes = []
    for i in range(number_of_hashes):
        h = nearpy.hashes.RandomBinaryProjections('rbp{}'.format(i),
                                                  hash_dimensions)
        hashes.append(h)

    engine = nearpy.Engine(vector_dim,
                           lshashes=hashes,
                           distance=nearpy.distances.CosineDistance())

    for ix, row in enumerate(orig_win_vectors):
        engine.store_vector(row, (ix, str(orig[ix: ix + window_size])))
    return engine

def multi_search_wrapper(work):
    result = _ANN_INDEX.search(work)
    return result

class AnnIndexSearch(object):
    def __init__(self, original_script_filename, window_size,
                 number_of_hashes, hash_dimensions, distance_threshold):
        orig_csv = load_markup_script(original_script_filename)
        orig_csv = orig_csv[1:]  # drop header
        orig_csv = [[i] + r for i, r in enumerate(orig_csv)]
        # [['ORIGINAL_SCRIPT_INDEX',
        #   'LOWERCASE',
        #   'SPACY_ORTH_ID',
        #   'SCENE',
        #   'CHARACTER']]

        (self.word_index,
         self.word_lowercase,
         self.orth_id,
         self.scene,
         self.character) = zip(*orig_csv)

        self.window_size = window_size
        self.distance_threshold = distance_threshold
        self.spacy_model = get_spacy_model()
        orig_doc = spacy.tokens.Doc(self.spacy_model.vocab, self.word_lowercase)
        self.engine = build_lsh_engine(orig_doc, window_size,
                                       number_of_hashes, hash_dimensions)
        self.reset_stats()

    def reset_stats(self):
        self._windows_processed = 0

    @property
    def windows_processed(self):
        return self._windows_processed

    def search(self, filename):
        with open(filename, encoding='utf8') as fan_file:
            fan = fan_file.read()
            fan = [t for ch in sp_parse_chunks(fan) for t in ch if not t.is_space]

        # Create the fan windows:
        fan_vectors = mk_vectors(fan)
        fan_win_vectors = numpy.array(
            [fan_vectors[i:i + self.window_size, :].ravel()
             for i in range(fan_vectors.shape[0] - self.window_size + 1)]
        )

        duplicate_records = defaultdict(list)
        for fan_ix, row in enumerate(fan_win_vectors):
            self._windows_processed += 1
            results = self.engine.neighbours(row)

            # Extract data about the original script
            # embedded in the engine's results.
            results = [(match_ix, match_str, distance)
                       for vec, (match_ix, match_str), distance in results
                       if distance < self.distance_threshold]

            # Create a new record with original script
            # information and fan work information.
            for match_ix, match_str, distance in results:
                fan_context = str(fan[fan_ix: fan_ix + self.window_size])
                lev_d = lev_distance(match_str, fan_context)

                for window_ix in range(self.window_size):
                    fan_word_ix = fan_ix + window_ix
                    fan_word = fan[fan_word_ix].orth_
                    fan_orth_id = fan[fan_word_ix].orth

                    orig_word_ix = match_ix + window_ix
                    orig_word = self.word_lowercase[orig_word_ix]
                    orig_orth_id = self.orth_id[orig_word_ix]
                    char = self.character[orig_word_ix]
                    scene = self.scene[orig_word_ix]

                    duplicate_records[(filename, fan_word_ix)].append(
                        # NOTE: This **must** match the definition
                        #       of `record_structure` above
                        [filename,
                         fan_word_ix,
                         fan_word,
                         fan_orth_id,
                         orig_word_ix,
                         orig_word,
                         orig_orth_id,
                         char,
                         scene,
                         distance,
                         lev_d,
                         distance * lev_d]
                    )

        # To deduplicate duplicate_records, we
        # pick the single best match, as measured by
        # the combined distance for the given n-gram
        # match that first identified the word.
        for k, dset in duplicate_records.items():
            duplicate_records[k] = min(dset, key=itemgetter(11))
        return sorted(duplicate_records.values())

def validate_markup_script(filename,
                           interactive=False,
                           _unbalanced_l=re.compile('<<[^>]*<<'),
                           _unbalanced_r=re.compile('>>[^<]*>>'),
                           _tags=re.compile('>>\s*([^<]*)\s*<<')):
    with open(filename, encoding='utf-8') as ip:
        script = ip.read()

    print('Checking script for markup errors.')
    print()

    errs = False
    unbal_l = _unbalanced_l.findall(script)
    if unbal_l:
        print('Unbalanced left tag delimiters:')
        for m in _unbalanced_l.finditer(script):
            line = script[:m.start() + 1].count('\n') + 1
            print('  On line {}'.format(line))
            print('    {}'.format(m.group().strip()))
        errs = True
        print()

    unbal_r = _unbalanced_r.findall(script)
    if unbal_r:
        print('Unbalanced right tag delimiters:')
        for m in _unbalanced_r.finditer(script):
            line = script[:m.start() + 1].count('\n') + 1
            print('  On line {}'.format(line))
            print('    {}'.format(m.group().strip()))
        errs = True
        print()

    tag_set = set(t.strip() for t in _tags.findall(script))
    expected_tags = set(('LINE', 'DIRECTION', 'SCENE_NUMBER', 'SCENE_DESCRIPTION', 'CHARACTER_NAME'))
    if tag_set - expected_tags:
        print('Unexpected tag labels:')
        for m in _tags.finditer(script):
            if m.group(1).strip() not in expected_tags:
                line = script[:m.start(1) + 1].count('\n') + 1
                print('  On line {}'.format(line))
                print('    {}'.format(m.group(1).strip()))
        errs = True
        print()

    if not errs:
        print('No markup errors found.')
        return True
    elif interactive and errs:
        print('Errors were found in the script markup. Do you want to continue? (Default is no.)')
        print()
        r = ''
        while r.lower() not in ('y', 'yes', 'n', 'no'):
            r = input('Enter y for yes or n for no: ')
            if not r.strip():
                r = 'n'
        return r.lower() in ('y', 'yes')
    else:
        return False

def validate_cmd(args):
    return validate_markup_script(args.script)

def load_markup_script(filename,
                        _line_rex=re.compile('LINE<<(?P<line>[^>]*)>>'),
                        _scene_rex=re.compile('SCENE_NUMBER<<(?P<scene>[^>]*)>>'),
                        _char_rex=re.compile('CHARACTER_NAME<<(?P<character>[^>]*)>>')):

    with open(filename, encoding='utf-8') as ip:
        spacy_model = get_spacy_model()

        current_scene = None
        current_scene_count = 0
        current_scene_error_fix = False
        current_char = None
        rows = [['LOWERCASE', 'SPACY_ORTH_ID', 'SCENE', 'CHARACTER']]
        for i, line in enumerate(ip):
            if _scene_rex.search(line):
                current_scene_count += 1
                scene_string = _scene_rex.search(line).group('scene')
                scene_string = ''.join(c for c in scene_string
                                       if c.isdigit())
                try:
                    scene_int = int(scene_string)
                    current_scene = scene_int
                except ValueError:
                    current_scene_error_fix = True
                    print("Error in Scene markup: {}".format(line))

                if current_scene_error_fix:
                    current_scene = current_scene_count

            elif _char_rex.search(line):
                current_char = _char_rex.search(line).group('character')
            elif _line_rex.search(line):
                tokens = spacy_model(_line_rex.search(line).group('line'))
                tokens = [t for t in tokens if not t.is_space]
                for t in tokens:
                    # original Spacy lexeme object can be recreated using
                    #     spacy.lexeme.Lexeme(get_spacy_model().vocab, t.orth)
                    row = [t.lower_, t.lower, current_scene, current_char]
                    rows.append(row)
    return rows

def write_records(records, filename):
    with open(filename, 'w', encoding='utf-8') as out:
        wr = csv.writer(out)
        wr.writerows(records)

def analyze(args,
            window_size=6,
            number_of_hashes=15,  # Bigger -> slower (linear), more matches
            hash_dimensions=14,   # Bigger -> faster (???), fewer matches
            distance_threshold=0.1,
            chunk_size=500
            ):
    fan_work_directory = args.fan_works
    original_script_markup = args.script
    subsample_start = 0 if args.skip_works < 0 else args.skip_works
    subsample_end = (None if args.num_works < 0 else 
                     args.num_works + subsample_start)

    fan_works = os.listdir(fan_work_directory)
    fan_works = [os.path.join(fan_work_directory, f)
                 for f in fan_works]

    # This will always generate the same "random" sample.
    random.seed(4815162342)
    random.shuffle(fan_works)

    # Optionally skip ahead in the list or stop early.
    fan_works = fan_works[subsample_start:subsample_end]

    start = 0
    fan_clusters = [fan_works[i:i + chunk_size]
                    for i in range(start, len(fan_works), chunk_size)]

    filename_base = 'match-{}gram{{}}'.format(window_size)
    batch_filename = filename_base.format('-batch-{}.csv')

    accumulated_records = [new_record_structure['fields']]
    ann_index = AnnIndexSearch(original_script_markup,
                               window_size,
                               number_of_hashes,
                               hash_dimensions,
                               distance_threshold)

    for i, fan_cluster in enumerate(fan_clusters, start=start):
        print('Processing cluster {} ({}-{})'.format(i,
                                                     chunk_size * i,
                                                     chunk_size * (i + 1)))

        global _ANN_INDEX
        _ANN_INDEX = ann_index
        with multiprocessing.Pool(processes=4, maxtasksperchild=10) as pool:
            record_sets = pool.map(
                multi_search_wrapper,
                fan_cluster,
                chunksize=chunk_size // (4 * pool._processes))
            records = [r for r_set in record_sets for r in r_set]
            write_records(records, batch_filename.format(i))
            accumulated_records.extend(records)

    i = 0
    today_str = '-{:%Y%m%d}.csv'.format(datetime.date.today())
    name_check = filename_base.format(today_str)
    while os.path.exists(name_check):
        i += 1
        today_str = '-{:%Y%m%d}-{}.csv'.format(datetime.date.today(), i)
        name_check = filename_base.format(today_str)

    write_records(accumulated_records,
                  name_check)
