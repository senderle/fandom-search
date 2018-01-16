
# coding: utf-8

import spacy
sp = spacy.load('en')

import re
import os
import random
import csv
import multiprocessing
import datetime
from collections import Counter, defaultdict
from operator import itemgetter

import numpy
import pandas
import nearpy
from Levenshtein import distance as lev_distance

### Script Settings ###
# Modify these to change input files and other parameters.

# Input filenames:
original_script_markup = os.path.join('.','original-scripts','markup-script.txt')
fan_work_directory = 'plaintext'

# Set N-Gram window size:
window_size = 10
window_size = 6

# Set cosine distance matching threshold:
distance_threshold = 0.25
distance_threshold = 0.1

# Set approximate nearest neighbor parameters:
number_of_hashes = 15  # Bigger -> slower (linear), more matches
hash_dimensions = 14   # Bigger -> faster (???), fewer matches

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

# Utility functions
def mk_vectors(sp_txt):
    """Given a parsed text in `spacy`'s native format, produce
    a sequence of vectors, one per token.
    """
 
    rows = len(sp_txt)
    cols = len(sp_txt[0].vector) if rows else 0

    vectors = numpy.empty((rows, cols), dtype=float)
    for i, word in enumerate(sp_txt):
        if word.has_vector:
            vectors[i] = word.vector
        else:
            # It seems `spacy` doesn't have a pre-trained vector for
            # this word. So we do something pretty dumb here to give
            # the word a vector that is unique to that word and not
            # too similar to other words.
            w_str = str(word)
      #      print(hash(w_str))
      #      print(cols)
            vectors[i] = 0
            vectors[i][hash(w_str) % cols] = 1.0
            vectors[i][hash(w_str * 2) % cols] = 1.0
            vectors[i][hash(w_str * 3) % cols] = 1.0
    return vectors

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

# TODO: Allow windows to be built over non-contiguous portions of text,
#       (to allow dropping of punctuation, etc) while preserving index
#       information for each word. This will require including a new
#       parameter that provides the correct index for each word in orig
# TODO: Allow input to be pre-parsed using spacy. This will involve 
#       capturing the spacy orth_id and using it to recreate the lexeme
#       and vetor data.
# TODO: Make this function more "format aware." It will simplify things
#       to accept more coupling here. This is a function for processing
#       scripts in exactly one format.
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

def find_matches_multi(fan_works, ann_index, pool):
    chunksize = len(fan_works) // (4 * pool._processes)
    record_sets = pool.map(ann_index.search, fan_works, chunksize=chunksize)
    records = []
    records.extend(r for r_set in record_sets for r in r_set)
    return records

#new function - return list of map of ann_index.search, fan words
def find_matches(fan_works, ann_index, pool):
    record_sets = map(ann_index.search, fan_works)
    records = []
    records.extend(r for r_set in record_sets for r in r_set)
    return records

class AnnIndexSearch(object):
    def __init__(self, original_script_filename, window_size,
                 number_of_hashes, hash_dimensions, distance_threshold):
        orig_csv = load_markup_script(original_script_filename)
        orig_csv = orig_csv[1:]  # drop header
        orig_csv = [[i] + r for i, r in enumerate(orig_csv)]
        # resulting csv format: 
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
        orig_doc = spacy.tokens.Doc(sp.vocab, self.word_lowercase)
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
            fan = sp(fan_file.read())
    
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
                fan_context = str(fan[fan_ix: fan_ix + window_size])
                lev_d = lev_distance(match_str, fan_context)
                
                for window_ix in range(window_size):
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
    df = pandas.DataFrame(match_cols,
                          index = range(maxn + 1),
                          columns=col_names)
    df.index.name = 'Word index in original script'
    df = df.loc[start:end]
    df.plot(figsize=figsize, colormap=colormap, legend=legend)

def load_markup_script(filename,
                        _line_rex=re.compile('LINE<<(?P<line>[^>]*)>>'),
                        _scene_rex=re.compile('SCENE_NUMBER<<(?P<scene>[^>]*)>>'),
                        _char_rex=re.compile('CHARACTER_NAME<<(?P<character>[^>]*)>>')):
    with open(filename, encoding='utf-8') as ip:
        current_scene = None
        current_char = None
        current_line = None
        rows = [['LOWERCASE', 'SPACY_ORTH_ID', 'SCENE', 'CHARACTER']]
        for i, line in enumerate(ip):
            if _scene_rex.search(line):
                current_scene = int(_scene_rex.search(line).group('scene'))
            elif _char_rex.search(line):
                current_char = _char_rex.search(line).group('character')
            elif _line_rex.search(line):
                tokens = sp(_line_rex.search(line).group('line'))
                for t in tokens:
                    # original Spacy lexeme object can be recreated using
                    #     spacy.lexeme.Lexeme(sp.vocab, t.orth)
                    # where `sp = spacy.load('en')`
                    row = [t.lower_, t.lower, current_scene, current_char]
                    rows.append(row)
    return rows
    
def write_records(records, filename):
    with open(filename, 'w', encoding='utf-8') as out:
        wr = csv.writer(out)
        wr.writerows(records)
    
### SCRIPT ###
if __name__ == '__main__':

    fan_works = os.listdir(fan_work_directory)
    fan_works = [os.path.join(fan_work_directory, f) 
                 for f in fan_works]   
    random.seed(4815162342)  # This will always generate the same "random" sample.
    random.shuffle(fan_works)
  
    cluster_size = 500
    start = 0
    fan_clusters = [fan_works[i:i + cluster_size] for i in range(start, len(fan_works), cluster_size)]
    #fan_clusters = fan_clusters[:1]
    
    filename_base = 'match-{}gram{{}}'.format(window_size)
    batch_filename = filename_base.format('-batch-{}.csv')
    
    accumulated_records = [new_record_structure['fields']]
    for i, fan_cluster in enumerate(fan_clusters, start=start):
        with multiprocessing.Pool(processes=5) as pool:
            ann_index = AnnIndexSearch(original_script_markup, 
                                       window_size, 
                                       number_of_hashes, 
                                       hash_dimensions,
                                       distance_threshold)
            #records = find_matches_multi(fan_cluster, ann_index, pool)
            records = find_matches(fan_cluster, ann_index, pool)
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