from inverted_index_gcp import InvertedIndex as InvGCP
from contextlib import closing
from collections import defaultdict
# from sklearn.preprocessing import MinMaxScaler

import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import pickle
from typing import List, Tuple


nltk.download('stopwords')

BLOCK_SIZE = 1999998
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

BASE_PATH = '/content/Drive/Shareddrives/ir_proj_resources'
CORPUS_SIZE = 6348910

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}

    def read(self, locs, n_bytes, postings_list_location):
        """
        This function was given by course staff, we added the postings_list_location as for where the posting_list
        will be for each inverted_index.
        """
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                self._open_files[f_name] = open(postings_list_location + f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def read_posting_list(inverted, w, posting_list_location):
    """
    This function was given by course staff
    """

    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, posting_list_location)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
            tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower())]


class QueriesCalc:
    def __init__(self, base_path: str = BASE_PATH):
        self.base_path = base_path
        self.body_inverted = None
        self.title_inverted = None
        self.anchor_inverted = None
        self.doc_len_d = None  # contains a dictionary of (wiki_id: doc length)
        self.doc_nf_d = None  # contains a dictionary of (wiki_id: nf score of doc) nf_doc is the sum of tf_idf square score of each term in the doc.
        self.doc_title_d = None  # contains a dictionary of (wiki_id: wiki title)
        self.page_rank_dataset = None  # data-frame of with two columns: doc_id, page_rank score. we use it to retrieve scores for a
        # list of wiki_ids fast
        self.page_rank_d = None  # contains a dictionary of (wiki_id: p_rank score)
        self.page_views_d = None  # contains a dictionary of (wiki_id: number of page views). if the wiki_id do not appear in the DB then we return 0.
        self.load_indices()
        self.load_resource()

    @staticmethod
    def get_token_per_query(query: str):
        """
        This function parse the query and return two lists, one with the unique query words (for fetching postings lists)
        and the other with the query words without stop words (for calculating query length).
        """
        token_list = tokenize(query)
        q_words = sorted([token for token in token_list if token not in all_stopwords])
        q_unique_words = sorted(list(set(q_words)))
        return q_words, q_unique_words

    def load_indices(self):
        """
        This function loads the 3 inverted_index from Disk to the class.
        """
        self.body_inverted = InvGCP.read_index(base_dir=f'{self.base_path}/postings_gcp/', name='index')
        self.title_inverted = InvGCP.read_index(base_dir=f'{self.base_path}/title_postings_gcp/', name='index')
        self.anchor_inverted = InvGCP.read_index(base_dir=f'{self.base_path}/anchor_postings_gcp/', name='index')

    def load_resource(self):
        """
        This function loads the class resources and pre-calculated dictionaries such as doc_len dict, page_rank dict and more
        """
        path = f'{self.base_path}/page_rank_data_set.csv.gz'
        self.page_rank_dataset = pd.read_csv(path, compression='gzip', header=None).rename({0: 'doc_id', 1: 'score'}, axis=1)
        self.page_rank_d = dict(self.page_rank_dataset.itertuples(index=False, name=None))
        self.doc_len_d = pd.read_pickle(f'{self.base_path}/doc_len_f.pkl')

        self.doc_title_d = pd.read_pickle(f'{self.base_path}/id_title_dict.pkl')

        with open(f'{self.base_path}/doc_nf_score_updated.pkl', 'rb') as fp:
            self.doc_nf_d = pickle.load(fp)

        self.page_views_d = pd.read_pickle(f'{self.base_path}/page_view.pkl')

    def get_wiki_titles(self, wiki_id_l: List[int]) -> List[Tuple[int, str]]:
        """
        This function gets a list of wiki ids and returns a list of tuples [(wiki_id, title)]
        """
        res = []
        for wiki_id in wiki_id_l:
            try:
                title = self.doc_title_d[wiki_id]
                res.append((wiki_id, title))
            except KeyError:
                pass
        return res
        # return list(self.doc_title_d[self.doc_title_d.index.isin(wiki_id_l)].itertuples(index=True, name=None))
        # return list(self.doc_title_d.loc[wiki_id_l].itertuples(index=True, name=None))

    def get_results_search_body(self, q_unique_words: List[str], q_len: int, k: int = 100, get_scores: bool = False):
        """
        This function contains the actual logic of search_body method. the function uses the body inverted index that contains
        word_count in the posting list and calculates the tf_idf score for each (term, doc) on the fly.
        :param q_unique_words: the unique query words that we fetch the posting lists for.
        :param q_len: query lengths (without stop words)
        :param k: number of results to return
        :param get_scores: if True the function returns the score and the wiki_id (and not just the wiki_id)
        :return:
        """
        sim_q_d = defaultdict(int)
        for w in q_unique_words:
            t_posting = read_posting_list(self.body_inverted, w, f'{self.base_path}/postings_gcp/')
            idf_w = np.log10(CORPUS_SIZE / self.body_inverted.df[w])
            for doc_id, wc in t_posting:
                sim_q_d[doc_id] += (wc/self.doc_len_d[int(doc_id)]) * idf_w
        updated_sim_s_q = defaultdict(int)
        for doc_id, sim_score in sim_q_d.items():
            # updated_sim_s_q[doc_id] = sim_score * (1 / q_len) * np.sqrt(self.doc_nf_d[str(doc_id)])
            updated_sim_s_q[doc_id] = sim_score * (1 / q_len) * self.doc_nf_d[str(doc_id)]
        res = sorted(updated_sim_s_q.items(), key=lambda item: item[1], reverse=True)[:k]
        wiki_ids = [tup[0] for tup in res]
        if get_scores:
            return res, wiki_ids
        return wiki_ids

    def get_page_rank_scores(self, wiki_id_l: List):
        """
        This function gets a list of wiki_id's and return their page rank scores.
        """
        res = []
        for wiki_id in wiki_id_l:
            try:
                score = self.page_rank_dataset[self.page_rank_dataset.doc_id == wiki_id]['score'].values[0]
                res.append(score)
            except Exception:
                res.append(0)
        return res

    def get_boolean_score_per_index(self, index_type: str, q_unique_words: List[str], get_scores: bool = False):
        """
        This function return the a list of results ordered by their boolean score for a given index ( title/anchor)
        :param index_type: anchor / title
        :param q_unique_words:
        :param get_scores: if True the function returns the score and the wiki_id (and not just the wiki_id)
        """
        if index_type == 'title':
            inv_idx = self.title_inverted
        else:
            inv_idx = self.anchor_inverted
        first = True
        for w in q_unique_words:
            t_posting = read_posting_list(inv_idx, w, f'{self.base_path}/{index_type}_postings_gcp/')
            if first:
                first = False
                res_df = pd.DataFrame(t_posting).set_index(0).rename(columns={1: w})
            else:
                t_res = pd.DataFrame(t_posting).set_index(0).rename(columns={1: w})
                res_df = res_df.join(t_res, how='outer')

        res_df[~res_df.isna()] = 1

        res_df.fillna(0, inplace=True)

        res_df['score'] = res_df.sum(axis=1)
        res_df = res_df.sort_values(by='score', ascending=False, inplace=False)
        if get_scores:
            scores = list(res_df[['score']].itertuples(name=None))
            return scores, res_df.index.tolist()
        else:
            return res_df.index.tolist()

    def get_combined_score(self, q_unique_words: List[str], q_len: int, body_k: int = 600, res_k: int = 300, w_body: float = 0.2,
                           w_title: float = 0.8, w_anchor: float = 0.6):
        """
        This function contains the logic of the search function. It first activates the search boby function and return {body_k} top results,
        Then its added the other inverted_index results' (title and anchor) and sum the weighted scores for each doc. finally, It return the {res_k}
        top results.
        :param q_unique_words:
        :param q_len:
        :param body_k:
        :param res_k:
        :param w_body: weight of inverted_index based on the body text results.
        :param w_title: weight of inverted_index based on the wiki titles results.
        :param w_anchor: weight of inverted_index based on the anchor text results.
        :param w_p_rank: weight of the page rank results - the p_rank scores are normalized with min_max scaler.
        :return:
        """
        body_tuples, body_wiki_id_l = self.get_results_search_body(q_unique_words=q_unique_words, q_len=q_len, k=body_k, get_scores=True)
        title_tuples, title_wiki_id_l = self.get_boolean_score_per_index(index_type='title', q_unique_words=q_unique_words, get_scores=True)
        anchor_tuples, anchor_wiki_id_l = self.get_boolean_score_per_index(index_type='anchor', q_unique_words=q_unique_words, get_scores=True)
        df_scores = pd.DataFrame(body_tuples).set_index(0).rename(columns={1: 'body_score'})
        title_scores = pd.DataFrame(title_tuples).set_index(0).rename(columns={1: 'title_score'})
        anchor_scores = pd.DataFrame(anchor_tuples).set_index(0).rename(columns={1: 'anchor_score'})
        df_scores = df_scores.join(title_scores, how='outer')
        df_scores = df_scores.join(anchor_scores, how='outer')
        # wiki_ids = list(set(body_wiki_id_l + title_wiki_id_l + anchor_wiki_id_l))

        # code for adding page rank to final score:
        # p_rank_df = self.page_rank_dataset[self.page_rank_dataset.doc_id.isin(wiki_ids)]  #
        # s_p_rank_df = p_rank_df.set_index('doc_id').rename(columns={'score': 'p_r_score'})
        # scaler = MinMaxScaler(feature_range=(0, df_scores['body_score'].max()))
        # s_p_rank_df['p_r_score'] = scaler.fit_transform(s_p_rank_df[['p_r_score']].values)
        # df_scores = df_scores.join(s_p_rank_df, how='outer')

        # code for adding page views to final score:
        # p_view_df = self.page_views_d[self.page_views_d.index.isin(wiki_ids)]
        # p_view_df = p_view_df.rename(columns={'n_views': 'p_view_score'})
        # scaler = MinMaxScaler(feature_range=(0, df_scores['body_score'].max()))
        # p_view_df['p_view_score'] = scaler.fit_transform(p_view_df[['p_view_score']].values)
        # df_scores = df_scores.join(p_view_df, how='outer')

        df_scores.fillna(0, inplace=True)
        df_scores['score'] = df_scores.apply(lambda row: w_body * row['body_score'] + w_title * row['title_score'] + w_anchor * row['anchor_score'], axis=1)
        # df_scores['score'] = df_scores.apply(lambda row: w_body * row['body_score'] + w_title * row['title_score'] + w_anchor * row['anchor_score']
        #                                      + w_p_view * row['p_view_score'], axis=1)
        return df_scores.sort_values(by='score', ascending=False).index.tolist()[:res_k]
