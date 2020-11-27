# Sentence Handler
import neuralcoref
import spacy
from spacy.lang.en import English
from typing import List
# Cluster Features
import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from typing import List
#transformer
from transformers import *
import logging
import torch
import numpy as np




class SentenceHandler(object):

    def __init__(self, language=English):
        self.nlp = language()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def process(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        """
        Processes the content sentences.
        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences.
        """
        doc = self.nlp(body)
        return [c.string.strip() for c in doc.sents if max_length > len(c.string.strip()) > min_length]

    def __call__(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        return self.process(body, min_length, max_length)


class CoreferenceHandler(SentenceHandler):

    def __init__(self, spacy_model: str = 'en_core_web_md', greedyness: float = 0.45):
        self.nlp = spacy.load(spacy_model)
        neuralcoref.add_to_pipe(self.nlp, greedyness=greedyness)

    def process(self, body: str, min_length: int = 40, max_length: int = 600):
        """
        Processes the content sentences.
        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences.
        """
        doc = self.nlp(body)._.coref_resolved
        doc = self.nlp(doc)
        return [c.string.strip() for c in doc.sents if max_length > len(c.string.strip()) > min_length]

class ClusterFeatures(object):
    """
    Basic handling of clustering features.
    """

    def __init__(
        self,
        features: ndarray,
        algorithm: str = 'kmeans',
        pca_k: int = None,
        random_state: int = 12345
    ):
        """
        :param features: the embedding matrix created by bert parent
        :param algorithm: Which clustering algorithm to use
        :param pca_k: If you want the features to be ran through pca, this is the components number
        :param random_state: Random state
        """
        self.features = features

        self.algorithm = algorithm
        self.random_state = random_state
        np.random.seed(random_state)

    def __get_model(self, k: int):
        """
        Retrieve clustering model
        :param k: amount of clusters
        :return: Clustering model
        """

        if self.algorithm == 'gmm':
            return GaussianMixture(n_components=k, random_state=self.random_state)

        return KMeans(n_clusters=k, random_state=self.random_state)

    def __get_centroids(self, model):
        """
        Retrieve centroids of model
        :param model: Clustering model
        :return: Centroids
        """

        if self.algorithm == 'gmm':
            return model.means_
        return model.cluster_centers_

    def __find_closest_args(self, centroids: np.ndarray):
        """
        Find the closest arguments to centroid
        :param centroids: Centroids to find closest
        :return: Closest arguments
        """

        centroid_min = 1e10
        cur_arg = -1
        args = {}
        used_idx = []

        for j, centroid in enumerate(centroids):

            for i, feature in enumerate(self.features):
                value = np.linalg.norm(feature - centroid)

                if value < centroid_min and i not in used_idx:
                    cur_arg = i
                    centroid_min = value

            used_idx.append(cur_arg)
            args[j] = cur_arg
            centroid_min = 1e10
            cur_arg = -1

        return args

    def cluster(self, ratio: float = 0.1) -> List[int]:
        """
        Clusters sentences based on the ratio
        :param ratio: Ratio to use for clustering
        :return: Sentences index that qualify for summary
        """

        k = 1 if ratio * len(self.features) < 1 else int(len(self.features) * ratio)
        model = self.__get_model(k).fit(self.features)
        centroids = self.__get_centroids(model)
        cluster_args = self.__find_closest_args(centroids)
        sorted_values = sorted(cluster_args.values())
        return sorted_values

    def __call__(self, ratio: float = 0.1) -> List[int]:
        return self.cluster(ratio)

class Summariser():
    def __init__(self):
        self.SentenceHandler = SentenceHandler()
        self.BertTokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BertModel = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True).to(self.device).eval()        

    def cluster(self, body,ratio=0.3):
        # Sentences are cleaned up by the sentence handler
        nlp_sentence_handler = self.SentenceHandler
        sentences = nlp_sentence_handler(body, min_length=2, max_length=600)
        
        # The Sentences are tokenised
        nlp_tokenizer = self.BertTokenizer
        tokenised_text_list = [ nlp_tokenizer.tokenize(t) for t in sentences ]
        
        # Indexes of tokens are obtained
        indexed_tokens_list = [ nlp_tokenizer.convert_tokens_to_ids(t) for t in tokenised_text_list ]

        # Init the nlp model
        nlp_model = self.BertModel
        
        # The sentences are passes through the model one at a time because of variable length
        sentence_embeddings = []
        hidden = -2
        for x in indexed_tokens_list:
            sent_tensor = torch.tensor([x]).to(self.device)
            pooled, hidden_states = nlp_model(sent_tensor)[-2:]
            pooled = hidden_states[hidden].mean(dim=1)
            sentence_embeddings.append(pooled)

        hidden_features = np.array([np.squeeze(x.detach().cpu().numpy()) for x in sentence_embeddings])

        hidden_args = ClusterFeatures(hidden_features, algorithm='kmeans', random_state=12345).cluster(ratio)

        if hidden_args[0] != 0:
            hidden_args.insert(0,0)

        result = [sentences[j] for j in hidden_args]


        # md_result = [x if (idx in hidden_args) else " *** "+x+" *** "  for idx, x in enumerate(sentences)  ]

        md_result = []
        for idx, x in enumerate(sentences):
            if idx in hidden_args:
                md_result.append("## "+x+"\n" )
            else:
                md_result.append(x)

        result = ''.join(result).replace('.','.\n')
        md_result = ''.join(md_result).replace('.','.\n')
        return result, md_result


if __name__ == "__main__":
    summariser = Summariser()
    body = '''
    The Chrysler Building, the famous art deco New York skyscraper, will be sold for a small fraction of its previous sales price.
    The deal, first reported by The Real Deal, was for $150 million, according to a source familiar with the deal.
    Mubadala, an Abu Dhabi investment fund, purchased 90% of the building for $800 million in 2008.
    Real estate firm Tishman Speyer had owned the other 10%.
    The buyer is RFR Holding, a New York real estate company.
    Officials with Tishman and RFR did not immediately respond to a request for comments.
    It's unclear when the deal will close.
    The building sold fairly quickly after being publicly placed on the market only two months ago.
    The sale was handled by CBRE Group.
    The incentive to sell the building at such a huge loss was due to the soaring rent the owners pay to Cooper Union, a New York college, for the land under the building.
    The rent is rising from $7.75 million last year to $32.5 million this year to $41 million in 2028.
    Meantime, rents in the building itself are not rising nearly that fast.
    While the building is an iconic landmark in the New York skyline, it is competing against newer office towers with large floor-to-ceiling windows and all the modern amenities.
    Still the building is among the best known in the city, even to people who have never been to New York.
    It is famous for its triangle-shaped, vaulted windows worked into the stylized crown, along with its distinctive eagle gargoyles near the top.
    It has been featured prominently in many films, including Men in Black 3, Spider-Man, Armageddon, Two Weeks Notice and Independence Day.
    The previous sale took place just before the 2008 financial meltdown led to a plunge in real estate prices.
    Still there have been a number of high profile skyscrapers purchased for top dollar in recent years, including the Waldorf Astoria hotel, which Chinese firm Anbang Insurance purchased in 2016 for nearly $2 billion, and the Willis Tower in Chicago, which was formerly known as Sears Tower, once the world's tallest.
    Blackstone Group (BX) bought it for $1.3 billion 2015.
    The Chrysler Building was the headquarters of the American automaker until 1953, but it was named for and owned by Chrysler chief Walter Chrysler, not the company itself.
    Walter Chrysler had set out to build the tallest building in the world, a competition at that time with another Manhattan skyscraper under construction at 40 Wall Street at the south end of Manhattan. He kept secret the plans for the spire that would grace the top of the building, building it inside the structure and out of view of the public until 40 Wall Street was complete.
    Once the competitor could rise no higher, the spire of the Chrysler building was raised into view, giving it the title.
    '''
    print(summariser.cluster(body))