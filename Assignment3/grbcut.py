from sklearn.mixture import GaussianMixture
# from GMM import GaussianMixture
import numpy as np
import igraph as ig

from config import *

class GrabCut:
    def __init__(self, image, mask, rect=None, num_neighbours=8, gamma=100, k_gmm_components=5):

        # Images and Mask
        self.image = image
        self.rows, self.cols, _ = image.shape
        self.mask = mask
        

        self.num_neighbours = num_neighbours

        # Add the rectangle to the mask if it is given
        if rect:
            # Initialise the interior of the rectangle to be probably foreground
            self.mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = DRAW_PR_FG['val']

        # Background and Foreground pixels
        self.bg_indices = None
        self.fg_indices = None
        # Initialize fg and bg pixel seggregation
        self.classify_pixels()
        # print(self.fg_indices.size)
        # print(len(self.bg_indices))

        # Graph 
        self.graph = ig.Graph(self.rows * self.cols + 2)
        self.graph_edges = []
        self.graph_edge_weights = []
        self.source = self.rows * self.cols
        self.target = self.rows * self.cols + 1

        # GMM Components
        self.k_gmm_components = k_gmm_components
        self.bg_gmm =  GaussianMixture(n_components=k_gmm_components)
        # self.bg_gmm =  GaussianMixture(self.image[self.bg_indices])#GaussianMixture(n_components=k_gmm_components)
        self.fg_gmm = GaussianMixture(n_components=k_gmm_components)
        # self.fg_gmm = GaussianMixture(self.image[self.fg_indices])#GaussianMixture(n_components=k_gmm_components)
        self.bg_gmm.fit(self.image[self.bg_indices])
        self.fg_gmm.fit(self.image[self.fg_indices])
        self.component_indices = np.empty((self.rows, self.cols), dtype=np.uint32)

        # Energy Parameters
        self.gamma = gamma
        self.beta = 0
        self.delta = {}

        # MRF parameters
        self.edge_potentials = {
            'N' : np.empty((self.rows-1,   self.cols)),
            'E' : np.empty((self.rows  , self.cols-1)),
            'S' : np.empty((self.rows-1,   self.cols)),
            'W' : np.empty((self.rows  , self.cols-1)),
            'NE': np.empty((self.rows-1, self.cols-1)),
            'SE': np.empty((self.rows-1, self.cols-1)),
            'SW': np.empty((self.rows-1, self.cols-1)),
            'NW': np.empty((self.rows-1, self.cols-1)),
        }

        # Define the delta for the image
        self.delta = self.compute_delta()

        # Compute the beta parameters
        self.beta = self.compute_beta()

        # Compute the edge potentials 
        self.edge_potentials = self.compute_edge_potentials()

    def classify_pixels(self):
        # Classify as Background pixels if the mask at those pixels is either background, or probably background
        self.bg_indices = np.where(np.logical_or(self.mask == DRAW_BG['val'], self.mask == DRAW_PR_BG['val']))

        # Classify as Foreground pixels if the mask at those pixels is either foreground, or probably foreground
        self.fg_indices = np.where(np.logical_or(self.mask == DRAW_FG['val'], self.mask == DRAW_PR_FG['val']))

    def compute_delta(self):
        self.delta = {
            'E'  : self.image[:, :-1] - self.image[:, 1:],
            'W'  : self.image[:, 1:] - self.image[:, :-1],
            'N'  : self.image[1:, :] - self.image[:-1, :],
            'S'  : self.image[:-1, :] - self.image[1:, :],
            'NE' : self.image[1:, :-1] - self.image[:-1, 1:],
            'NW' : self.image[1:, 1:] - self.image[:-1, :-1],
            'SE' : self.image[:-1, :-1] - self.image[1:, 1:],
            'SW' : self.image[:-1, 1:] - self.image[1:, :-1],
        }
        return self.delta

    def compute_beta(self):
        if self.num_neighbours == 4:
            beta = np.sum(np.square(self.delta['E'])) + \
                   np.sum(np.square(self.delta['W'])) + \
                   np.sum(np.square(self.delta['N'])) + \
                   np.sum(np.square(self.delta['S']))
            
            num_elems = self.delta['E'].size + \
                    self.delta['W'].size + \
                    self.delta['N'].size + \
                    self.delta['S'].size

            beta = 1 / (2*beta/num_elems)

            self.beta = beta
            return self.beta

        elif self.num_neighbours == 8:
            beta = np.sum(np.square(self.delta['E'])) + \
                   np.sum(np.square(self.delta['W'])) + \
                   np.sum(np.square(self.delta['N'])) + \
                   np.sum(np.square(self.delta['S'])) + \
                   np.sum(np.square(self.delta['NE'])) + \
                   np.sum(np.square(self.delta['SE'])) + \
                   np.sum(np.square(self.delta['SW'])) + \
                   np.sum(np.square(self.delta['NW']))
            
            num_elems = self.delta['E'].size + \
                    self.delta['W'].size + \
                    self.delta['N'].size + \
                    self.delta['S'].size + \
                    self.delta['NE'].size + \
                    self.delta['SE'].size + \
                    self.delta['SW'].size + \
                    self.delta['NW'].size

            beta = 1 / (2*beta/num_elems)

            self.beta = beta
            return self.beta

    def compute_edge_potentials(self):
        self.edge_potentials['N'] = self.gamma * np.exp(-self.beta * np.sum(np.square(self.delta['N']), axis=2))
        self.edge_potentials['S'] = self.gamma * np.exp(-self.beta * np.sum(np.square(self.delta['S']), axis=2))
        self.edge_potentials['E'] = self.gamma * np.exp(-self.beta * np.sum(np.square(self.delta['E']), axis=2))
        self.edge_potentials['W'] = self.gamma * np.exp(-self.beta * np.sum(np.square(self.delta['W']), axis=2))
        self.edge_potentials['NE'] = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(np.square(self.delta['NE']), axis=2))
        self.edge_potentials['SE'] = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(np.square(self.delta['SE']), axis=2))
        self.edge_potentials['SW'] = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(np.square(self.delta['SW']), axis=2))
        self.edge_potentials['NW'] = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(np.square(self.delta['NW']), axis=2))
        return self.edge_potentials

    def compute_data_potential(self, gmm, indices):

        _D = -gmm.score_samples(self.image.reshape(-1, 3)[indices])
        # .np.log(gmm.predict_proba(self.image.reshape(-1, 3)[indices]))
        # print(_D)
        return _D

        # posterior_prob = gmm.predict_proba(self.image.reshape(-1, 3)[indices])
        # data_potential = -np.log(posterior_prob)
        return data_potential

    def _add_t_links(self, uk_indices, fg_indices, bg_indices):
        self.graph_edges.extend(
            list(zip([self.source]*uk_indices.size, uk_indices))
        )
        data_potential = self.compute_data_potential(self.bg_gmm, uk_indices)
        self.graph_edge_weights.extend(data_potential.tolist())
        assert len(self.graph_edges) == len(self.graph_edge_weights)

        self.graph_edges.extend(
            list(zip([self.target]*uk_indices.size, uk_indices))
        )
        data_potential = self.compute_data_potential(self.fg_gmm, uk_indices)
        self.graph_edge_weights.extend(data_potential.tolist())
        assert len(self.graph_edges) == len(self.graph_edge_weights)

        self.graph_edges.extend(
            list(zip([self.source]*fg_indices.size, fg_indices))
        )
        data_potential = [9*self.gamma]*fg_indices.size
        self.graph_edge_weights.extend(data_potential)
        assert len(self.graph_edges) == len(self.graph_edge_weights)

        self.graph_edges.extend(
            list(zip([self.target]*fg_indices.size, fg_indices))
        )
        data_potential = [0]*fg_indices.size
        self.graph_edge_weights.extend(data_potential)
        assert len(self.graph_edges) == len(self.graph_edge_weights)

        self.graph_edges.extend(
            list(zip([self.source]*bg_indices.size, bg_indices))
        )
        data_potential = [0]*bg_indices.size
        self.graph_edge_weights.extend(data_potential)
        assert len(self.graph_edges) == len(self.graph_edge_weights)

        self.graph_edges.extend(
            list(zip([self.target]*bg_indices.size, bg_indices))
        )
        data_potential = [9*self.gamma]*bg_indices.size
        self.graph_edge_weights.extend(data_potential)
        assert len(self.graph_edges) == len(self.graph_edge_weights)

    def _add_n_links(self):
        # N Links and the Edgewise potentials Smoothness terms
        image_indices = np.arange(self.rows * self.cols, dtype=np.uint32).reshape(self.rows, self.cols)

        # North Neighbours
        mask_query = image_indices[1:,:].reshape(-1)
        mask_neigh = image_indices[:-1,:].reshape(-1)
        self.graph_edges.extend(
            list(zip(mask_query, mask_neigh))
        )
        self.graph_edge_weights.extend(self.edge_potentials['N'].reshape(-1).tolist())
        assert len(self.graph_edges) == len(self.graph_edge_weights)

        # South Neighbours
        mask_query = image_indices[:-1,:].reshape(-1)
        mask_neigh = image_indices[1:,:].reshape(-1)
        self.graph_edges.extend(
            list(zip(mask_query, mask_neigh))
        )
        self.graph_edge_weights.extend(self.edge_potentials['S'].reshape(-1).tolist())
        assert len(self.graph_edges) == len(self.graph_edge_weights)


        # East Neighbours 
        mask_query = image_indices[:,:-1].reshape(-1)
        mask_neigh = image_indices[:,1:].reshape(-1)
        self.graph_edges.extend(
            list(zip(mask_query, mask_neigh))
        )
        self.graph_edge_weights.extend(self.edge_potentials['E'].reshape(-1).tolist())
        assert len(self.graph_edges) == len(self.graph_edge_weights)


        # West Neighbours
        mask_query = image_indices[:,1:].reshape(-1)
        mask_neigh = image_indices[:,:-1].reshape(-1)
        self.graph_edges.extend(
            list(zip(mask_query, mask_neigh))
        )
        self.graph_edge_weights.extend(self.edge_potentials['W'].reshape(-1).tolist())
        assert len(self.graph_edges) == len(self.graph_edge_weights)

        if self.num_neighbours == 4:
            return


        # North East Neighbours
        mask_query = image_indices[1:,:-1].reshape(-1)
        mask_neigh = image_indices[:-1,1:].reshape(-1)
        self.graph_edges.extend(
            list(zip(mask_query, mask_neigh))
        )
        self.graph_edge_weights.extend(self.edge_potentials['NE'].reshape(-1).tolist())
        assert len(self.graph_edges) == len(self.graph_edge_weights)


        # North West Neighbours
        mask_query = image_indices[1:,1:].reshape(-1)
        mask_neigh = image_indices[:-1,:-1].reshape(-1)
        self.graph_edges.extend(
            list(zip(mask_query, mask_neigh))
        )
        self.graph_edge_weights.extend(self.edge_potentials['NW'].reshape(-1).tolist())
        assert len(self.graph_edges) == len(self.graph_edge_weights)


        # South East Neighbours
        mask_query = image_indices[:-1,:-1].reshape(-1)
        mask_neigh = image_indices[1:,1:].reshape(-1)
        self.graph_edges.extend(
            list(zip(mask_query, mask_neigh))
        )
        self.graph_edge_weights.extend(self.edge_potentials['SE'].reshape(-1).tolist())
        assert len(self.graph_edges) == len(self.graph_edge_weights)


        # South West Neighbours
        mask_query = image_indices[:-1,1:].reshape(-1)
        mask_neigh = image_indices[1:,:-1].reshape(-1)
        self.graph_edges.extend(
            list(zip(mask_query, mask_neigh))
        )
        self.graph_edge_weights.extend(self.edge_potentials['SW'].reshape(-1).tolist())
        assert len(self.graph_edges) == len(self.graph_edge_weights)

    def construct_graph(self):

        # Get all the labels and markings for all the pixels
        fg_indices = np.where(self.mask.reshape(-1) == DRAW_FG['val'])[0]
        bg_indices = np.where(self.mask.reshape(-1) == DRAW_BG['val'])[0]
        uk_indices = np.where(np.logical_or(self.mask.reshape(-1)==DRAW_PR_BG['val'], self.mask.reshape(-1)==DRAW_PR_FG['val']))[0]
        
        self.graph_edges = []
        self.graph_edge_weights = [] 

        self._add_t_links(uk_indices, fg_indices, bg_indices)
        self._add_n_links()

        self.graph = ig.Graph(self.rows * self.cols + 2)
        self.graph.add_edges(self.graph_edges)

    def assign_gmm(self):
        # self.component_indices[self.bg_indices] = self.bg_gmm.which_component(self.image[self.bg_indices])
        # self.component_indices[self.fg_indices] = self.fg_gmm.which_component(self.image[self.fg_indices])
        
        self.component_indices[self.bg_indices] = self.bg_gmm.predict(self.image[self.bg_indices])
        self.component_indices[self.fg_indices] = self.fg_gmm.predict(self.image[self.fg_indices])

    def learn_gmm(self):
        # self.bg_gmm.fit(self.image[self.bg_indices], self.component_indices[self.bg_indices])
        # self.fg_gmm.fit(self.image[self.fg_indices], self.component_indices[self.fg_indices])
        self.bg_gmm.fit(self.image[self.bg_indices], self.component_indices[self.bg_indices])
        self.fg_gmm.fit(self.image[self.fg_indices], self.component_indices[self.fg_indices])


    def get_segmentation(self):
        st_mincut = self.graph.st_mincut(self.source, self.target, self.graph_edge_weights)
        image_indices = np.arange(self.rows * self.cols, dtype=np.uint32).reshape(self.rows, self.cols)
        uk_indices = np.where(np.logical_or(self.mask==DRAW_PR_BG['val'], self.mask==DRAW_PR_FG['val']))
        self.mask[uk_indices] = np.where(np.isin(image_indices[uk_indices], st_mincut.partition[0]), DRAW_PR_FG['val'], DRAW_PR_BG['val'])
        self.classify_pixels()

    def run(self, n_iters=1):
        for i in range(n_iters):
            self.assign_gmm()
            self.learn_gmm()
            self.construct_graph()
            self.get_segmentation()
            
