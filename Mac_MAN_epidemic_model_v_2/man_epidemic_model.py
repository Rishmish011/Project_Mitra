# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 15:22:05 2025

@author: risha
"""

"""
Mitra: Multi-scale Adaptive Network (MAN) Epidemic Model

Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)

Copyright (c) 2025 Rishabh Mishra

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. 
  You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- NonCommercial — You may not use the material for commercial purposes.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others 
from doing anything the license permits.

Notices:
You do not have to comply with the license for elements of the material in the public domain 
or where your use is permitted by an applicable exception or limitation.

No warranties are given. The license may not give you all of the permissions necessary for your intended use. 
For example, other rights such as publicity, privacy, or moral rights may limit how you use the material.

## DISCLAIMER OF LIABILITY

This software is provided "as is" without warranty of any kind, express or implied. The authors and contributors disclaim all warranties, including, but not limited to, any implied warranties of merchantability, fitness for a particular purpose, and non-infringement.
In no event shall the authors, contributors, or copyright holders be liable for any claim, damages, or other liability, whether in action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
This software is intended for research and educational purposes only. Any decisions made based on the outputs of this model are the sole responsibility of the user. The software developers are not responsible for any actions taken based on the software's results or interpretations thereof.
This epidemic model is a simplified representation of complex real-world systems and should not be the sole basis for public health decision-making.
=================================================

A comprehensive implementation of the MAN epidemic model that incorporates:
1. Hierarchical network structure
2. Adaptive parameters
3. Behavior-pathogen feedback loop
4. Computational efficiency through selective resolution
5. Bayesian data assimilation
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Set
import pickle
import os
import multiprocessing as mp
from enum import Enum, auto

# Set random seed for reproducibility
np.random.seed(42)

class HealthState(Enum):
    """Health states for individuals in the epidemic model."""
    SUSCEPTIBLE = auto()
    EXPOSED = auto()
    INFECTIOUS = auto()
    ASYMPTOMATIC = auto()
    RECOVERED = auto()
    DECEASED = auto()

class NetworkScale(Enum):
    """Scales for the hierarchical contact network."""
    MICRO = auto()  # Household, close contacts
    MESO = auto()   # Community
    MACRO = auto()  # Regional

class ModelResolution(Enum):
    """Resolution levels for adaptive computation."""
    AGENT_BASED = auto()     # Highest resolution
    METAPOPULATION = auto()  # Medium resolution
    COMPARTMENTAL = auto()   # Lowest resolution

@dataclass
class Parameters:
    """Model parameters with default values."""
    # Basic parameters
    population_size: int = 10000
    simulation_days: int = 100
    time_step: float = 0.1
    
    # Network parameters
    scale_weights: Dict[NetworkScale, float] = None
    network_rewiring_rate: float = 0.01
    
    # Disease parameters
    base_transmission_rate: float = 0.3
    base_incubation_rate: float = 0.2
    base_recovery_rate: float = 0.1
    base_mortality_rate: float = 0.01
    asymptomatic_fraction: float = 0.4
    asymptomatic_transmission_factor: float = 0.5
    
    # Pathogen evolution parameters
    mutation_rate: float = 0.001
    max_transmissibility: float = 2.0
    evolution_noise_scale: float = 0.05
    
    # Behavior parameters
    behavior_adaption_rate: float = 0.2
    information_generation_rate: float = 0.5
    information_decay_rate: float = 0.1
    media_influence_factor: float = 0.3
    
    # Resolution parameters
    high_resolution_threshold: float = 0.05
    low_resolution_threshold: float = 0.01
    
    # Data assimilation parameters
    ensemble_size: int = 20
    observation_noise_sd: float = 0.1
    
    # Environmental factors
    seasonality_amplitude: float = 0.2
    
    # Gossip propagation parameters
    information_diffusion_mode: str = "uniform"  # Options: "uniform", "gossip"
    gossip_transmission_probability: float = 0.75  # Probability q from paper
    information_hop_limit: int = 2  # How far information spreads
    gossip_delay_factor: float = 1.0  # Controls spreading time scaling
    optimal_degree_effect: float = 0.5  # Strength of the optimal degree effect
    
    # Optimal degree parameters (from paper findings)
    optimal_degree_base: float = 7.0  # Approximate k₀ value from paper
    optimal_degree_network_scaling: float = 4.64  # Parameter a in eq(2)
    optimal_degree_connectivity_scaling: float = 1.34  # Parameter b in eq(2)
    
    def __post_init__(self):
        """Initialize derived parameters."""
        if self.scale_weights is None:
            self.scale_weights = {
                NetworkScale.MICRO: 0.6,
                NetworkScale.MESO: 0.3,
                NetworkScale.MACRO: 0.1
            }


class Individual:
    """Represents a single individual in the population."""
    
    def __init__(self, idx: int, age: float, location: Tuple[float, float], 
                 risk_factors: np.ndarray, initial_behavior: np.ndarray):
        """
        Initialize an individual.
        
        Args:
            idx: Unique identifier
            age: Age (years)
            location: Spatial coordinates (x, y)
            risk_factors: Vector of health risk factors
            initial_behavior: Initial behavioral state vector
        """
        self.idx = idx
        self.age = age
        self.location = np.array(location)
        self.risk_factors = risk_factors
        self.behavior = initial_behavior
        
        # Health state (default: susceptible)
        self.health_state = HealthState.SUSCEPTIBLE
        self.days_in_state = 0
        self.environmental_factors = np.random.normal(0, 1, size=3)
        
        # Infection tracking
        self.infection_source = None
        self.secondary_infections = 0
        
        # History tracking
        self.state_history = [(0, HealthState.SUSCEPTIBLE)]
        self.behavior_history = [(0, initial_behavior.copy())]
        
        # Gossip-related attributes
        self.spread_factor = 0.0
        self.spreading_time = 0.0

    def update_behavior(self, information_field: float, policy_interventions: np.ndarray, 
                        media_influence: float, params: Parameters) -> np.ndarray:
        """
        Update behavioral state based on current information and policies.
        
        Args:
            information_field: Current information about disease prevalence
            policy_interventions: Vector of active policy interventions
            media_influence: Strength of media influence
            params: Model parameters
            
        Returns:
            Change in behavior vector
        """
        # Simple linear model for behavior change
        target_behavior = (
            0.3 * information_field + 
            0.5 * policy_interventions +
            0.2 * media_influence * np.ones_like(self.behavior)
        )
        
        # Behavior changes gradually toward target
        behavior_change = params.behavior_adaption_rate * (target_behavior - self.behavior)
        
        # Add individual variation
        behavior_change += np.random.normal(0, 0.05, size=behavior_change.shape)
        
        return behavior_change
    
    def update_health_state(self, force_of_infection: float, params: Parameters, time: float) -> None:
        """
        Update health state based on disease progression.
        
        Args:
            force_of_infection: Current force of infection
            params: Model parameters
            time: Current simulation time
        """
        old_state = self.health_state
        self.days_in_state += params.time_step
        
        # Compute transition probabilities based on current state
        if self.health_state == HealthState.SUSCEPTIBLE:
            # Probability of becoming exposed
            p_infection = 1 - np.exp(-force_of_infection * params.time_step)
            if np.random.random() < p_infection:
                self.health_state = HealthState.EXPOSED
                self.days_in_state = 0
                
        elif self.health_state == HealthState.EXPOSED:
            # Age-dependent incubation rate
            incubation_rate = params.base_incubation_rate * (1 + 0.2 * (self.age - 40) / 40)
            p_infectious = 1 - np.exp(-incubation_rate * params.time_step)
            
            if np.random.random() < p_infectious:
                # Some fraction become asymptomatic
                if np.random.random() < params.asymptomatic_fraction:
                    self.health_state = HealthState.ASYMPTOMATIC
                else:
                    self.health_state = HealthState.INFECTIOUS
                self.days_in_state = 0
                
        elif self.health_state == HealthState.INFECTIOUS:
            # Risk factor and age-dependent recovery and mortality rates
            risk_modifier = 1.0 + 0.5 * np.sum(self.risk_factors)
            age_modifier = np.exp(0.03 * max(0, self.age - 50))
            
            recovery_rate = params.base_recovery_rate / (risk_modifier * age_modifier)
            mortality_rate = params.base_mortality_rate * risk_modifier * age_modifier
            
            p_recovery = 1 - np.exp(-recovery_rate * params.time_step)
            p_mortality = 1 - np.exp(-mortality_rate * params.time_step)
            
            rand = np.random.random()
            if rand < p_mortality:
                self.health_state = HealthState.DECEASED
                self.days_in_state = 0
            elif rand < p_mortality + p_recovery:
                self.health_state = HealthState.RECOVERED
                self.days_in_state = 0
                
        elif self.health_state == HealthState.ASYMPTOMATIC:
            # Asymptomatic individuals recover faster and don't die
            recovery_rate = 1.2 * params.base_recovery_rate
            p_recovery = 1 - np.exp(-recovery_rate * params.time_step)
            
            if np.random.random() < p_recovery:
                self.health_state = HealthState.RECOVERED
                self.days_in_state = 0
        
        # Record state change
        if self.health_state != old_state:
            self.state_history.append((time, self.health_state))


class Region:
    """Represents a spatial region in the model."""
    
    def __init__(self, region_id: str, population: List[Individual], 
                 location: Tuple[float, float], radius: float):
        """
        Initialize a region.
        
        Args:
            region_id: Unique identifier for the region
            population: List of individuals in the region
            location: Center coordinates of the region
            radius: Region radius
        """
        self.region_id = region_id
        self.population = set(population)
        self.location = np.array(location)
        self.radius = radius
        
        # Region-level variables
        self.resolution = ModelResolution.AGENT_BASED
        
        # For compartmental modeling
        self.compartments = {
            HealthState.SUSCEPTIBLE: 0,
            HealthState.EXPOSED: 0,
            HealthState.INFECTIOUS: 0,
            HealthState.ASYMPTOMATIC: 0,
            HealthState.RECOVERED: 0,
            HealthState.DECEASED: 0
        }
        
        # For metapopulation modeling
        self.subregions = []
        
    def contains(self, location: np.ndarray) -> bool:
        """Check if a location is within this region."""
        distance = np.linalg.norm(location - self.location)
        return distance <= self.radius
    
    def add_individual(self, individual: Individual) -> None:
        """Add an individual to the region."""
        self.population.add(individual)
        
    def remove_individual(self, individual: Individual) -> None:
        """Remove an individual from the region."""
        if individual in self.population:
            self.population.remove(individual)
            
    def get_epidemic_intensity(self) -> float:
        """
        Calculate the epidemic intensity in this region.
        
        Returns:
            A value representing epidemic activity level
        """
        # Weights for different health states
        weights = {
            HealthState.SUSCEPTIBLE: 0,
            HealthState.EXPOSED: 0.5,
            HealthState.INFECTIOUS: 1.0,
            HealthState.ASYMPTOMATIC: 0.7,
            HealthState.RECOVERED: 0,
            HealthState.DECEASED: 0
        }
        
        if not self.population:
            return 0
            
        # Calculate weighted sum of cases
        intensity = sum(weights[ind.health_state] for ind in self.population)
        return intensity / len(self.population)
    
    def determine_resolution(self, high_threshold: float, low_threshold: float) -> ModelResolution:
        """
        Determine appropriate modeling resolution based on epidemic intensity.
        
        Args:
            high_threshold: Threshold for high-resolution modeling
            low_threshold: Threshold for low-resolution modeling
            
        Returns:
            Appropriate model resolution for this region
        """
        intensity = self.get_epidemic_intensity()
        
        if intensity > high_threshold:
            return ModelResolution.AGENT_BASED
        elif intensity > low_threshold:
            return ModelResolution.METAPOPULATION
        else:
            return ModelResolution.COMPARTMENTAL
            
    def update_compartments(self) -> None:
        """Update compartmental counts based on individual states."""
        # Reset counts
        for state in HealthState:
            self.compartments[state] = 0
            
        # Count individuals in each state
        for ind in self.population:
            self.compartments[ind.health_state] += 1


class ContactNetwork:
    """Manages the multi-scale contact network structure."""
    
    def __init__(self, population: List[Individual], params: Parameters):
        """
        Initialize contact networks at different scales.
        
        Args:
            population: List of all individuals
            params: Model parameters
        """
        self.population = population
        self.params = params
        self.n_individuals = len(population)
        
        # Create network structures at each scale
        self.networks = {
            NetworkScale.MICRO: self._create_micro_network(),
            NetworkScale.MESO: self._create_meso_network(),
            NetworkScale.MACRO: self._create_macro_network()
        }
        
        # Effective network (weighted combination)
        self.effective_network = self._compute_effective_network()
        
    def _create_micro_network(self) -> nx.Graph:
        """
        Create household and close contact network.
        
        Returns:
            NetworkX graph of micro-scale contacts
        """
        # Start with an empty graph
        G = nx.Graph()
        
        # Add all individuals as nodes
        for ind in self.population:
            G.add_node(ind.idx)
        
        # Create households (complete graphs of 2-6 people)
        remaining = set(range(self.n_individuals))
        
        while remaining:
            # Determine household size
            household_size = min(np.random.choice([2, 3, 4, 5, 6], p=[0.2, 0.3, 0.3, 0.15, 0.05]), 
                               len(remaining))
            
            # Select household members
            household = np.random.choice(list(remaining), size=household_size, replace=False)
            
            # Add edges for complete graph within household
            for i in range(household_size):
                for j in range(i+1, household_size):
                    G.add_edge(household[i], household[j], weight=1.0)
            
            # Remove from remaining
            remaining -= set(household)
        
        # Add some random close contacts
        avg_close_contacts = 3  # Average number of close contacts outside household
        
        for ind_idx in range(self.n_individuals):
            n_contacts = np.random.poisson(avg_close_contacts)
            potential_contacts = [i for i in range(self.n_individuals) 
                               if i != ind_idx and not G.has_edge(ind_idx, i)]
            
            if potential_contacts and n_contacts > 0:
                actual_contacts = min(n_contacts, len(potential_contacts))
                contacts = np.random.choice(potential_contacts, size=actual_contacts, replace=False)
                
                for contact in contacts:
                    # Weight based on age similarity (closer ages = stronger connection)
                    age_i = self.population[ind_idx].age
                    age_j = self.population[contact].age
                    age_similarity = np.exp(-0.05 * abs(age_i - age_j))
                    
                    # Weight based on geographical proximity
                    loc_i = self.population[ind_idx].location
                    loc_j = self.population[contact].location
                    distance = np.linalg.norm(loc_i - loc_j)
                    proximity = np.exp(-0.1 * distance)
                    
                    # Combined weight
                    weight = 0.7 * age_similarity + 0.3 * proximity
                    
                    G.add_edge(ind_idx, contact, weight=weight)
        
        return G
    
    def _create_meso_network(self) -> nx.Graph:
        """
        Create community-level contact network.
        
        Returns:
            NetworkX graph of meso-scale contacts
        """
        # Start with an empty graph
        G = nx.Graph()
        
        # Add all individuals as nodes
        for ind in self.population:
            G.add_node(ind.idx)
        
        # Create community structures
        # We'll use a spatial approach - people close to each other are more likely to be connected
        
        # Number of communities (approximately population / 100)
        n_communities = max(1, self.n_individuals // 100)
        
        # Randomly select community centers
        centers_idx = np.random.choice(range(self.n_individuals), size=n_communities, replace=False)
        centers = [self.population[idx].location for idx in centers_idx]
        
        # Assign each person to nearest community
        communities = [[] for _ in range(n_communities)]
        
        for i, ind in enumerate(self.population):
            distances = [np.linalg.norm(ind.location - center) for center in centers]
            nearest_community = np.argmin(distances)
            communities[nearest_community].append(i)
        
        # Create connections within communities (with some randomness)
        for community in communities:
            community_size = len(community)
            
            # Probability of connection depends on community size
            # Smaller communities are more tightly connected
            if community_size < 2:
                continue
                
            p_connect = min(10.0 / community_size, 0.8)
            
            for i in range(community_size):
                for j in range(i+1, community_size):
                    if np.random.random() < p_connect:
                        ind_i = community[i]
                        ind_j = community[j]
                        
                        # Weight based on geographical proximity
                        loc_i = self.population[ind_i].location
                        loc_j = self.population[ind_j].location
                        distance = np.linalg.norm(loc_i - loc_j)
                        weight = np.exp(-0.2 * distance)
                        
                        G.add_edge(ind_i, ind_j, weight=weight)
        
        return G
    
    def _create_macro_network(self) -> nx.Graph:
        """
        Create regional-level contact network.
        
        Returns:
            NetworkX graph of macro-scale contacts
        """
        # Start with an empty graph
        G = nx.Graph()
        
        # Add all individuals as nodes
        for ind in self.population:
            G.add_node(ind.idx)
        
        # Create sparse regional connections
        # We'll use a configuration model with power-law degree distribution
        
        # Generate power-law degrees (most people have few regional contacts, some have many)
        alpha = 2.5  # Power law exponent
        degrees = np.random.zipf(alpha, self.n_individuals)
        degrees = np.minimum(degrees, self.n_individuals-1)  # Cap max degree
        
        # Adjust total to be even (required for configuration model)
        if sum(degrees) % 2 == 1:
            degrees[0] += 1
        
        # Create graph using configuration model
        G = nx.configuration_model(degrees)
        G = nx.Graph(G)  # Remove parallel edges
        G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops
        
        # Add weights based on geographic distance
        for i, j in G.edges():
            loc_i = self.population[i].location
            loc_j = self.population[j].location
            distance = np.linalg.norm(loc_i - loc_j)
            
            # Regional connections have lower weights for large distances
            weight = np.exp(-0.01 * distance)
            G[i][j]['weight'] = weight
        
        return G
    
    def _compute_effective_network(self) -> np.ndarray:
        """
        Compute effective contact network as weighted combination of all scales.
        
        Returns:
            Adjacency matrix of effective network
        """
        # Create adjacency matrices
        adj_matrices = {}
        for scale, network in self.networks.items():
            adj = np.zeros((self.n_individuals, self.n_individuals))
            
            for i, j, data in network.edges(data=True):
                weight = data.get('weight', 1.0)
                adj[i, j] = weight
                adj[j, i] = weight  # Ensure symmetry
                
            adj_matrices[scale] = adj
        
        # Combine with scale weights
        effective_adj = np.zeros_like(adj_matrices[NetworkScale.MICRO])
        
        for scale, adj in adj_matrices.items():
            scale_weight = self.params.scale_weights[scale]
            effective_adj += scale_weight * adj
            
        return effective_adj
    
    def update_networks(self, time: float) -> None:
        """
        Update networks based on individual behavior changes.
        
        Args:
            time: Current simulation time
        """
        # Update each network scale
        for scale in NetworkScale:
            G = self.networks[scale]
            
            # Random rewiring with probability based on network scale
            rewiring_probability = self.params.network_rewiring_rate
            
            if scale == NetworkScale.MICRO:
                # Household connections are relatively stable
                rewiring_probability *= 0.2
            elif scale == NetworkScale.MESO:
                # Community connections change at moderate rate
                rewiring_probability *= 1.0
            else:  # MACRO
                # Regional connections change more frequently
                rewiring_probability *= 2.0
            
            # Apply rewiring
            for i, j in list(G.edges()):
                if np.random.random() < rewiring_probability:
                    # Remove this edge
                    weight = G[i][j]['weight']
                    G.remove_edge(i, j)
                    
                    # Find new connection
                    potential_targets = [k for k in range(self.n_individuals) 
                                      if k != i and k != j and not G.has_edge(i, k)]
                    
                    if potential_targets:
                        new_target = np.random.choice(potential_targets)
                        
                        # Calculate new weight based on behavior similarity
                        behavior_i = self.population[i].behavior
                        behavior_j = self.population[new_target].behavior
                        
                        behavior_distance = np.linalg.norm(behavior_i - behavior_j)
                        new_weight = weight * np.exp(-0.5 * behavior_distance)
                        
                        G.add_edge(i, new_target, weight=new_weight)
            
            # Behavior affects existing connection weights
            for i, j in G.edges():
                behavior_i = self.population[i].behavior
                behavior_j = self.population[j].behavior
                
                # More similar behaviors strengthen connections
                behavior_distance = np.linalg.norm(behavior_i - behavior_j)
                behavior_factor = np.exp(-0.2 * behavior_distance)
                
                # Health state affects connection weights
                health_i = self.population[i].health_state
                health_j = self.population[j].health_state
                
                health_factor = 1.0
                if health_i in [HealthState.INFECTIOUS, HealthState.ASYMPTOMATIC] or \
                   health_j in [HealthState.INFECTIOUS, HealthState.ASYMPTOMATIC]:
                    # People reduce contact with infectious individuals
                    # More cautious individuals (higher behavior value) reduce more
                    caution_i = behavior_i[0]  # Assuming first behavior dimension is caution
                    caution_j = behavior_j[0]
                    health_factor = np.exp(-0.5 * (caution_i + caution_j))
                
                # Update weight
                current_weight = G[i][j]['weight']
                new_weight = current_weight * (0.9 + 0.1 * behavior_factor * health_factor)
                
                # Keep weights bounded
                new_weight = max(0.1, min(1.0, new_weight))
                G[i][j]['weight'] = new_weight
        
        # Recompute effective network
        self.effective_network = self._compute_effective_network()


class GossipPropagation:
    """Models the propagation of information (gossip) through the network."""
    
    def __init__(self, model, params: Parameters):
        """
        Initialize gossip propagation model.
        
        Args:
            model: Reference to main model
            params: Model parameters
        """
        self.model = model
        self.params = params
        self.spread_factors = {}  # Maps individual idx to their spread factor
        self.spreading_times = {}  # Maps individual idx to their spreading time
        
    def calculate_spread_factor(self, target_idx: int, hop_limit: int = None) -> float:
        """
        Calculate spread factor for a target individual.
        
        Args:
            target_idx: Index of the target individual
            hop_limit: How many hops information can travel (defaults to parameter value)
            
        Returns:
            Spread factor f value for the target
        """
        if hop_limit is None:
            hop_limit = self.params.information_hop_limit
            
        # Get the target's neighbors
        micro_network = self.model.contact_network.networks[NetworkScale.MICRO]
        if target_idx not in micro_network:
            return 0.0
            
        neighbors = list(micro_network.neighbors(target_idx))
        if not neighbors:
            return 0.0
            
        # For each potential originator of gossip
        total_reached = 0
        for originator in neighbors:
            # Track visited nodes
            visited = {originator}
            to_visit = {originator}  # Start with originator
            current_hop = 0
            
            while to_visit and current_hop < hop_limit:
                next_to_visit = set()
                
                # Explore current nodes
                for node in to_visit:
                    # For each of this node's neighbors
                    for neighbor in micro_network.neighbors(node):
                        # Only include neighbors of target that aren't already visited
                        if (neighbor in neighbors and 
                            neighbor != originator and 
                            neighbor not in visited):
                            
                            # Apply transmission probability
                            if np.random.random() < self.params.gossip_transmission_probability:
                                visited.add(neighbor)
                                next_to_visit.add(neighbor)
                
                # Prepare for next iteration
                to_visit = next_to_visit
                current_hop += 1
            
            # Count reached neighbors (excluding originator)
            reached = len([n for n in visited if n in neighbors]) - 1
            reached = max(0, reached)  # Ensure non-negative
            total_reached += reached
            
        # Average over all potential originators
        spread_factor = total_reached / (len(neighbors) * (len(neighbors) - 1)) if len(neighbors) > 1 else 0
        return spread_factor
    
    def calculate_spreading_time(self, target_idx: int, hop_limit: int = None) -> float:
        """
        Calculate the time needed for information to reach maximum spread.
        
        Args:
            target_idx: Index of the target individual
            hop_limit: How many hops information can travel
            
        Returns:
            Spreading time τ value for the target
        """
        if hop_limit is None:
            hop_limit = self.params.information_hop_limit
            
        # Get the target's neighbors
        micro_network = self.model.contact_network.networks[NetworkScale.MICRO]
        if target_idx not in micro_network:
            return 0.0
            
        neighbors = list(micro_network.neighbors(target_idx))
        if not neighbors:
            return 0.0
            
        # For each potential originator of gossip
        total_time = 0
        count = 0
        
        for originator in neighbors:
            # BFS to find shortest paths to all other neighbors
            visited = {originator: 0}  # node -> time to reach
            queue = [(originator, 0)]  # (node, time)
            
            while queue:
                node, time = queue.pop(0)
                
                if time >= hop_limit:
                    continue
                
                # For each neighbor of this node
                for neighbor in micro_network.neighbors(node):
                    # If also neighbor of target and not visited yet
                    if neighbor in neighbors and neighbor not in visited:
                        # Apply transmission probability
                        if np.random.random() < self.params.gossip_transmission_probability:
                            visited[neighbor] = time + 1
                            queue.append((neighbor, time + 1))
            
            # Find maximum time among reached neighbors
            reached_neighbors = [n for n in visited if n in neighbors and n != originator]
            if reached_neighbors:
                max_time = max(visited[n] for n in reached_neighbors)
                total_time += max_time
                count += 1
        
        # Average over all originators that successfully spread
        spreading_time = (total_time / count) if count > 0 else 0
        return spreading_time * self.params.gossip_delay_factor
    
    def update_all_metrics(self) -> None:
        """Update spread factors and spreading times for all individuals."""
        for ind in self.model.population:
            self.spread_factors[ind.idx] = self.calculate_spread_factor(ind.idx)
            self.spreading_times[ind.idx] = self.calculate_spreading_time(ind.idx)
            
            # Update individual's attributes
            ind.spread_factor = self.spread_factors[ind.idx]
            ind.spreading_time = self.spreading_times[ind.idx]
            
    def get_optimal_degree(self) -> float:
        """
        Calculate the optimal degree k₀ that minimizes gossip vulnerability.
        Uses equation (2) from the paper.
        """
        N = len(self.model.population)
        m = np.mean([len(list(self.model.contact_network.networks[NetworkScale.MICRO].neighbors(ind.idx))) 
                     for ind in self.model.population]) / 2  # Average degree / 2
        
        # Apply equation (2) from paper: k₀ ∝ (ln N)^a / (ln m)^b
        k0 = self.params.optimal_degree_base * \
             (np.log(N) ** self.params.optimal_degree_network_scaling) / \
             (np.log(max(1.1, m)) ** self.params.optimal_degree_connectivity_scaling)
        
        return k0
        
    def calculate_vulnerability(self, individual_idx: int) -> float:
        """
        Calculate an individual's vulnerability to gossip based on their degree.
        
        Returns:
            Value between 0-1 where higher means more vulnerable
        """
        # Get degree of individual
        degree = len(list(self.model.contact_network.networks[NetworkScale.MICRO].neighbors(individual_idx)))
        
        # Get current optimal degree
        k0 = self.get_optimal_degree()
        
        # Calculate vulnerability based on distance from optimal degree
        # This creates a u-shaped function with minimum at k0
        vulnerability = 1.0 - self.params.optimal_degree_effect * np.exp(-0.2 * ((degree - k0)/k0)**2)
        
        return max(0.0, min(1.0, vulnerability))


class PathogenModel:
    """Models pathogen evolution over time."""
    
    def __init__(self, params: Parameters):
        """
        Initialize pathogen model.
        
        Args:
            params: Model parameters
        """
        self.params = params
        self.transmissibility = 1.0  # Relative to base rate
        self.history = [(0, 1.0)]
        
    def update(self, time: float) -> None:
        """
        Update pathogen properties based on evolution model.
        
        Args:
            time: Current simulation time
        """
        # Logistic growth with stochastic jumps
        deterministic_change = self.params.mutation_rate * self.transmissibility * \
                              (1 - self.transmissibility / self.params.max_transmissibility)
        
        # Random evolutionary jumps (possibly representing variants)
        stochastic_change = np.random.normal(0, self.params.evolution_noise_scale)
        
        # Apply changes
        self.transmissibility += (deterministic_change + stochastic_change) * self.params.time_step
        
        # Bound transmissibility
        self.transmissibility = max(0.1, min(self.params.max_transmissibility, self.transmissibility))
        
        # Record history
        self.history.append((time, self.transmissibility))


class BehaviorModel:
    """Models population-level behavioral dynamics."""
    
    def __init__(self, population: List[Individual], params: Parameters):
        """
        Initialize behavior model.
        
        Args:
            population: List of all individuals
            params: Model parameters
        """
        self.population = population
        self.params = params
        
        # Information and media influence fields
        self.information_level = 0.0
        self.media_influence = 0.0
        
        # Policy interventions (vector of strengths for different intervention types)
        # [0]: Social distancing, [1]: Mask wearing, [2]: Vaccination
        self.policy_interventions = np.zeros(3)
        
        # History
        self.info_history = [(0, 0.0)]
        self.media_history = [(0, 0.0)]
        self.policy_history = [(0, self.policy_interventions.copy())]
        
        # Reference to gossip model - will be set by main model if needed
        self.gossip_model = None
        
    def update_information_field(self, cases: int, time: float) -> None:
        """
        Update information field based on current cases and selected diffusion mode.
        
        Args:
            cases: Current number of infectious cases
            time: Current simulation time
        """
        if self.params.information_diffusion_mode == "uniform":
            # Original uniform information update logic
            self._uniform_information_update(cases)
            
        elif self.params.information_diffusion_mode == "gossip":
            # Gossip-based information diffusion
            if self.gossip_model:
                # Calculate base information change
                case_fraction = cases / len(self.population)
                base_change = (
                    self.params.information_generation_rate * case_fraction - 
                    self.params.information_decay_rate * self.information_level
                ) * self.params.time_step
                
                # Modify based on gossip dynamics
                avg_spread_factor = np.mean(list(self.gossip_model.spread_factors.values())) if self.gossip_model.spread_factors else 0.5
                avg_spreading_time = np.mean(list(self.gossip_model.spreading_times.values())) if self.gossip_model.spreading_times else 1.0
                
                # Information spreads faster with higher spread factor and lower spreading time
                gossip_multiplier = avg_spread_factor * np.exp(-0.2 * avg_spreading_time)
                self.information_level += base_change * (1.0 + gossip_multiplier)
            else:
                # Fall back to uniform if gossip model not initialized
                self._uniform_information_update(cases)
        
        # Bound information level
        self.information_level = max(0.0, min(1.0, self.information_level))
        
        # Update media influence (lags behind information with some noise)
        media_target = self.information_level * (0.8 + 0.4 * np.random.random())
        self.media_influence += 0.2 * (media_target - self.media_influence) * self.params.time_step
        
        # Record history
        self.info_history.append((time, self.information_level))
        self.media_history.append((time, self.media_influence))
        
    def _uniform_information_update(self, cases: int) -> None:
        """
        Update information using uniform diffusion model.
        
        Args:
            cases: Current number of infectious cases
        """
        # Information increases with cases and decays over time
        case_fraction = cases / len(self.population)
        
        information_change = (
            self.params.information_generation_rate * case_fraction - 
            self.params.information_decay_rate * self.information_level
        ) * self.params.time_step
        
        self.information_level += information_change
        
    def update_policies(self, cases: int, time: float) -> None:
        """
        Update policy interventions based on current cases.
        
        Args:
            cases: Current number of infectious cases
            time: Current simulation time
        """
        # Simplified policy model - policies respond to case numbers
        case_fraction = cases / len(self.population)
        
        # Target policy levels based on case fraction
        policy_targets = np.zeros(3)
        
        # Social distancing
        policy_targets[0] = min(1.0, 2.0 * case_fraction)
        
        # Mask wearing
        policy_targets[1] = min(1.0, 1.5 * case_fraction)
        
        # Vaccination (increases more slowly but doesn't decrease)
        current_vax = self.policy_interventions[2]
        policy_targets[2] = max(current_vax, min(1.0, case_fraction))
        
        # Policies change gradually
        self.policy_interventions += 0.1 * (policy_targets - self.policy_interventions) * self.params.time_step
        
        # Record history
        self.policy_history.append((time, self.policy_interventions.copy()))
        
    def update_behaviors(self, time: float) -> None:
        """
        Update individual behaviors based on current information and policies.
        
        Args:
            time: Current simulation time
        """
        for ind in self.population:
            # Get gossip vulnerability if available
            vulnerability = 1.0
            if self.gossip_model:
                vulnerability = self.gossip_model.calculate_vulnerability(ind.idx)
                
            # Apply vulnerability to information field for this individual
            behavior_change = ind.update_behavior(
                self.information_level * vulnerability,  # More vulnerable = more responsive to information
                self.policy_interventions,
                self.media_influence,
                self.params
            )
            
            # Apply behavior change
            ind.behavior += behavior_change * self.params.time_step
            
            # Bound behavior
            ind.behavior = np.clip(ind.behavior, 0, 1)
            
            # Record history
            ind.behavior_history.append((time, ind.behavior.copy()))


class DataAssimilation:
    """Implements Bayesian data assimilation for model calibration."""
    
    def __init__(self, model, params: Parameters):
        """
        Initialize data assimilation module.
        
        Args:
            model: Reference to main model
            params: Model parameters
        """
        self.model = model
        self.params = params
        
        # Create ensemble of model states
        self.ensemble_size = params.ensemble_size
        self.ensemble = []
        
    def initialize_ensemble(self) -> None:
        """Initialize ensemble with perturbed model states."""
        # Create copies of the model with perturbed parameters
        for _ in range(self.ensemble_size):
            # Create parameter perturbation
            perturbed_params = self._perturb_parameters(self.params)
            
            # In a full implementation, we would create copies of the model
            # with perturbed parameters and initial conditions
            # For simplicity, we'll just store the parameters for now
            self.ensemble.append(perturbed_params)
    
    def _perturb_parameters(self, params: Parameters) -> Parameters:
        """Create perturbed copy of parameters."""
        # Copy parameters
        new_params = Parameters(
            population_size=params.population_size,
            simulation_days=params.simulation_days,
            time_step=params.time_step
        )
        
        # Perturb key parameters
        new_params.base_transmission_rate = params.base_transmission_rate * np.random.normal(1, 0.1)
        new_params.base_recovery_rate = params.base_recovery_rate * np.random.normal(1, 0.1)
        new_params.base_mortality_rate = params.base_mortality_rate * np.random.normal(1, 0.1)
        new_params.asymptomatic_fraction = params.asymptomatic_fraction * np.random.normal(1, 0.1)
        
        return new_params
    
    def assimilate_data(self, observed_data: Dict) -> None:
        """
        Assimilate new observation data to update model state.
        
        Args:
            observed_data: Dictionary of observed epidemic data
        """
        # In a full implementation, this would update the model state based on
        # the ensemble Kalman filter update equations
        # For simplicity, we'll just log the observation
        print(f"Assimilating data: {observed_data}")
        
        # Here we would:
        # 1. Forecast each ensemble member forward
        # 2. Compute the Kalman gain matrix
        # 3. Update each ensemble member based on observations
        # 4. Derive updated parameter estimates
        
        # For demonstration, just print the current ensemble size
        print(f"Current ensemble size: {len(self.ensemble)}")


class MANEpidemicModel:
    """Main epidemic model class implementing the MAN framework."""
    
    def __init__(self, params: Parameters = None):
        """
        Initialize the epidemic model.
        
        Args:
            params: Model parameters (uses defaults if None)
        """
        self.params = params if params else Parameters()
        self.time = 0.0
        self.initialized = False
        
    def initialize(self) -> None:
        """Set up the model for simulation."""
        # Create population
        self.population = self._create_population()
        
        # Create regions
        self.regions = self._create_regions()
        
        # Initialize networks
        self.contact_network = ContactNetwork(self.population, self.params)
        
        # Initialize pathogen model
        self.pathogen = PathogenModel(self.params)
        
        # Initialize behavior model
        self.behavior_model = BehaviorModel(self.population, self.params)
        
        # Initialize gossip model if using gossip diffusion mode
        if self.params.information_diffusion_mode == "gossip":
            print("Initializing gossip-based information diffusion model")
            self.gossip_model = GossipPropagation(self, self.params)
            self.behavior_model.gossip_model = self.gossip_model
            self.gossip_model.update_all_metrics()
            
            # Add tracking variables for gossip
            self.spread_factor_history = []
            self.spreading_time_history = []
        
        # Initialize data assimilation module
        self.data_assimilation = DataAssimilation(self, self.params)
        self.data_assimilation.initialize_ensemble()
        
        # Initialize tracking variables
        self.case_counts = {state: [] for state in HealthState}
        self.R_effective = []
        
        # Seed initial infections
        self._seed_infections()
        
        self.initialized = True
        print(f"Model initialized with {len(self.population)} individuals")
        
    def _create_population(self) -> List[Individual]:
        """Create the initial population."""
        population = []
        
        for i in range(self.params.population_size):
            # Generate age (bimodal distribution)
            if np.random.random() < 0.8:
                age = np.random.normal(40, 15)
            else:
                age = np.random.normal(75, 8)
            age = max(0, min(100, age))
            
            # Generate location (random in unit square)
            location = np.random.random(2) * 10
            
            # Generate risk factors (3-dimensional)
            risk_factors = np.random.random(3)
            
            # Generate initial behavior (3-dimensional)
            # [0]: Caution level, [1]: Mask compliance, [2]: Vaccination willingness
            behavior = np.random.random(3) * 0.2  # Start with low values
            
            individual = Individual(i, age, location, risk_factors, behavior)
            population.append(individual)
            
        return population
    
    def _create_regions(self) -> List[Region]:
        """Divide the space into regions."""
        # Create a simple grid of regions
        grid_size = 3  # 3x3 grid
        region_size = 10 / grid_size
        
        regions = []
        region_id = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Region center
                center_x = (i + 0.5) * region_size
                center_y = (j + 0.5) * region_size
                
                # Find individuals in this region
                region_pop = []
                for ind in self.population:
                    x, y = ind.location
                    if (i * region_size <= x < (i+1) * region_size and
                        j * region_size <= y < (j+1) * region_size):
                        region_pop.append(ind)
                
                # Create region
                region = Region(
                    region_id=f"R{region_id}",
                    population=region_pop,
                    location=(center_x, center_y),
                    radius=region_size / np.sqrt(2)
                )
                
                regions.append(region)
                region_id += 1
                
        return regions
    
    def _seed_infections(self) -> None:
        """Seed initial infections in the population."""
        # Number of initial infections (0.1% of population)
        n_initial = max(1, int(self.params.population_size * 0.001))
        
        # Randomly select individuals to infect
        initial_cases = np.random.choice(range(self.params.population_size), size=n_initial, replace=False)
        
        for idx in initial_cases:
            individual = self.population[idx]
            individual.health_state = HealthState.INFECTIOUS
            individual.state_history.append((0, HealthState.INFECTIOUS))
            
        print(f"Seeded {n_initial} initial infections")
    
    def _count_states(self) -> Dict[HealthState, int]:
        """Count individuals in each health state."""
        counts = {state: 0 for state in HealthState}
        
        for ind in self.population:
            counts[ind.health_state] += 1
            
        return counts
    
    def _compute_force_of_infection(self) -> np.ndarray:
        """
        Compute force of infection for each individual.
        
        Returns:
            Array of force of infection values
        """
        n = len(self.population)
        force = np.zeros(n)
        
        # Current transmissibility multiplier from pathogen evolution
        pathogen_factor = self.pathogen.transmissibility
        
        # Environmental factor (e.g., seasonality)
        time_in_years = self.time / 365
        seasonal_factor = 1.0 + self.params.seasonality_amplitude * np.sin(2 * np.pi * time_in_years)
        
        # Compute for each individual
        for i in range(n):
            # Skip non-susceptible individuals
            if self.population[i].health_state != HealthState.SUSCEPTIBLE:
                continue
                
            # Individual susceptibility factors
            age_i = self.population[i].age
            age_factor = 1.0 + 0.01 * max(0, age_i - 50)  # Older people more susceptible
            
            behavior_i = self.population[i].behavior
            caution_factor = np.exp(-2.0 * behavior_i[0])  # More cautious = less susceptible
            
            # Combined susceptibility
            susceptibility = age_factor * caution_factor
            
            # Sum over all contacts
            for j in range(n):
                # Skip non-infectious individuals
                if self.population[j].health_state not in [HealthState.INFECTIOUS, HealthState.ASYMPTOMATIC]:
                    continue
                    
                # Base transmission rate
                beta_0 = self.params.base_transmission_rate
                
                # Contact weight from effective network
                contact_weight = self.contact_network.effective_network[i, j]
                
                # Infectiousness factor
                if self.population[j].health_state == HealthState.ASYMPTOMATIC:
                    infectiousness = self.params.asymptomatic_transmission_factor
                else:
                    infectiousness = 1.0
                    
                # Behavior factors of infectious individual
                behavior_j = self.population[j].behavior
                mask_factor = np.exp(-1.5 * behavior_j[1])  # Mask wearing reduces transmission
                
                # Combined transmission rate
                beta_ij = (beta_0 * pathogen_factor * seasonal_factor * 
                         infectiousness * mask_factor * contact_weight)
                
                # Contribute to force of infection
                force[i] += beta_ij * susceptibility
                
                # Track potential transmission for R_effective calculation
                if force[i] > 0 and np.random.random() < self.params.time_step * beta_ij * susceptibility:
                    self.population[j].secondary_infections += 1
                    self.population[i].infection_source = j
            
        return force
    
    def _compute_r_effective(self) -> float:
        """
        Compute current effective reproduction number.
        
        Returns:
            Current R_effective value
        """
        # Count secondary infections for individuals who became infectious in last week
        # and have now recovered or died
        recent_recovered = [ind for ind in self.population 
                          if (ind.health_state in [HealthState.RECOVERED, HealthState.DECEASED] and 
                             ind.days_in_state <= 7)]
        
        if not recent_recovered:
            # If no recent recoveries, use previous R value or 0
            return self.R_effective[-1][1] if self.R_effective else 0
            
        secondary_infections = [ind.secondary_infections for ind in recent_recovered]
        r_effective = np.mean(secondary_infections)
        
        return r_effective
    
    def update_regions(self) -> None:
        """Update region information and resolution."""
        for region in self.regions:
            # Update compartmental counts
            region.update_compartments()
            
            # Determine appropriate resolution
            old_resolution = region.resolution
            new_resolution = region.determine_resolution(
                self.params.high_resolution_threshold,
                self.params.low_resolution_threshold
            )
            
            # Log resolution changes
            if old_resolution != new_resolution:
                print(f"Region {region.region_id} changed resolution: {old_resolution} -> {new_resolution}")
                region.resolution = new_resolution
    
    def simulate_step(self) -> None:
        """Simulate a single time step."""
        if not self.initialized:
            raise RuntimeError("Model must be initialized before simulation")
            
        # Update pathogen properties
        self.pathogen.update(self.time)
        
        # Update contact networks
        self.contact_network.update_networks(self.time)
        
        # Update gossip metrics periodically if using gossip diffusion
        if self.params.information_diffusion_mode == "gossip" and hasattr(self, 'gossip_model'):
            # Update gossip metrics every 5 time steps to save computation
            if int(self.time / self.params.time_step) % 5 == 0:
                self.gossip_model.update_all_metrics()
                
                # Record average gossip metrics
                avg_sf = np.mean(list(self.gossip_model.spread_factors.values()))
                avg_st = np.mean(list(self.gossip_model.spreading_times.values()))
                self.spread_factor_history.append((self.time, avg_sf))
                self.spreading_time_history.append((self.time, avg_st))
        
        # Update behaviors based on current information
        counts = self._count_states()
        infectious_count = counts[HealthState.INFECTIOUS] + counts[HealthState.ASYMPTOMATIC]
        
        self.behavior_model.update_information_field(infectious_count, self.time)
        self.behavior_model.update_policies(infectious_count, self.time)
        self.behavior_model.update_behaviors(self.time)
        
        # Compute force of infection
        force_of_infection = self._compute_force_of_infection()
        
        # Update individual states
        for i, individual in enumerate(self.population):
            individual.update_health_state(force_of_infection[i], self.params, self.time)
        
        # Update regions and their resolution
        self.update_regions()
        
        # Record current state
        counts = self._count_states()
        for state, count in counts.items():
            self.case_counts[state].append((self.time, count))
            
        # Compute and record R_effective
        r_eff = self._compute_r_effective()
        self.R_effective.append((self.time, r_eff))
        
        # Advance time
        self.time += self.params.time_step
        
    def simulate(self, duration: float = None) -> None:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation duration in days (uses params.simulation_days if None)
        """
        if not self.initialized:
            self.initialize()
            
        if duration is None:
            duration = self.params.simulation_days
            
        end_time = self.time + duration
        steps = int(duration / self.params.time_step)
        
        print(f"Starting simulation for {duration} days ({steps} steps)")
        
        for _ in tqdm(range(steps)):
            self.simulate_step()
            
        print(f"Simulation completed. Current time: {self.time:.1f} days")
    
    def plot_results(self) -> None:
        """Generate plots of simulation results."""
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        # Plot epidemic curves
        ax = axes[0]
        for state in [HealthState.SUSCEPTIBLE, HealthState.EXPOSED, HealthState.INFECTIOUS, 
                     HealthState.ASYMPTOMATIC, HealthState.RECOVERED, HealthState.DECEASED]:
            data = self.case_counts[state]
            times, counts = zip(*data)
            ax.plot(times, counts, label=state.name.capitalize())
            
        ax.set_ylabel('Number of individuals')
        ax.set_title('Epidemic Curves')
        ax.legend()
        ax.grid(True)
        
        # Plot R effective
        ax = axes[1]
        times, r_values = zip(*self.R_effective)
        ax.plot(times, r_values, 'r-')
        ax.axhline(y=1, color='k', linestyle='--')
        ax.set_ylabel('R effective')
        ax.set_title('Effective Reproduction Number')
        ax.grid(True)
        
        # Plot information and policy
        ax = axes[2]
        info_times, info_values = zip(*self.behavior_model.info_history)
        media_times, media_values = zip(*self.behavior_model.media_history)
        
        ax.plot(info_times, info_values, 'b-', label='Information')
        ax.plot(media_times, media_values, 'g-', label='Media')
        
        policy_times, policy_values = zip(*self.behavior_model.policy_history)
        policy_values = np.array(policy_values)
        
        ax.plot(policy_times, policy_values[:, 0], 'm--', label='Distancing Policy')
        ax.plot(policy_times, policy_values[:, 1], 'c--', label='Mask Policy')
        ax.plot(policy_times, policy_values[:, 2], 'y--', label='Vaccination Policy')
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Level')
        ax.set_title('Information and Policy Measures')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def save_results(self, filename: str) -> None:
        """
        Save simulation results to file.
        
        Args:
            filename: Output filename
        """
        results = {
            'params': self.params,
            'case_counts': self.case_counts,
            'R_effective': self.R_effective,
            'info_history': self.behavior_model.info_history,
            'media_history': self.behavior_model.media_history,
            'policy_history': self.behavior_model.policy_history,
            'pathogen_history': self.pathogen.history
        }
        
        # Add gossip data if available
        if hasattr(self, 'gossip_model') and hasattr(self, 'spread_factor_history'):
            results['spread_factor_history'] = self.spread_factor_history
            results['spreading_time_history'] = self.spreading_time_history
            
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
            
        print(f"Results saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Create parameters
    params = Parameters(
        population_size=1000,  # Smaller for testing
        simulation_days=100,
        time_step=0.1
    )
    
    # Create and run model
    model = MANEpidemicModel(params)
    model.initialize()
    model.simulate()
    
    # Plot and save results
    model.plot_results()
    model.save_results("epidemic_simulation_results.pkl")