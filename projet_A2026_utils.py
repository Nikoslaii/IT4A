from collections import Counter
from itertools import takewhile

from cdlib import algorithms
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random


def generate_planted_partition_graph(n, n_com, p_in, p_out, seed=None):
    """Génère un graphe à communautés plantées.

    Args:
        n: nombre total de sommets
        n_com: nombre de communautés
        p_in: probabilité d'arête entre deux sommets d'une même communauté
        p_out: probabilité d'arête entre deux sommets de communautés différentes
        seed: graine aléatoire

    Returns:
        Un graphe networkx, et un dictionnaire associant chaque sommet à sa
        communauté de référence.
    """

    G = nx.Graph()                                                                  #instancie un graphe vide
    nb_sommets_par_com = n // n_com                                                 #instancie le nombre de sommets par communauté
    random.seed(seed)                                                               #instancie le générateur de nombres aléatoires avec la graine fournie


    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            is_in_community = (i // nb_sommets_par_com) == (j // nb_sommets_par_com) #comparre si les deux sommets sont dans la même communauté
            is_out_community = not is_in_community                                   #en deduit si les deux sommets sont dans des communautés différentes

            if (is_in_community and random.random() < p_in) \
            or (is_out_community and random.random() < p_out):
                G.add_edge(i, j)

    return G, {i: i // nb_sommets_par_com for i in range(n)}                          #rend le graphe et un dictionnaire associant chaque sommet à sa communauté de référence


def load_real_graph(name):
    """Charge un graphe réel depuis un fichier local.

    Args:
        name: identifiant du jeu de données ("amazon", "cora", etc.)

    Returns:
        Le graphe et la partition de référence si elle est disponible.
    """

    Ground_truth = {}
    G = nx.DiGraph()

    with open(f"db/{name}", "r") as f:
        edges = [tuple(map(int, line.strip().split())) for line in f.readlines()]

    G.add_edges_from(edges)

    with open(f"db/{name}", "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            node_id = int(parts[0])
            label = parts[-1]
            Ground_truth[node_id] = label


    return G, Ground_truth

    raise NotImplementedError

def load_cora(directed=False):
    """Charge le jeu de données Cora.

    Télécharge l'archive si nécessaire, extrait les fichiers, construit le
    graphe et renvoie la partition de référence.

    Args:
        directed: si True, renvoie un graphe orienté

    Returns:
        (G, ground_truth) : graphe NetworkX et dictionnaire
        {sommet: étiquette de classe}
    """

    Ground_truth = {}
    G = nx.DiGraph() if directed else nx.Graph()

    with open("cora/cora.cites", "r") as f:
        edges = [tuple(map(int, line.strip().split())) for line in f.readlines()]

    G.add_edges_from(edges)

    with open("cora/cora.content", "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            node_id = int(parts[0])
            label = parts[-1]
            Ground_truth[node_id] = label


    return G, Ground_truth

def girvan_newman_communities(G, n_communities=None):
    """Détection de communautés par Girvan-Newman.

    Args:
        G: graphe networkx
        n_communities: nombre de communautés visé. Si None, on s'arrête au
            maximum de modularité.

    Returns:
        Liste de listes de sommets.

    aide https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.centrality.girvan_newman.html#networkx.algorithms.community.centrality.girvan_newman
    
    puis https://stackoverflow.com/questions/62951320/why-is-the-girvan-newman-algorithm-in-networkx-so-slow
    """
    import itertools


    graph=G[0].copy()   # Copie du graphe pour ne pas le modifier

    is_n_communities = n_communities is not None #instancie un booléen pour savoir si on a un nombre de communautés visé ou non


    best_partition = None #instancie la meilleure partition trouvée à None pour pouvoir la remplacer par la partition actuelle (is_n_communities True) ou pour pouvoir la comparer avec les partitions calculées (is_n_communities=False)
    best_modularity = -1 #instancie la meilleure modularité trouvée à -1 pour pouvoir la comparer avec les modularités calculées (is_n_communities False)
    
    comp = nx.community.girvan_newman(graph) #instancie un générateur de partitions de communautés avec l'algorithme de Girvan-Newman

    """
    #parcours les partitions de communautés générées par l'algorithme de Girvan-Newman, jusqu'à n_communities-1 si is_n_communities est True, sinon jusqu'à la fin du générateur
    for communities in itertools.islice(comp, n_communities-1 if is_n_communities else None): #stoppe l'itération si on a atteint le nombre de communautés visé (is_n_communities True) ou si on a épuisé le générateur (is_n_communities False)
        modularity_current = nx.community.modularity(graph, communities) #calcule la modularité de la partition actuelle

        if is_n_communities: #si on a un nombre de communautés visé, on remplace la meilleure partition trouvée par la partition actuelle
            best_partition = tuple(sorted(c) for c in communities) 
        else: #sinon, on compare la modularité de la partition actuelle avec la meilleure modularité trouvée, et on remplace la meilleure partition trouvée par la partition actuelle si la modularité actuelle est meilleure
            if modularity_current > best_modularity:
                best_modularity = modularity_current
                best_partition = tuple(sorted(c) for c in communities)

            else:
                print(f"Modularité actuelle : {modularity_current:.4f}, meilleure modularité : {best_modularity:.4f} fini")
                return best_partition #si la modularité actuelle est inférieure à la meilleure modularité trouvée, on arrête l'itération, car les partitions suivantes auront une modularité encore plus faible
    """


    
    if is_n_communities:
        limited = takewhile(lambda c: len(c) <= n_communities, comp)
        for communities in limited:
            best_partition = tuple(sorted(c) for c in communities)

    else:

        limited = takewhile(lambda c: nx.community.modularity(graph, c) > best_modularity, comp)
        for communities in limited:
            modularity_current = nx.community.modularity(graph, communities)
            if modularity_current > best_modularity:
                best_modularity = modularity_current
                best_partition = tuple(sorted(c) for c in communities)

            if len(best_partition) >= np.sqrt(G[0].number_of_nodes())//1 + 1:
                break
            
    return best_partition

def louvain_communities(G, resolution=1.0):
    """Détection de communautés par l'algorithme de Louvain.

    Args:
        G: graphe networkx
        resolution: paramètre de résolution de la modularité

    Returns:
        Liste de listes de sommets.

    aide https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html#networkx.algorithms.community.louvain.louvain_communities
    """
    
    return nx.community.louvain_communities(G[0], resolution=resolution) #renvoie la partition de communautés trouvée par l'algorithme de Louvain avec le graphe donné et le paramètre de résolution donné, en utilisant la fonction louvain_communities de NetworkX




def label_propagation_communities(G, max_iter=100):
    """Détection de communautés par propagation d'étiquettes.

    Args:
        G: graphe networkx
        max_iter: nombre maximal d'itérations

    Returns:
        Liste de listes de sommets.

    aide AI
    """
        # initialisation des labels de chaque noeud avec son propre index
    labels = {node: i for i, node in enumerate(G[0].nodes())}
    nodes = list(G[0].nodes())
    
    for _ in range(max_iter):
        labels_changed = False
        # mélange aléatoire des noeuds pour éviter les biais d'ordre
        random.shuffle(nodes)
        
        for node in nodes:
            # Récupère les voisins du noeud actuel
            neighbors = list(G[0].neighbors(node))
            if not neighbors:
                continue
                
            # compte les labels des voisins et trouve le label le plus fréquent
            neighbor_labels = [labels[nbr] for nbr in neighbors]
            top_label, top_count = Counter(neighbor_labels).most_common(1)[0]
            
            # si le label le plus fréquent est différent du label actuel, on met à jour le label et on note qu'il y a eu un changement
            if labels[node] != top_label:
                labels[node] = top_label
                labels_changed = True
                
        # si aucun label n'a changé lors de cette itération, on peut arrêter l'algorithme
        if not labels_changed:
            break
            
    # cree un dictionnaire de communautés à partir des labels finaux
    communities = {}
    for node, label in labels.items():
        communities.setdefault(label, set()).add(node)

    # renvoie la liste des communautés sous forme de listes de sommets
    return list(communities.values())    


def exact_modularity_maximization(G):
    """Recherche exacte de la partition de modularité maximale.

    Réservé aux très petites instances (n <= 10).
    """
    raise NotImplementedError


def modularity(G, communities, resolution=1.0):
    """Calcule la modularité d'une partition.

    Args:
        G: graphe networkx
        communities: liste de listes de sommets
        resolution: paramètre de résolution

    Returns:
        La modularité, valeur flottante.

    aide https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.quality.modularity.html
    """
    return nx.community.modularity(G[0], communities, resolution=resolution) #calcul la modularité de la partition donnée avec le graphe donné et le paramètre de résolution donné, en utilisant la fonction modularity de NetworkX


def plot_communities(G, communities, title="", with_labels=False):
    """Visualise une partition en communautés.

    Args:
        G: graphe networkx
        communities: liste de listes de sommets
        title: titre du graphique
    """

    graph = G[0]

    #fais avec les liste de sommets de chaque communauté un graphe avec les sommets de chaque communauté et les arêtes entre eux, en utilisant la fonction subgraph de NetworkX et en leur donnant une couleur par comunauté avec rajoute sur chaque points le numéro du sommet avec la fonction draw_networkx_labels de NetworkX
    
    
    nx.draw(graph, pos=nx.spring_layout(graph), node_color=[i for i, c in enumerate(communities) for _ in c], with_labels=with_labels)
    plt.title(title)
    plt.show() 

def benchmark_communities(G, algorithm, repeats=5):
    """Exécute un algorithme plusieurs fois et renvoie temps moyen et modularité.

    Args:
        G: graphe networkx
        algorithm: fonction prenant G en argument et renvoyant une partition
        repeats: nombre de répétitions

    Returns:
        Un couple (temps moyen, modularité moyenne).
    """

    Q = 0
    t0 = time.time()

    for _ in range(repeats):    
        # Détection avec l'algorithme donné
        com = algorithm(G)
        Q += modularity(G, com)


    return (time.time() - t0) / repeats, Q / repeats

    raise NotImplementedError