import numpy as np
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing
from python_tsp.exact import solve_tsp_dynamic_programming
import matplotlib.pyplot as plt
import sklearn.cluster as clst


def plot_routes(X, Y, locations, route, tsp_dis, wlk_dis=0):
    '''
    Plot the optimized tour
    
    Parameters
    ----------
    X : array-like size N
        x-coordinates of the locations
    Y : array-like size N
        y-coordinates of the locations
    locations : array-like shape (N, 2)
        2D array of the locations
    route : array-like shape (N,)
        optimized route
    tsp_dis : float
        total distance of the TSP tour
    wlk_dis : float
        total distance of the walk tour
    '''
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title('Optimized tour')
    # ax.scatter(X, Y)
    distance = 0.0
    N = locations.shape[0]
    start_node = 0
    for i in range(N):
        start_pos = locations[start_node]
        next_node = route[i+1] if i < N-1 else route[0]
        end_pos = locations[next_node]
        ax.annotate("",
                xy=start_pos, xycoords='data',
                xytext=end_pos, textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"))
        distance += np.linalg.norm(end_pos - start_pos)
        start_node = next_node
    textstr = "N nodes: %d\nTotal TSP length: %.3f\nTotal Wlk length: %.3f\nTotal length: %.3f" % (N, tsp_dis, wlk_dis, tsp_dis+wlk_dis)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, # Textbox
        verticalalignment='top', bbox=props)
    return ax



num_demand, magnitude = 50, 100
num_fixed_stops = 15
discount_ratio_array = np.arange(0.1, 1, 0.2)
num_demand_array = np.array([25, 50, 75, 100, 150, 200, 250, 300])
# num_fixed_stops_array = np.around(discount_ratio * sizes_array)
# num_fixed_stops_array = np.array([5, 15, 30, 50, 80, 100])


for num_demand in num_demand_array:
    ondmd_total_dises = []
    fx_total_dises, wlk_dises, tsp_dises = [], [], []
    num_stops = []
    X = magnitude*np.random.rand(num_demand)
    Y = magnitude*np.random.rand(num_demand)
    locations = np.stack((X, Y), axis=1)
    distance_matrix = euclidean_distance_matrix(locations)
    route, dis = solve_tsp_simulated_annealing(distance_matrix)

    # for num_fixed_stops in num_fixed_stops_array:
    for discount_ratio in discount_ratio_array:
        num_fixed_stops = int(np.around(num_demand * discount_ratio))
        num_stops.append(num_fixed_stops)
        if num_demand > num_fixed_stops:
            # ax1 = plot_routes(X, Y, locations, route, dis)
            # ax1.scatter(X, Y)

            kmeans = clst.KMeans(n_clusters=num_fixed_stops, random_state=0)
            clusters = kmeans.fit(locations)

            labels, centers = kmeans.labels_, kmeans.cluster_centers_
            center_X, center_Y = centers[:, 0], centers[:, 1]
            center_distance_matrix = euclidean_distance_matrix(centers)
            center_route, center_dis = solve_tsp_simulated_annealing(center_distance_matrix)
            # center_route, center_dis = solve_tsp_local_search(center_distance_matrix)
            center_ctgr = np.array([centers[i] for i in labels])
            wlk_dis = np.sum(np.linalg.norm(locations - center_ctgr, axis=1))
            # ax2 = plot_routes(center_X, center_Y, centers, center_route, center_dis, wlk_dis)
            # ax2.scatter(X, Y, s=9, c=labels)
            # ax2.scatter(center_X, center_Y, c='black')
            ondmd_total_dises.append(dis)
            fx_total_dises.append(center_dis + wlk_dis)
            wlk_dises.append(wlk_dis)
            tsp_dises.append(center_dis)


    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.plot(num_stops[:len(tsp_dises)], ondmd_total_dises, label="On demand total distance")
    ax1.plot(num_stops[:len(tsp_dises)], fx_total_dises, label="Fixed route total distance")
    ax1.plot(num_stops[:len(tsp_dises)], wlk_dises, label="Fixed route walk distance")
    ax1.plot(num_stops[:len(tsp_dises)], tsp_dises, label="Fixed route tsp distance")
    plt.xlabel('Num stops')
    plt.ylabel('Distances')
    ax1.legend()
    ax1.set_title(f'number of demands {num_demand} \'s distances')
    fig1.savefig(f'variant_num_stops/demand_size_{num_demand}.png')


# for num_fixed_stops in num_fixed_stops_array:
max_num_demand = num_demand_array[-1]
for discount_ratio in discount_ratio_array:
    num_fixed_stops = int(np.around(max_num_demand * discount_ratio))
    ondmd_total_dises = []
    fx_total_dises, wlk_dises, tsp_dises = [], [], []

    for num_demand in num_demand_array:
        if num_demand > num_fixed_stops:
            X = magnitude*np.random.rand(num_demand)
            Y = magnitude*np.random.rand(num_demand)
            locations = np.stack((X, Y), axis=1)
            distance_matrix = euclidean_distance_matrix(locations)
            route, dis = solve_tsp_simulated_annealing(distance_matrix)

            kmeans = clst.KMeans(n_clusters=num_fixed_stops, random_state=0)
            clusters = kmeans.fit(locations)

            labels, centers = kmeans.labels_, kmeans.cluster_centers_
            center_X, center_Y = centers[:, 0], centers[:, 1]
            center_distance_matrix = euclidean_distance_matrix(centers)
            center_route, center_dis = solve_tsp_simulated_annealing(center_distance_matrix)
            center_ctgr = np.array([centers[i] for i in labels])
            wlk_dis = np.sum(np.linalg.norm(locations - center_ctgr, axis=1))

            ondmd_total_dises.append(dis)
            fx_total_dises.append(center_dis + wlk_dis)
            wlk_dises.append(wlk_dis)
            tsp_dises.append(center_dis)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.plot(num_demand_array[len(num_demand_array) - len(tsp_dises):], ondmd_total_dises, label="On demand total distance")
    ax2.plot(num_demand_array[len(num_demand_array) - len(tsp_dises):], fx_total_dises, label="Fixed route total distance")
    ax2.plot(num_demand_array[len(num_demand_array) - len(tsp_dises):], wlk_dises, label="Fixed route walk distance")
    ax2.plot(num_demand_array[len(num_demand_array) - len(tsp_dises):], tsp_dises, label="Fixed route tsp distance")
    plt.xlabel('Num of demands')
    plt.ylabel('Distances')
    ax2.legend()
    ax2.set_title(f'{num_fixed_stops} with {discount_ratio} as discount stops \'s distances')
    fig2.savefig(f'variant_demand_size/{num_fixed_stops}_stops.png')
