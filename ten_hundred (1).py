'''
Author: Akshat Raika
Date Edited: 4/4/2020
Project: Clustering of countries/regions based on Growth Trend of COVID-19 
'''

import csv, math, numpy

''' reads data from the give file and returns a list'''
def load_data(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        data = list(reader)
        # REMOVE THE HEADER
        data.remove(data[0])
        return data

'''takes in one row from the data loaded from the previous function, 
    calculates the corresponding x, y values for that region as specified in the video, 
    and returns them in a single structure.'''
def calculate_x_y(time_series):
    count = len(time_series)
    n = int(time_series[count-1])
    x = None
    y = None
    if int(time_series[count-1]) == 0:
        x = y = math.nan
        return (x,y)
    for i in range(count):
        curr = time_series[count-i-1]
        if curr.isnumeric():
            curr = int(curr)
            if curr <= n/10 and x == None:
                x = i
            if curr <= n/100 and y == None:
                y = abs(i - x)
            if x != None and y != None:
                return (x, y)
    return None


# returns minimum euclidian distance between clusters
def dist(c1, c2):
    min_dist = 10000
    for n1 in c1.nodes:
        for n2 in c2.nodes:
            d = math.sqrt((n1[0]-n2[0])*(n1[0]-n2[0]) + (n1[1]-n2[1])*(n1[1]-n2[1]))
            if d < min_dist:
                min_dist = d
    return min_dist

class Cluster:
    def __init__(self, idx):
        self.nodes = []
        self.index = idx

def HAC(dataset):
    ''' RETURNS: An (m-1) by 4 matrix Z. At the i-th iteration, 
    clusters with indices Z[i, 0] and Z[i, 1] are combined to form cluster m + i. 
    A cluster with an index less than m corresponds to one of the m original observations. 
    The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. 
    The fourth value Z[i, 3] represents the number of original observations in the newly formed cluster.'''
    clusters = []
    Z = []
    
    # initialize clusters (singleton clusters)
    for i in range(len(dataset)):
        tempZ = [0,0,0,1]
        Z.append(tempZ)
        cl = Cluster(i)
        cl.nodes.append(dataset[i])
        clusters.append(cl)
    while(len(clusters) > 1):
        # find closest clusters
        # add a row to Z
        # merge them
        # remove those clusters from the list
        min_dist = 10000
        c1 = None
        c2 = None
        for i in range(len(clusters)):
            for j in range(len(clusters)):
                if i != j:
                    d = dist(clusters[i], clusters[j])
                    #     tie breaking:
                    #     Given a set of pairs with equal distance {(xi, xj)} where i < j, 
                    #     we prefer the pair with the smallest first cluster index i. 
                    #     If there are still ties (xi, xj), ... (xi, xk) where i is that smallest first index, 
                    #     we prefer the pair with the smallest second cluster index.
                    if d <= min_dist:
                        if d == min_dist:
                            if clusters[i].index <= c1.index:
                                if clusters[i].index == c1.index:
                                    if clusters[j].index < c2.index:
                                        min_dist = d
                                        c1 = clusters[i]
                                        c2 = clusters[j]
                                else:
                                    min_dist = d
                                    c1 = clusters[i]
                                    c2 = clusters[j]
                        else:
                            min_dist = d
                            c1 = clusters[i]
                            c2 = clusters[j]
                            
        Z.append([c1.index, c2.index, min_dist, len(c1.nodes)+len(c2.nodes)])
        new_cluster = Cluster(len(Z)-1)
        new_cluster.nodes = c1.nodes.copy() + c2.nodes.copy()
        clusters.remove(c1)
        clusters.remove(c2)
        clusters.append(new_cluster)
    return numpy.asmatrix(Z[-(len(dataset)-1):])

if __name__ == "__main__":
    df = load_data("time_series_covid19_confirmed_global.csv")
    coords = []
    for i in range(len(df)):
        temp = df[i]
        item = calculate_x_y(temp)
        if item != None:
            if not( math.isnan(item[0]) or math.isnan(item[1])):
                coords.append(item)
    print(HAC(coords))