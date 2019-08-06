"""
Created on Mon Jul  8 14:41:25 2019

@author: Mallory

HW 3
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics
import matplotlib.cm as cm  

# Decision making with Matrices

# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations. 

# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  Then you should decided if you should split into two groups so eveyone is happier.  

# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.
  
# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of decsion making problems that are currently not leveraging machine learning.  



# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.

# find random values for the weights for people dict 
print(np.array([np.random.dirichlet(np.ones(4),size=1)]))

  
people = {'Janet': {'willingness to travel': 0.3402,
                  'desire for new experience': 0.2488,
                  'cost': 0.0293,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.3815,
                  #'vegitarian':, 
                  },
'Mark': {'willingness to travel': 0.0193,
                  'desire for new experience': 0.0671,
                  'cost': 0.0336,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.8797,
                  #'vegitarian':,
                  },
'Dan': {'willingness to travel': 0.0768,
                  'desire for new experience': 0.0186,
                  'cost': 0.8544,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.0499,
                  #'vegitarian':, 
                  },
'Cheryl': {'willingness to travel': 0.1737,
                  'desire for new experience': 0.4280,
                  'cost': 0.2146,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.1835,
                  #'vegitarian':, 
                  },
'Jim': {'willingness to travel': 0.1251,
                  'desire for new experience': 0.4160,
                  'cost': 0.1235,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.3352,
                  #'vegitarian':, 
                  },
'Marci': {'willingness to travel': 0.1789,
                  'desire for new experience': 0.0062,
                  'cost': 0.0871,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.7275,
                  #'vegitarian':, 
                  },
'Max': {'willingness to travel': 0.1219,
                  'desire for new experience': 0.0427,
                  'cost': 0.2615,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.5737,
                  #'vegitarian':, 
                  },          
          }          

# Transform the user data into a matrix(M_people). Keep track of column and row ids.   
      # convert each person's values to a list
peopleKeys, peopleValues = [], []
lastKey = 0
for k1, v1 in people.items():
    row = []
    
    for k2, v2 in v1.items():
        peopleKeys.append(k1+'_'+k2)
        if k1 == lastKey:
            row.append(v2)      
            lastKey = k1
            
        else:
            peopleValues.append(row)
            row.append(v2)   
            lastKey = k1
            
# view the column keys and values
print(peopleKeys)
print(peopleValues)

# transform values to a matrix
peopleMatrix = np.array(peopleValues)
peopleMatrix.shape
peopleMatrix


# Next you collected data from an internet website. You got the following information.
# 1 is bad, 5 is great

resturants  = {'tacos2go':{'distance' : 3, 
                        'novelty' : 4,
                        'cost': 3,
                        #'average rating':, 
                        'cuisine': 2
                        #'vegitarians':,
                        },
'SpaghettiWagon':{'distance' : 5,
                        'novelty' :1,
                        'cost': 2,
                        #'average rating': ,
                        'cuisine':5
                        #'vegitarians':,
                        },
'HealthNut':{'distance' : 3, 
                        'novelty' : 1,
                        'cost':1, 
                        #'average rating': 
                        'cuisine':5
                        #'vegitarians'
                        },
'McDonalds':{'distance' : 5, 
                        'novelty' : 1,
                        'cost':5, 
                        #'average rating': 
                        'cuisine':1
                        #'vegitarians'
                        },
'PameraBread':{'distance' : 2, 
                        'novelty' : 3,
                        'cost':1, 
                        #'average rating': 
                        'cuisine':3
                        #'vegitarians'
                        },
'PizzaHut':{'distance' : 4, 
                        'novelty' : 1,
                        'cost':2, 
                        #'average rating': 
                        'cuisine':3
                        #'vegitarians'
                        },
'Chickfila':{'distance' : 4, 
                        'novelty' : 2,
                        'cost':5, 
                        #'average rating': 
                        'cuisine':5
                        #'vegitarians'
                        },
}


# Transform the restaurant data into a matrix(M_resturants) use the same column index.

restaurantsKeys, restaurantsValues = [], []

for k1, v1 in resturants.items():
    for k2, v2 in v1.items():
        restaurantsKeys.append(k1+'_'+k2)
        restaurantsValues.append(v2)

# view column keys and values
print(restaurantsKeys)
print(restaurantsValues)

# look at the length. we will need to reshape to 7 rows and 4 columns
len(restaurantsValues)

# converting list to np.arrays and reshape it
restaurantsMatrix = np.reshape(restaurantsValues, (7,4))

# view the Matrix and its shape
restaurantsMatrix
restaurantsMatrix.shape

# get the resturant and people matrix in the correct form for later matric multiplication
    # need to swap axis on peopleMatrix: so will be (3,4)X(4,3)
restaurantsMatrix.shape, peopleMatrix.shape

#https://docs.scipy.org/doc/numpy/reference/generated/numpy.swapaxes.html
newPeopleMatrix = np.swapaxes(peopleMatrix, 0, 1)

restaurantsMatrix.shape, newPeopleMatrix.shape

# The most imporant idea in this project is the idea of a linear combination.  
# Informally describe what a linear combination is  and how it will relate to our resturant matrix.

print('A linear combination is a mathematical expression constructed by multiplying each term in a set of term by a constant and then adding the results.')
    # from https://en.wikipedia.org/wiki/Linear_combination
print('For our purposes, we will probably be doing a linear combination of our matrices. We will probably be multiplying'
    ' our people matrix values by the constants from the resturant matrix and summing the results to create an overall'
    ' measurement of how well a particular person would like that resturant.')

# Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent. 

# view the data
print(peopleKeys)
print(peopleValues)

newPeopleMatrix # remember this one reformatted for multiplication

print(restaurantsKeys)
restaurantsMatrix

# Do the linear combination manually first for one person to find the top resturant

# Dan's score for tacos2go
print('Dans score for tacos2go is:', round((3*0.0768 + 4*0.0186 + 3*0.8544 + 2*0.0499),2), ".")

# Dan's score for SpaghettiWagon
print('Dans score for SpaghettiWagon is:',round((5*0.0768 + 1*0.0186 + 2*0.8544 + 5*0.0499),2), ".")

# Dan's score for HealthNut
print('Dans score for HealthNut is:',round((3*0.0768 + 1*0.0186 + 1*0.8544 + 5*0.0499),2), ".")

# Dan's score for McDonalds
print('Dans score for McDonalds is:',round((5*0.0768 + 1*0.0186 + 5*0.8544 + 1*0.0499),2), ".")

# Dan's score for PameraBread
print('Dans score for PameraBread is:',round((2*0.0768 + 3*0.0186 + 1*0.8544 + 3*0.0499),2), ".")

# Dan's score for PizzaHut
print('Dans score for PizzaHut is:',round((4*0.0768 + 1*0.0186 + 2*0.8544 + 3*0.0499),2), ".")

# Dan's score for Chickfila
print('Dans score for Chickfila is:',round((4*0.0768 + 2*0.0186 + 5*0.8544 + 5*0.0499),2), ".")

print('The resulting value from the linear combination is Dans overall score for each resturant. Dans top resturant has the highest score, which is ChickfilA at 4.87')


# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent? 

# matrix multiply the two matrices and view the resulting matrix 
results = np.matmul(restaurantsMatrix, newPeopleMatrix)
results 

print('the new results matrix represents the overall score for each resturant for each person! 7 resturants and 7'
      ' people, which results in a 7X7 matrix.')
      

# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?

#  sum the columns of the results matrix for an overall resturant score for all people
np.sum(results, axis=1)

print('Summing the columns of the results matrix represents the overall score of each resturant for all people.'
      ' The resturant with the highest score is the resturant that would best satisfy everyone.')

# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal resturant choice.  
results

# Say that rank 1 is best
    # Argsort returns the indices that would sort an array - https://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
sortedResults1 = results.argsort()[::-1] +1
sortedResults1
# Say that rank 0 is best
sortedResults0 = results.argsort()[::-1]
sortedResults0

# find the optimal resturant choice
np.sum(sortedResults1, axis=0)
np.sum(sortedResults0, axis=0)

print('From summing the rankings of the matrices, it looks like the optimal resturant choice is SpaghettiWagon.')

# Why is there a difference between the two?  What problem arrives?  What does represent in the real world?

# first plot heatmap
plot_dims = (16,14)
fig, ax = plt.subplots(figsize=plot_dims)
sns.heatmap(ax=ax, data=results, annot=True)
plt.show()

# view the keys again
print(peopleKeys)
print(restaurantsKeys)

# view the shapes
print(results.shape)
print(peopleMatrix.shape)
peopleMatrix
results

print('The problem is that these values are and not scaled. They need to be scaled so they are relative to each other. An additional problem is it is not super'
      ' useful to view the data this way, it would be better to view it in a cluster, rather than a ranked or summed matrix. The Argsort function'
      ' also provides the indices that sort the matrix, not the rankings, and it sorts by rows by default.')

# How should you preprocess your data to remove this problem. 
print('The data needs to be scaled or normalized before being used in any model. The data will then be clustered to identify the similar people and resturants.')

''' Begin Clustering'''

# perform PCA on the people data to find the principal components with the more explained variance
    # source: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
pca = PCA(n_components=2)  
peopleMatrixPcaTransform = pca.fit_transform(peopleMatrix)  

print(pca.components_)
print(pca.explained_variance_)

print('It looks like the two principal components do not describe a lot of variation, which means the people attributes dont very a lot.') 

# Show the PC arrows
    # source: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# plot principal components
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

ax.scatter(peopleMatrixPcaTransform[:, 0], peopleMatrixPcaTransform[:, 1], alpha=0.2)
draw_vector([0, 0], [0, 1], ax=ax)
draw_vector([0, 0], [1, 0], ax=ax)
ax.axis('equal')
ax.set(xlabel='component 1', ylabel='component 2',
          title='principal components',
          xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
fig.show

print('This plot shows that there is more variability in the x axis than the y axis (in PC 1 vs PC 2).'
      ' The reason to do PCA is to reduce the variable to just 2 for easier cluster plotting, so we dont'
      ' have to plot 7D clusters. It is much easier to plot and interpret x vs y!') 

# Use peoplePCA for clustering and plotting
    # source: https://scikit-learn.org/stable/modules/clustering.html 
kmeans = KMeans(n_clusters=4)
kmeans.fit(peopleMatrixPcaTransform)

centroid = kmeans.cluster_centers_
labels = kmeans.labels_

print (centroid)
print(labels)


fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

#https://matplotlib.org/users/colors.html
colors = ["g.","r.","c.","b."]
labelList = ['Janet', 'Mark', 'Dan', 'Cheryl', 'Jim', 'Marci', 'Max']

for i in range(len(peopleMatrixPcaTransform)):
   print ("coordinate:" , peopleMatrixPcaTransform[i], "label:", labels[i])
   ax.plot(peopleMatrixPcaTransform[i][0],peopleMatrixPcaTransform[i][1],colors[labels[i]],markersize=10)
   # https://matplotlib.org/users/annotations_intro.html
   # https://matplotlib.org/users/text_intro.html
   ax.annotate(labelList[i], (peopleMatrixPcaTransform[i][0],peopleMatrixPcaTransform[i][1]), size=25)
ax.scatter(centroid[:,0],centroid[:,1], marker = "x", s=150, linewidths = 5, zorder =10)

plt.show()

# Resturant Clustering

# PCA for the resturants
    # Souce: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
restaurantsMatrix.shape

pca = PCA(n_components=2)  
restaurantsMatrixPcaTransform = pca.fit_transform(restaurantsMatrix)  

print(pca.components_)
print(pca.explained_variance_)


fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

ax.scatter(restaurantsMatrixPcaTransform[:, 0], restaurantsMatrixPcaTransform[:, 1], alpha=0.2)
draw_vector([0, 0], [0, 3], ax=ax)
draw_vector([0, 0], [3, 0], ax=ax)
ax.axis('equal')
ax.set(xlabel='component 1', ylabel='component 2',
          title='principal components',
          xlim=(-4, 4), ylim=(-4, 4))
fig.show


# Resturant PCA plotting
    # Source: https://scikit-learn.org/stable/modules/clustering.html
kmeans = KMeans(n_clusters=4)
kmeans.fit(restaurantsMatrixPcaTransform)

centroid = kmeans.cluster_centers_
labels = kmeans.labels_

print (centroid)
print(labels)


fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

#https://matplotlib.org/users/colors.html
colors = ["g.","r.","c.","b."]
labelList = ['tacos2go', 'SpaghettiWagon', 'HealthNut', 'McDonalds', 'PameraBread', 'PizzaHut', 'Chickfila']

for i in range(len(restaurantsMatrixPcaTransform)):
   print ("coordinate:" , restaurantsMatrixPcaTransform[i], "label:", labels[i])
   ax.plot(restaurantsMatrixPcaTransform[i][0],restaurantsMatrixPcaTransform[i][1],colors[labels[i]],markersize=10)
   #https://matplotlib.org/users/annotations_intro.html
   #https://matplotlib.org/users/text_intro.html
   ax.annotate(labelList[i], (restaurantsMatrixPcaTransform[i][0],restaurantsMatrixPcaTransform[i][1]), size=25)
ax.scatter(centroid[:,0],centroid[:,1], marker = "x", s=150, linewidths = 5, zorder =10)

plt.show()


# Find  user profiles that are problematic, explain why?
print('From the above people cluster analysis, it is immediately obvious that there is one outlier profile: Dan.'
      ' There seem to be two distinct groups of people, but Dan is off on his own in the far bottom right corner.'
      ' Because Dan is an outlier in the group of people preferences, he will most likely be hard to please'
      ' as he has very different resturant preferences than the other people.')

# Hierarchical Clustering for People Matrix
# this can be used to see how many clusters may be appropriate
    # source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

pca = PCA(n_components=2)  
peopleMatrixPcaTransform = pca.fit_transform(peopleMatrix)  

# heirarchical clustering
linked = linkage(peopleMatrixPcaTransform, 'single')


labelList = ['Janet', 'Mark', 'Dan', 'Cheryl', 'Jim', 'Marci', 'Max']

# explicit interface
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1)
dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True, ax=ax)
ax.tick_params(axis='x', which='major', labelsize=25)
ax.tick_params(axis='y', which='major', labelsize=25)
plt.show()  

print('From the cluster dendogram, it looks like an appropriate number of clusters is 3 for the people data.'
      ' This is based off drawing a horizontal line and seeing how many vertical lines you pass through.'
      ' Any more than three clusters would result in mulitple people being in their own cluster, and that doesnt'
      ' make a lot of sense since we are trying to group similar people.')

# Hierarchical Clustering for Resturant Matrix
# this can be used to see how many clusters may be appropriate
pca = PCA(n_components=2)  
restaurantsMatrixPcaTransform = pca.fit_transform(restaurantsMatrix)  

linked = linkage(restaurantsMatrixPcaTransform, 'single')

labelList = ['tacos2go', 'SpaghettiWagon', 'HealthNut', 'McDonalds', 'PameraBread', 'PizzaHut', 'Chickfila']

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)
dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True, ax=ax)
ax.tick_params(axis='x', which='major', labelsize=25)
ax.tick_params(axis='y', which='major', labelsize=25)
plt.show()

print('From the cluster dendogram, it looks like an appropriate number of clusters is also 3 for the resturant data.'
      ' This is based off drawing a horizontal line and seeing how many vertical lines you pass through.'
      ' Any more than three clusters would result in mulitple resturants being in their own cluster, and that doesnt'
      ' make a lot of sense since we are trying to group similar resturants.')

# Think of two metrics to compute the disatistifaction with the group.  

# Validate the people clusters with two different metrics 
    # Source: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

# Calinski-Harabaz
print("The Calinski-Harabaz Index is used to measure better defined clusters.")
print("\nThe Calinski-Harabaz score is higher when clusters are dense and well separated.\n")

import warnings
warnings.filterwarnings("ignore")

range_n_clusters = [2, 3, 4, 5, 6]
for n_clusters in range_n_clusters:
     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
     cluster_labels = clusterer.fit_predict(peopleMatrixPcaTransform)
     score = metrics.calinski_harabaz_score(peopleMatrixPcaTransform, cluster_labels)  
     print("The Calinski-Harabaz score for :", n_clusters, " clusters is: ", score)
     
print('The Calinski-Harabaz score shows that 6 clusters is the best, since it is the highest score.'
      ' This contradicts what we saw when using the visual inspection and cluster dendogram.'
      ' This would suggest that the people will be the most satisfied when they are grouped in 6 groups because'
      ' 6 is the most optimal number of clusters, according to this metric.')     
     
print("The Davies-Bouldin Index is used to measure better defined clusters.")
print("\nThe Davies-Bouldin score is lower when clusters more separated (e.g. better partitioned).\n")
print("Zero is the lowest possible Davies-Bouldin score.\n")

import warnings
warnings.filterwarnings("ignore")

range_n_clusters = [2, 3, 4, 5, 6]
for n_clusters in range_n_clusters:
     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
     cluster_labels = clusterer.fit_predict(peopleMatrixPcaTransform)
     score = metrics.davies_bouldin_score(peopleMatrixPcaTransform, cluster_labels)  
     print("The Davies-Bouldin score for :", n_clusters, " clusters is: ", score)
     
print('The Davies-Bouldin score shows that 6 clusters is the best, since it is the highest score.'
      ' This contradicts what we saw when using the visual inspection and cluster dendogram, but is the same result as the'
      ' Calinski-Harabaz metric. This would suggest that the people will be the most satisfied when they are grouped in 6 groups because'
      ' 6 is the most optimal number of clusters, according to this metric.') 

# Should you split in two groups today? 

# looking another way to validate clusters, besides the two metrics and the dendogram
# Using Silhouette Analysis with Kmeans Clustering on the PCA transformed People Matrix
    # Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(peopleMatrixPcaTransform) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(peopleMatrixPcaTransform)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = metrics.silhouette_score(peopleMatrixPcaTransform, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = metrics.silhouette_samples(peopleMatrixPcaTransform, cluster_labels)
    
    # The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. 
    # Scores around zero indicate overlapping clusters.
    # The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.

    print("\n\n\nFor n_clusters =", n_clusters,
          "\n\nThe average silhouette_score is :", silhouette_avg,
          "\n\n* The silhouette score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.",
          "\n* Scores around zero indicate overlapping clusters.",
          "\n* The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster",
          "\n\nThe individual silhouette scores were :", sample_silhouette_values,
          "\n\nAnd their assigned clusters were :", cluster_labels,
          "\n\nWhich correspond to : 'Jane', 'Bob', 'Mary', 'Mike', 'Alice', 'Skip', 'Kira', 'Moe', 'Sara', and 'Tom'")
    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.rainbow(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.9)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.", fontsize=20)
    ax1.set_xlabel("The silhouette coefficient values", fontsize=20)
    ax1.set_ylabel("Cluster label", fontsize=20)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)


    # 2nd Plot showing the actual clusters formed
    colors = cm.rainbow(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(peopleMatrixPcaTransform[:, 0], peopleMatrixPcaTransform[:, 1], marker='.', s=300, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=400, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=400, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.", fontsize=20)
    ax2.set_xlabel("Feature space for the 1st feature", fontsize=20)
    ax2.set_ylabel("Feature space for the 2nd feature", fontsize=20)

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=25, fontweight='bold')
        
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)

plt.show()

print('The Silhouette analysis, my favorite way of analyzing clusters, shows that 3 clusters is the most optimal number as it has the highest average'
      ' silhoutte score at about 0.5. This means that most observations were clustered correctly with the maximum distance between clusters.'
      ' These results confirm the visual understanding before, that 3 clusters for the group of people is optimal. It does not make since to separate almost every'
      ' person into their own cluster, as the Calinski-Harabaz and Davies-Bouldin metrics suggested. Unless the plan is for every single person to eat alone!'
      ' That is why it important to visualize the clusters several ways and use multiple validation metrics. It is easy in this case to visualize the clusters'
      ' since it is a 2D plot from the 2 principal components, but it is harder when you are clustering on more variables. In this case you may prefer to use'
      ' one of the other metrics to analyze the clusters if you cannot easily plot them.')
print('From analyzing the clusters of the people, it looks like the group of people would be happiest if they split into three groups. This way people would be'
      ' grouped according to the similarity of their resturant preferences. However, Dan is alone in this scenario since he is an outlier. It may be better to'
      ' split into two groups and Dan can choose to join one of those groups so he doesnt have to eat alone! :) ')
print('You can do the same cluster analysis with the resturant groups, but I wasnt sure what the question was asking, and it makes more sense to me'
      ' to analyze the people clusters.')


# Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?

print('If the boss is no longer paying for a meal, then you should re-do the analysis with cost as having no weight/no significance on the peoples resturant choice.'
      'That way you dont have to change the resturant information, just the people survey information.')

people2 = {'Janet': {'willingness to travel': 0.3402,
                  'desire for new experience': 0.2488,
                  'cost': 0,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.3815,
                  #'vegitarian':, 
                  },
'Mark': {'willingness to travel': 0.0193,
                  'desire for new experience': 0.0671,
                  'cost': 0,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.8797,
                  #'vegitarian':,
                  },
'Dan': {'willingness to travel': 0.0768,
                  'desire for new experience': 0.0186,
                  'cost': 0,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.0499,
                  #'vegitarian':, 
                  },
'Cheryl': {'willingness to travel': 0.1737,
                  'desire for new experience': 0.4280,
                  'cost': 0,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.1835,
                  #'vegitarian':, 
                  },
'Jim': {'willingness to travel': 0.1251,
                  'desire for new experience': 0.4160,
                  'cost': 0,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.3352,
                  #'vegitarian':, 
                  },
'Marci': {'willingness to travel': 0.1789,
                  'desire for new experience': 0.0062,
                  'cost': 0,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.7275,
                  #'vegitarian':, 
                  },
'Max': {'willingness to travel': 0.1219,
                  'desire for new experience': 0.0427,
                  'cost': 0,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.5737,
                  #'vegitarian':, 
                  },          
          }          

# Transform the user data into a matrix(M_people). Keep track of column and row ids.   
      # convert each person's values to a list
peopleKeys, peopleValues = [], []
lastKey = 0
for k1, v1 in people2.items():
    row = []
    
    for k2, v2 in v1.items():
        peopleKeys.append(k1+'_'+k2)
        if k1 == lastKey:
            row.append(v2)      
            lastKey = k1
            
        else:
            peopleValues.append(row)
            row.append(v2)   
            lastKey = k1
            
# view the column keys and values
print(peopleKeys)
print(peopleValues)

# transform values to a matrix
peopleMatrix = np.array(peopleValues)
peopleMatrix.shape
peopleMatrix

# change axis
newPeopleMatrix = np.swapaxes(peopleMatrix, 0, 1)

restaurantsMatrix.shape, newPeopleMatrix.shape

# matrix multiply the two matrices and view the resulting matrix 
# the new results matrix represents the overall score for each resturant for each person!
results = np.matmul(restaurantsMatrix, newPeopleMatrix)
results 

#  sum the columns of the results matrix for an overall resturant score for all people
np.sum(results, axis=1)

print('Summing the columns of the new results matrix (without price as a consideration) represents the overall score of each resturant for all people.'
      ' The resturant with the highest score is the resturant that would best satisfy everyone.'
      ' From this new sum it looks like the best resturant (not considering price) is Chickfila with a score of 22.25.')

# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix? 

### Trying to answer if we can find weights from rankings
# redo the people matrix with the cost weights
people = {'Janet': {'willingness to travel': 0.3402,
                  'desire for new experience': 0.2488,
                  'cost': 0.0293,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.3815,
                  #'vegitarian':, 
                  },
'Mark': {'willingness to travel': 0.0193,
                  'desire for new experience': 0.0671,
                  'cost': 0.0336,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.8797,
                  #'vegitarian':,
                  },
'Dan': {'willingness to travel': 0.0768,
                  'desire for new experience': 0.0186,
                  'cost': 0.8544,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.0499,
                  #'vegitarian':, 
                  },
'Cheryl': {'willingness to travel': 0.1737,
                  'desire for new experience': 0.4280,
                  'cost': 0.2146,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.1835,
                  #'vegitarian':, 
                  },
'Jim': {'willingness to travel': 0.1251,
                  'desire for new experience': 0.4160,
                  'cost': 0.1235,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.3352,
                  #'vegitarian':, 
                  },
'Marci': {'willingness to travel': 0.1789,
                  'desire for new experience': 0.0062,
                  'cost': 0.0871,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.7275,
                  #'vegitarian':, 
                  },
'Max': {'willingness to travel': 0.1219,
                  'desire for new experience': 0.0427,
                  'cost': 0.2615,
                  #'indian food':,
                  #'mexican food':,
                  'hipster points': 0.5737,
                  #'vegitarian':, 
                  },          
          }          

# Transform the user data into a matrix(M_people). Keep track of column and row ids.   
      # convert each person's values to a list
peopleKeys, peopleValues = [], []
lastKey = 0
for k1, v1 in people.items():
    row = []
    
    for k2, v2 in v1.items():
        peopleKeys.append(k1+'_'+k2)
        if k1 == lastKey:
            row.append(v2)      
            lastKey = k1
            
        else:
            peopleValues.append(row)
            row.append(v2)   
            lastKey = k1
            
# view the column keys and values
print(peopleKeys)
print(peopleValues)

# transform values to a matrix
peopleMatrix = np.array(peopleValues)
peopleMatrix.shape
peopleMatrix

newPeopleMatrix = np.swapaxes(peopleMatrix, 0, 1)

results = np.matmul(restaurantsMatrix, newPeopleMatrix) # used the original people dict with the cost weights
results                             

newPeopleMatrix.shape


# Rank 1 is highest
    # reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html

#np.set_printoptions(threshold=np.nan)

sortedResults = results.argsort()[::-1]
sortedResults.shape

np.sort(results)


# CAN WE GO BACKWARDS AND FIND WEIGHTS?
    # Source: #https://en.wikipedia.org/wiki/Invertible_matrix
# If we know restaurantsMatrix and Results Matrix can we calculate peopleMatrix?
# If the matrix A is invertible, then the equation Ax=b has a unique solution, namely x=A−1b.
# If A is not invertible, there may be either zero or many solutions to your problem.
  
    # Notes from Chris
#Ax = b 
#b is results.
#x is people weights
#A is restaurantsMatrix
#if A was and n*n matrix
#then, x = A^-1*b

# pinv returns the inverse of your matrix (A) when it is available and the pseduo inverse when A isn't
# an n by n matrix.

# The pseudo inverse of a matrix A, A^+ is the matrix that solves Ax=b
# if x is the solution, then A^+ is the matrix such that xbar = (A^+)(b)

b = results

ainv = np.linalg.pinv(restaurantsMatrix)
#x is an approximation of the peopleMatrix
x = np.matmul(ainv, b)
x.shape
x = np.swapaxes(x,0,1)
x.shape

#show how similar they are
peopleMatrix
x

#https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
#They are the same 
np.allclose(peopleMatrix, x, rtol=1e-14, atol=1e-14, equal_nan=False)


#The Moore-Penrose pseudoinverse is a matrix that can act as a partial replacement for the matrix inverse 
#in cases where it does not exist. This matrix is frequently used to solve a system of linear equations when 
#the system does not have a unique solution or has many solutions.

#For any matrix A, the pseudoinverse B exists, it is unique, and has the same dimensions as A' (the transpose matrix of A). 

# If A is a 3 x 2 matrix:    
A = np.array([[111, 222],[333, 444],[555, 666]])                                                                                           

A.shape
  
# A' is a 2 x 3 matrix (A transpose):    
Atranspose = A.T          
Atranspose.shape            
# This means that the pseudoinverse B would have the same dimensions (2 x 3) as A transpose.

                                                                     
#If A is square and not singular (e.g. matrices are singular if and only if their determinant is 0),  
#(also: Note that determinants are defined only for square matrices.)
#http://mathworld.wolfram.com/SingularMatrix.html 
#http://mathworld.wolfram.com/Determinant.html
#Determinant Intuition: https://math.stackexchange.com/questions/668/whats-an-intuitive-way-to-think-about-the-determinant

#then pinv(A), A^+, is simply an expensive way to compute inv(A) (when A is square and not singular). 
#However, if A is not square, or is square and singular, then inv(A) does not exist. 

#The pinv(A) is computed through singular value decomposition
#https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Singular_value_decomposition_(SVD)
#https://en.wikipedia.org/wiki/Singular_value_decomposition

print('It is difficult to go from the rank matrix to the results matrix. It would be easy to use the pinv of the resturant matrix'
      ' and the results matrix to find the people weights matrix. But we do not have results, we only have the ranking of the results.'
      ' Ranking the results matrix results in a loss of information. You may still be able to gain some information from clusting the data.')



