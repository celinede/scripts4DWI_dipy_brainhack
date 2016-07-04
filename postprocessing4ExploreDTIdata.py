
# coding: utf-8

# Import data

# In[2]:

def importData(ftrk,ffa,fdwi):  
    import nibabel as nib
    
    fa_img = nib.load(ffa)
    fa = fa_img.get_data()
    affine = fa_img.get_affine()

    img = nib.load(fdwi)
    data = img.get_data()

    from nibabel import trackvis
    streams, hdr = trackvis.read(ftrk)
    streamlines = [s[0] for s in streams]
    
    return (fa,affine,data,streamlines)


# Length infos

# In[3]:

def getLengths(streamlines):
    from dipy.tracking.utils import length
    lengths = list(length(streamlines))

    nb_stl = len(streamlines)
    min_len = min(length(streamlines))
    max_len = max(length(streamlines))

#print('Nb. streamlines:')
#   print(nb_stl)
#   print('Min length:')
#   print(min_len)
#   print('Max length:')
#print(max_len)
    
    return lengths


# Filter with length

# In[4]:

def filterLength(streamlines, thr_len):
    from dipy.tracking.utils import length
    new_streamlines = [ s for s, l in zip(streamlines, getLengths(streamlines)) if l > thr_len ] #3.5 #2.5

    #new_streamlines_l = list(new_streamlines)
    new_lengths = list(length(new_streamlines))
    # print('Nb. new streamlines:')
    #print(len(new_streamlines))
    
    return new_streamlines


# QuickBundle

# In[16]:

def computeQuickBundles(streamlines, threshold): #1.
    from dipy.segment.clustering import QuickBundles

    qb = QuickBundles(threshold)
    clusters = qb.cluster(streamlines)

#print("Nb. clusters:", len(clusters))
    #print("Cluster sizes:", map(len, clusters))
    #   print("Nb. small clusters:", sum(clusters < 10))
    #print("Streamlines indices of the first cluster:\n", clusters[0].indices)
    #print("Centroid of the last cluster:\n", clusters[-1].centroid)

    return clusters


# In[17]:

def renderCentroids(clusters):
    from dipy.viz import fvtk
    import numpy as np
    
    ren = fvtk.ren()
    ren.SetBackground(0, 0, 0)
    colormap = fvtk.create_colormap(np.arange(len(clusters)))

    colormap_full = np.ones((len(streamlines), 3))
    for cluster in clusters:
        colormap_full[cluster.indices] = np.random.rand(3)

    #fvtk.add(ren, fvtk.streamtube(streamlines, fvtk.colors.white, opacity=0.05))
    fvtk.add(ren, fvtk.line(clusters.centroids, linewidth=0.4, opacity=1))
    #fvtk.record(ren, n_frames=1, out_path='fornix_centroids.png', size=(600, 600))
    fvtk.show(ren)
    fvtk.clear(ren)


# In[18]:

def renderBundles(clusters):
    from dipy.viz import fvtk
    import numpy as np
    
    ren = fvtk.ren()
    ren.SetBackground(0, 0, 0)

    colormap = fvtk.create_colormap(np.arange(len(clusters)))

    colormap_full = np.ones((len(streamlines), 3))
    for cluster in clusters:
        colormap_full[cluster.indices] = np.random.rand(3)

    fvtk.add(ren, fvtk.line(streamlines, colormap_full))
    #fvtk.record(ren, n_frames=1, out_path='fornix_clusters.png', size=(600, 600))
    fvtk.show(ren)
    fvtk.clear(ren)


# Filter if small bundle

# In[21]:

def filterSmallBundles(streamlines, clusters, len_thr): #10
    j = 0
    smallbundles_list =[]
    for c,i in zip(clusters,range(len(clusters))):
        if len(c)<len_thr:
            j = j+1
            #print j
            #print clusters[i]
            for ii in range(len(clusters[i])):
                smallbundles_list.append(clusters[i].indices[ii])


#print 'Nb. streamlines from small bundles: '
#   print len(smallbundles_list)
    #print smallbundles_list

    new_streamlines = []
    for i in range(len(streamlines)):
        if i not in smallbundles_list:
            new_streamlines.append(streamlines[i])

#print 'Nb. streamlines before filtering'
#   print len(streamlines)
#   print 'Nb. streamlines after filtering'
#   print len(new_streamlines)
    return new_streamlines

