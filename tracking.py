import matplotlib.pyplot as plt
import numpy
from skimage.measure import label, regionprops


# Input is masks (3D binary images in x,y, and time) and output is nuclei tracks (n by t by 2 numpy arrays where n is 
# the number of macrophages, t is the number of timepoints, and 2 corresponds to (x,y) coordinates of that nucleus
# at that timepoint
def track(segmented_image):
    centroids = [[] for _ in range(segmented_image.shape[0])]
    for t, this_image in enumerate(segmented_image):
        print(f"Slice index: {t}")
        labelled_image = label(this_image)
        regions = regionprops(labelled_image)
        for this_region in regions:
            centroids[t].append(this_region.centroid)
        centroids[t] = numpy.array(centroids[t])
        print(len(centroids[t]))
        if t == 0:  # Initialize storage of tracked centroids, only take as many nuclei as were detected on first slice
            tracked_centroids = numpy.full([len(centroids[t]), segmented_image.shape[0], 2], None)
            tracked_centroids[:, t, :] = centroids[t]
            num_centroids = len(centroids[t])
        else:
            # Matrix of distances between all current frame centroids and previous frame centroids
            centroids[t] = centroids[t][:num_centroids]  # Remove any additional falsely detected nuclei
            print(tracked_centroids[:, t-1, :])
            if len(centroids[t]) < num_centroids:
                centroids[t] = numpy.concatenate(centroids[t], numpy.full((num_centroids - len(centroids[t]), 2), None))
            dists = numpy.ones([len(centroids[t]), len(centroids[t - 1])], dtype=numpy.float)
            for c, this_centroid in enumerate(centroids[t]):
                for p, prev_centroid in enumerate(tracked_centroids[:, t - 1, :]):
                    if this_centroid[0] is None or this_centroid[1] is None:
                        # Set distance to maximum possible distance within image bounds
                        dists[c, p] = max(segmented_image.shape[1], segmented_image.shape[2])
                    else:
                        dists[c, p] = numpy.abs(this_centroid[0] - prev_centroid[0]) + \
                                      numpy.abs(this_centroid[1] + prev_centroid[1])

            for c in range(len(centroids[t])):
                min_dist = numpy.min(dists)
                dist_loc = numpy.where(dists == min_dist)
                dist_loc = [dist_loc[0][0], dist_loc[1][0]]
                if centroids[t][dist_loc[0]][0] is None or centroids[t][dist_loc[0]][1] is None:
                    # Object disappeared, use last centroid as a fill-in
                    print("filling hole")
                    print(dist_loc)
                    tracked_centroids[dist_loc[1], t, :] = tracked_centroids[dist_loc[1], t - 1, :]

                else:
                    tracked_centroids[dist_loc[1], t, :] = centroids[t][dist_loc[0]]

                # Remove nuclei that have been used from the distance matrix (to prevent them being used twice)
                dists = numpy.delete(dists, dist_loc[0], 0)
                dists = numpy.delete(dists, dist_loc[1], 1)

    return tracked_centroids


def track(segmented_image):
    # Centroids has the centroids for each object in each frame in no particular order
    centroids = [[] for _ in range(segmented_image.shape[0])]
    # Global ids stores the location of the object i in centroids on frame t
    global_ids = []
    for t, this_image in enumerate(segmented_image):
        labelled_image = label(this_image)
        regions = regionprops(labelled_image)
        for this_region in regions:
            centroids[t].append(this_region.centroid)

        if t == 0:  # Initialize storage of tracked centroid ids
            for c in range(len(centroids[t])):
                global_ids.append([c])
        else:
            dists = numpy.empty([len(centroids[t]), len(centroids[t - 1])])
            for c, this_centroid in enumerate(centroids[t]):
                for p, prev_centroid in enumerate(centroids[t - 1]):
                    dists[c, p] = numpy.abs(this_centroid[0] - prev_centroid[0]) + \
                                  numpy.abs(this_centroid[1] + prev_centroid[1])

            row_ids = numpy.array(range(len(centroids[t])))
            col_ids = numpy.array(range(len(centroids[t - 1])))
            for c in range(min(len(centroids[t]), len(centroids[t - 1]))):
                # Dist loc gives index of current centroid on current frame, index of prev centroid on prev frame
                dist_loc = numpy.where(dists == numpy.min(dists))
                dist_loc = [dist_loc[0][0], dist_loc[1][0]]

                prev_loc = col_ids[dist_loc[1]]
                curr_loc = row_ids[dist_loc[0]]
                for o, this_object in enumerate(global_ids):
                    if this_object[t - 1] == prev_loc:  # Object that matches the id of the previous centroid
                        global_ids[o].append(curr_loc)  # Append new id for current frame
                        break

                dists = numpy.delete(dists, dist_loc[0], 0)
                row_ids = numpy.delete(row_ids, dist_loc[0])
                dists = numpy.delete(dists, dist_loc[1], 1)
                col_ids = numpy.delete(col_ids, dist_loc[1])

            # Add new tracks for objects that appeared on this frame
            if len(centroids[t]) > len(centroids[t - 1]):
                for remaining_object in row_ids:
                    new_ids = [None for _ in range(t - 1)] + [remaining_object]
                    global_ids.append(new_ids)

            # Set all objects that don't have an id for this timepoint to None
            for o, this_object in enumerate(global_ids):
                if len(this_object) < t + 1:
                    global_ids[o].append(None)

    # Prune ids so that we get tracks of at least 30 good frames at a time (no spurious objects)
    good_ids = []
    for this_object in global_ids:
        if sum(1 for _ in filter(None.__ne__, this_object)) > 30:
            good_ids.append(this_object)

    global_ids = good_ids

    # Reconstruct this back into a list of tracked centroids
    tracked_centroids = []
    for this_object in global_ids:
        these_centroids = []
        for t, this_index in enumerate(this_object):
            if this_index is not None:
                these_centroids.append(centroids[t][this_index])
            else:
                these_centroids.append(None)
        tracked_centroids.append(these_centroids)
    return numpy.array(tracked_centroids)

