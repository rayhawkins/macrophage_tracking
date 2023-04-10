from pyjamas.pjscore import PyJAMAS
import os
import pandas as pd

a = PyJAMAS()

folders = ["C:/Users/Ray/Documents/MASc/BME1462/Project/macrophage_tracking/macrophage_nuclei/macrophage_nuclei/h2o/20230326_embryo_1/",
           "C:/Users/Ray/Documents/MASc/BME1462/Project/macrophage_tracking/macrophage_nuclei/macrophage_nuclei/h2o/20230326_embryo_2/",
           "C:/Users/Ray/Documents/MASc/BME1462/Project/macrophage_tracking/macrophage_nuclei/macrophage_nuclei/h2o/20230326_embryo_3/",
           "C:/Users/Ray/Documents/MASc/BME1462/Project/macrophage_tracking/macrophage_nuclei/macrophage_nuclei/20230327-nlsgfp-50mMcbx/20230327_embryo_1/",
           "C:/Users/Ray/Documents/MASc/BME1462/Project/macrophage_tracking/macrophage_nuclei/macrophage_nuclei/20230327-nlsgfp-50mMcbx/20230327_embryo_2/",
           "C:/Users/Ray/Documents/MASc/BME1462/Project/macrophage_tracking/macrophage_nuclei/macrophage_nuclei/20230327-nlsgfp-50mMcbx/20230327_embryo_3/"]

for this_folder in folders:
    im_file = None
    ann_file = None
    for this_file in os.listdir(this_folder):
        if im_file is None and os.path.splitext(this_file)[1] == '.tif':
            im_file = this_file
        elif ann_file is None and this_file == 'nuclei_fiducials.pjs':
            ann_file = this_file
        else:
            continue
    if im_file is None or ann_file is None:
        print('what')
        continue

    a.io.cbLoadAnnotations([os.path.join(this_folder, ann_file)], os.path.join(this_folder, im_file))
    x_coords = []
    y_coords = []
    frames = []
    for this_frame in range(a.n_frames):
        if not a.fiducials[this_frame]:
            continue
        for this_fiducial in a.fiducials[this_frame]:
            y_coords.append(this_fiducial[0])
            x_coords.append(this_fiducial[1])
            frames.append(this_frame)
    dictio = {'x': y_coords, 'y': x_coords, 'frame': frames}
    df = pd.DataFrame(dictio)
    df.to_csv(os.path.join(this_folder, 'manual_tracking.csv'))

