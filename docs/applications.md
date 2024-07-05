# Examples
## Projecting image-based segmentation predictions to geospatial coordinates
For tasks like vegetation cover segmentation, it is common to use a semantic segmentation network that produces a class label for each pixel in an image. This workflow shows how to take these per-image predictions and project them to a 3D mesh representation. From there, the most commonly-predicted class is identified for each face of the mesh. Then, this information can be exported in a geospatial format showing the boundaries of each class. This information can also be post-processed to determine the most common class for a pre-determined region, such as a single tree or a management region. This workflow is fairly well-developed and works for a variety of applications.

* [geograypher/examples/aggregate_predictions.ipynb](https://github.com/open-forest-observatory/geograypher/blob/main/examples/aggregate_predictions.ipynb)

## Projecting object detections to geospatial coordinates
Other tasks consist of identifying individual objects such as trees or birds. Given object detections in individual images, this workflow can determine the corresponding geospatial boundaries. The quality of this approach is heavily dependent on the quality of the mesh model of the scene. In cases where the scale of the reconstruction errors is significant compared to the size of individual objects, the localization may be poor. An alternative workflow is to triangulate multiple detections from different images to localize the objects without using a mesh. This works in some cases, but often under-predicts the true number of unique objects. Both workflows are highly experimental.

* [geograypher/examples/project_detections.ipynb](https://github.com/open-forest-observatory/geograypher/blob/main/examples/project_detections.ipynb)
## Generating per-image labels from geospatial ground truth
The prior examples have assumed that per-image predictions are available. In many practical applications, the user must train their own machine learning model to generate these predictions. Since the predictions are generated on individual raw images, it is important that the labeled data to train the model also consists of individual images, and not another representation such as an orthomosaic. The user can hand-annotate individual images, but this process is laborious and ecological classes (e.g. plant species) cannot always be reliably determined without in-field observations. This workflow takes geospatial ground truth data collected in-field and projects it to the viewpoint of each individual image. These per-pixel labels corresponding to each real image can be used to train a machine learning model.

* [geograypher/examples/render_labels.ipynb](https://github.com/open-forest-observatory/geograypher/blob/main/examples/render_labels.ipynb)

## Workflow using simulated data
All the other examples use real data. To support conceptual understanding and rapid debugging, we developed an end-to-end workflow using only simulated data. The scene consists of various geometric objects arranged on a flat plane. Users can configure the objects in the scene and the locations of the virtual cameras observing them.

* [geograypher/examples/concept_figure.ipynb](https://github.com/open-forest-observatory/geograypher/blob/main/examples/concept_figure.ipynb)

# Projects using geograypher
## Cross-site tree species classification for Western Conifers
The goal of this project is to identify the species of western conifers in regions which have been burned by severe fire in the past decade. Data was collected at four sites and consisted of both drone surveys and manual field surveys. This work showed geograypher's multiview workflow enabled 75% prediction accuracy on a leave-on-site-out prediction task. Details can be found in this ArXiv [paper](https://arxiv.org/abs/2405.09544).