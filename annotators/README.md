# Annotation instructions

The full cycle of parking lot annotating consists of two stages:

- marking the location of parking spaces using widget A.
- marking the status of parking spaces using widget B.

## Widget A usage

In the same directory with the widget, a potential user should create parking_photos folder with one image per each point of view (the name of images correspond to the name of the folders with the point of view, e.g. "cam_1.jpg" for the folder "cam_1"). After launching .ipynb, the user marks parking spaces on each image with the help of the "Brush" tool (the full functionality of the widget is illustrated [in this video](https://drive.google.com/file/d/1YBV01vzFHIfsJ6lkjICdrCTEHLMahpNo/view?usp=sharing)).

## Widget B usage

In the same directory with the widget, the "annotations.json" file (obtained after saving the results of widget A) and folders with images grouped relative to angles are placed. After launching .ipynb, the user marks occupied (red) and free (blue) parking lots on each image (the functionality of the widget is illustrated [in this video](https://drive.google.com/file/d/12plriorxmw5o1Y9IhHOKey1IUUKSY1sx/view?usp= sharing)). When the annotation process is will be done, a user can save results in automatically created patch_markup and int_markup directories as the separate .json files for each image.
