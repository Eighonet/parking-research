# Annotation instructions

The entire cycle of parking lot annotating consists of two stages:

- annotation of parking spaces (widget A).
- labelling of parking lot status (widget B).

## Widget A usage

A potential user should create `parking_photos` folder in the widget directory and add one image per each point of view (names of the images have to correspond with the name of the folders, e.g. `cam_1.jpg` for the folder `cam_1`). After launching the widget in Jupyter Notebook, a user can annotate parking lots at each image with the help of the "Brush" tool (the full functionality of the widget is illustrated [in this video](https://drive.google.com/file/d/1YBV01vzFHIfsJ6lkjICdrCTEHLMahpNo/view?usp=sharing)).

## Widget B usage

Generated `annotations.json` file should be placed in the directory with the widget along with image folders. After launching the widget in Jupyter Notebook, a user labels occupied (red) and free (blue) parking lots on each image (the functionality of the widget is illustrated [in this video](https://drive.google.com/file/d/12plriorxmw5o1Y9IhHOKey1IUUKSY1sx/view?usp=sharing)). When the annotation process is done, a user can save results in automatically created `patch_markup` and `int_markup` directories as separate .json files.
