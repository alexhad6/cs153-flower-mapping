# CS153 Flower Mapping

Flower mapping final project for CS 153 at Harvey Mudd College.

## Downloads

The following files can be downloaded. We have included the annotations we used, along with their corresponding masked images. Note that the code in this repository expects all of the data to be present in order to generate the tiles used for training (see __Setup__ below). In theory, the code could be adjusted to generate everything using these masked images, but in order to be more general to other datasets and annotations, we left it expecting all the data.

- [Annotations](https://github.com/alexhad6/cs153-flower-mapping/releases/download/v1.0.0/annotations.tgz)
- [Masked Images](https://github.com/alexhad6/cs153-flower-mapping/releases/download/v1.0.0/masked_images.tgz)

The weights for our trained models (see the training and running model sections below) can be downloaded here.

- [Model Weights](https://github.com/alexhad6/cs153-flower-mapping/releases/download/v1.0.0/model_weights.tgz)

## Setup

Follow the steps below to set up the repository.

1. Install [Pipenv](https://pipenv.pypa.io/en/latest/).

2. Run `pipenv install` from the root of this repository to install Python dependencies.

3. Run `./download-weights.sh` to download the COCO-Stuff DeepLab weights from [kazuto1011/deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch).

4. Place drone images and JSON data in subdirectories in the `data` directory. For access to the data, contact the [HMC Bee Lab](https://hmcbee.blogspot.com).

    <details><summary>Data directory file structure</summary>

    Our `data` directory had the following file structure:

    ```
    070921_North
        070921_North100_0007_0044.JPG
        070921_North100_0007_0044.json
        ...
    071121_CentralEastern
        071121_CentralEastern100_0009_0002.JPG
        071121_CentralEastern100_0009_0002.json
        ...
    071121_Western
        071121_Western100_0011_0015.JPG
        071121_Western100_0011_0015.json
        ...
    2017_6617East1
        2017_6617East1DJI_0029.JPG
        2017_6617East1DJI_0029.json
        ...
    2017_6617East2
        2017_6617East2DJI_0345.JPG
        2017_6617East2DJI_0345.json
        ...
    2017_6917West
        2017_6917WestDJI_0007.JPG
        2017_6917WestDJI_0007.json
        ...
    ```

    </details>

    <details><summary>JSON file format</summary>

    The JSON files associated with each plant are expected to at least have the following data:

    ```json
    {
        "classes": [
            "ERFA"
        ],
        "labels": [
            {
                "class": "ERFA",
                "segment": [
                    1741.56, 1014.43,
                    1710.62, 1001.17,
                    1681.89, 981.28,
                    ...
                ]
            },
            ...
        ]
    }
    ```

    The classes are each type of plant present, and the segment contains the points of a boundary polygon for that plant in the format `[x1, y1, x2, y2, ...]`.

    </details>

5. Run `pipenv run python src/generate_plants.py` to generate plant images in the directory `plant_images`.

6. Place ground truth annotation masks for each plant in the `annotations` directory. Note that the dimensions of each annotations should be equal to the bounding box of the polygon created using the OpenCV `fillPoly` function for the plant segmentation boundary from the JSON data in step 4.

7. Run `pipenv run python src/processing.py` to clean up the annotations and set everything up for the training and running the models.

## Running the HSV Model

These steps assume that everything was set up from the previous steps.

1. Run `pipevn run python src/hsv_histogram.py` to generate HSV histograms of flower and non-flower areas across all annotated plants.

2. Run `pipenv run python src/hsv.py` to generate HSV histograms of flower and non-flower areas for all annotated plants in the `processed/hsv_masks` directory.

## Training the DeepLab Models on XSEDE

1. Run `pipenv run python src/xsede-setup.py` to copy the relevant files to the `xsede_deeplabv2` and `xsede_deeplabv3` directories and copy these directories to XSEDE. Also, change the paths within the `model.job` files to reflect the path on XSEDE.

2. On XSEDE, `cd` into each folder and run `sbatch model.job` to train the model. Run `tail -f output.txt` to see what is going on as the model trains.

3. When complete, copy the file `deeplabv2.pth` or `weights.pt` (for the DeepLab v3 model) back to your computer and place in the `model` directory.

## Running the DeepLab Models

1. Run `pipenv run python src/run_deeplabv2.py` to generate annotations to the `processed/deeplabv2_masks` folder, and relevant performance statistics.

2. Run `pipenv run python src/run_deeplabv3.py` to generate annotations to the `processed/deeplabv3_masks` folder, and relevant performance statistics.
