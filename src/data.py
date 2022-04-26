import os
import config

class Plant:
    def __init__(self, plant_id, location, drone_image_name, raw_annotation_file, x, y):
        self.id = plant_id
        self.x = x
        self.y = y
        self._drone_image_name = drone_image_name
        self._plant_name = f'{drone_image_name}_{plant_id}'
        self._tile_name = self._plant_name + '_{tile_id}'
        self._location_path = os.path.join(config.DIRS['data'], location)
        self._raw_annotation_file = raw_annotation_file

    @property
    def drone_image_data_path(self):
        return os.path.join(
            self._location_path,
            f'{self._drone_image_name}.{config.JSON_EXT}',
        )

    @property
    def data_path(self):
        return os.path.join(
            config.DIRS['processed']['info'],
            f'{self._plant_name}.{config.JSON_EXT}',
        )

    @property
    def drone_image_path(self):
        return os.path.join(
            self._location_path,
            f'{self._drone_image_name}.{config.DRONE_IMAGE_EXT}',
        )

    @property
    def raw_annotation_path(self):
        return os.path.join(
            config.DIRS['annotations'],
            self._raw_annotation_file,
        )

    @property
    def annotation_path(self):
        return os.path.join(
            config.DIRS['processed']['annotations'],
            f'{self._plant_name}.{config.OUTPUT_IMAGE_EXT}',
        )

    @property
    def mask_path(self):
        return os.path.join(
            config.DIRS['processed']['masks'],
            f'{self._plant_name}.{config.OUTPUT_IMAGE_EXT}',
        )

    @property
    def masked_image_path(self):
        return os.path.join(
            config.DIRS['processed']['masked_images'],
            f'{self._plant_name}.{config.OUTPUT_IMAGE_EXT}',
        )

    @property
    def hsv_mask_path(self):
        return os.path.join(
            config.DIRS['processed']['hsv_masks'],
            f'{self._plant_name}.{config.OUTPUT_IMAGE_EXT}',
        )

    def tile_path(self, tile_id):
        return os.path.join(
            config.DIRS['processed']['tiles'],
            f'{self._tile_name.format(tile_id=tile_id)}.{config.OUTPUT_IMAGE_EXT}',
        )

    def tile_mask_path(self, tile_id):
        return os.path.join(
            config.DIRS['processed']['tile_masks'],
            f'{self._tile_name.format(tile_id=tile_id)}.{config.OUTPUT_IMAGE_EXT}',
        )

    def tile_annotation_path(self, tile_id):
        return os.path.join(
            config.DIRS['processed']['tile_annotations'],
            f'{self._tile_name.format(tile_id=tile_id)}.{config.OUTPUT_IMAGE_EXT}',
        )

locations = {
    location: {
        drone_image_name
        for drone_image_name, ext in map(
            os.path.splitext,
            os.listdir(os.path.join(config.DIRS['data'], location))
        )
    }
    for location in os.listdir(config.DIRS['data'])
}

plants_by_location = {location: [] for location in locations}
for annotation_file in os.listdir(config.DIRS['annotations']):
    name, ext = os.path.splitext(annotation_file)
    name_components = name.split('_')
    drone_image_name = '_'.join(name_components[:-4])
    plant_id, x, y = (int(n) for n in name_components[-4:-1])
    for location, drone_image_names in locations.items():
        if drone_image_name in drone_image_names:
            break
    plants_by_location[location].append(Plant(
        plant_id,
        location,
        drone_image_name,
        annotation_file,
        x,
        y,
    ))

plants = [
    plant
    for location_plants in plants_by_location.values()
    for plant in location_plants
]
