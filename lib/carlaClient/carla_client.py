import carla
import numpy as np
import cv2
import os
import time


class CarlaClient:
    def __init__(self, ip="host.docker.internal", port=2000, camera_position=None, camera_rotation=None, resolution=(800, 600),
                 output_folder="output_images", save_image=False):
        """
        Initialize the CARLA client.

        Args:
        ip (str): IP address of the CARLA simulator, default "host.docker.internal".
        port (int): Port of the CARLA simulator, default 2000.
        camera_position (tuple): Camera position (x, y, z), in meters.
        camera_rotation (tuple): Camera rotation (pitch, yaw, roll), in degrees.
        resolution (tuple): Image resolution (width, height).
        output_folder (str): Folder path where images are saved.
        save_image (bool): Whether to save images.
        """
        self.ip = ip
        self.port = port
        self.camera_position = camera_position or (1.5, 0, 2.5)
        self.camera_rotation = camera_rotation or (0, 0, 0)
        self.resolution = resolution
        self.output_folder = output_folder
        self.save_image = save_image

        self.client = None
        self.world = None
        self.camera = None
        self.image_data = None
        self.image_name = None

        if save_image:
            os.makedirs(output_folder, exist_ok=True)


        self.connect_to_carla()

    def connect_to_carla(self):
        """Connect to the CARLA simulator."""
        self.client = carla.Client(self.ip, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        print(f"Connected to CARLA at {self.ip}:{self.port}")

    def setup_camera(self):
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')

        camera_bp.set_attribute('image_size_x', str(self.resolution[0]))
        camera_bp.set_attribute('image_size_y', str(self.resolution[1]))
        camera_bp.set_attribute('fov', '90') 

        transform = carla.Transform(
            carla.Location(x=self.camera_position[0], y=self.camera_position[1], z=self.camera_position[2]),
            carla.Rotation(pitch=self.camera_rotation[0], yaw=self.camera_rotation[1], roll=self.camera_rotation[2])
        )

        
        self.camera = self.world.spawn_actor(camera_bp, transform)
        self.camera.listen(self._camera_callback)
        print("Camera sensor set up successfully.")

    def _camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.image_data = array.reshape((image.height, image.width, 4))[:, :, :3]
        self.image_name = f"frame_{image.frame}"  

    def get_image(self):
        """
        Get the latest image data. If save_image is set, save the image here.

        Returns:
        numpy.ndarray: Image data, in RGB format. Returns None if the image has not been updated.
        """
        if self.image_data is not None:
            if self.save_image:
                file_path = os.path.join(self.output_folder, f"{self.image_name}.png")
                cv2.imwrite(file_path, cv2.cvtColor(self.image_data, cv2.COLOR_RGB2BGR))
                print(f"Image saved: {file_path}")
            return self.image_data, self.image_name
        else:
            print("Image data not available yet.")
            return None

    def stop(self):
        """Stop camera monitoring and destroy the sensor."""
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
            print("Camera sensor destroyed.")

    def __del__(self):
        """Destructor, ensures resource release."""
        self.stop()

