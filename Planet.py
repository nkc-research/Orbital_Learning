# File to define a planet(server) and connect it to its respective container unit.
import uuid
import docker
import requests
import time

DC = docker.from_env()


class Planet:

    def __init__(self, port, vol_id):
        self.PlanetID = uuid.uuid4()
        self.ContainerPort = port
        self.VolID = vol_id
        self.Key = uuid.uuid4()
        self.AccessEnvironment = ['ACCESS_KEY='+str(self.Key)]
        self.ContainerURL = 'http://0.0.0.0:' + str(self.ContainerPort)
        # Note: For IP address use self.Container.attrs['NetworkSettings']['IPAddress']
        self.ContainerID = self.PlaContRun()
        self.Container = DC.containers.get(self.ContainerID)

    # Container launch
    def PlaContRun(self):
        vols = {'modelvol' + str(self.VolID): {'bind': '/mnt/VolumeLocal/model_zoo_local/', 'mode': 'rw'},
                'localvol' + str(self.VolID): {'bind': '/mnt/Data/', 'mode': 'rw'}}
        DC.volumes.create(name='modelvol' + str(self.VolID), driver='local')
        device_requests = [docker.types.DeviceRequest(device_ids=['0'], capabilities=[['gpu']])]
        cont_id = DC.containers.run("orbitallearning/central_node:latest", detach=True,
                                    ports={'80': self.ContainerPort}, volumes=vols, environment=self.AccessEnvironment,
                                    device_requests=device_requests)
        time.sleep(40)
        return str(cont_id)[12:-1]

    def inference(self, data):
        results = requests.post(self.ContainerURL, json=data)
        return results
