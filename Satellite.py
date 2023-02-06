# File to define a satellite and connect it to its respective container unit.

import docker
import numpy as np
import requests
from scipy import linalg as la
import time
import uuid
import json
import binascii

DC = docker.from_env()


class Satellite:

    def __init__(self, initial_entropy, port, orbit, satellite_id, orbital_plane, vol_id):
        self.SatelliteID = satellite_id
        self.Azimuth_local = 200
        self.VolID = vol_id
        self.OrbitalPlane = orbital_plane
        self.OrbitID = orbit
        self.UpdateNo = 0
        self.StatelessReason = None     # Reason for being stateless satellite, If not stateless then set it to None.
        self.IndividualEntropy = initial_entropy
        self.IndividualAnomaly = 1.0
        self.IndividualEntropyDelta = 0.0
        self.OrbitStats = {}
        self.UpdateTimestamps = {}
        self.Key = '1abcd'
        self.SharedStages = orbital_plane.LocalStages  # Shared stages are after the second stage (Mandatory stage).
        self.NoSharedStages = len(list(orbital_plane.LocalStages))
        self.ContainerPort = port
        self.AccessEnvironment = ['ACCESS_KEY=' + str(self.Key), 'PLANET_PORT=3002', 'RTJ=False',
                                  'ENTROPY=' + str(self.IndividualEntropy), 'ID_SAT=' + str(satellite_id)]
        self.ContainerURL = 'http://0.0.0.0:' + str(self.ContainerPort)
        # Note: For IP address use self.Container.attrs['NetworkSettings']['IPAddress']
        self.ContainerID = self.SatContRun()
        self.Container = DC.containers.get(self.ContainerID)

        self.WeightMatrix = []

    # Container launch
    def SatContRun(self):
        vols = {'modelvol' + str(self.VolID): {'bind': '/mnt/VolumeLocal/model_zoo_local/', 'mode': 'rw'},
                'localvol' + str(self.VolID): {'bind': '/mnt/Data/', 'mode': 'rw'}}
        DC.volumes.create(name='modelvol' + str(self.VolID), driver='local')
        device_requests = [docker.types.DeviceRequest(device_ids=['0'], capabilities=[['gpu']])]
        cont_id = DC.containers.run("orbitallearning/client:latest", detach=True, ports={'80': self.ContainerPort},
                                    volumes=vols, environment=self.AccessEnvironment, device_requests=device_requests)
        time.sleep(40)
        return str(cont_id)[12:-1]

    def update_azimuth(self):
        keys = list(self.UpdateTimestamps.keys())
        self.Azimuth_local = self.UpdateTimestamps[keys[-1]]-self.UpdateTimestamps[keys[-2]]

    def time_the_update(self):
        self.UpdateTimestamps[self.UpdateNo] = time.time()
        if self.UpdateNo > 1:
            self.update_azimuth()
        else:
            self.Azimuth_local = 200
        self.UpdateNo += 1

    # Von Neumann Entropy
    def get_entropy(self, project_id):
        data = {'ProjectID': str(project_id)}
        s = requests.post(self.ContainerURL + '/get_entropy', json=data)
        # print(s.content)
        # print(s.json())
        # print(json.loads(s.json()))
        ik = dict(json.loads(s.json()))
        # print(ik)
        # print(ik['ed'])
        self.IndividualEntropyDelta = float(ik['ed'])
        self.IndividualEntropy = float(ik['e'])
        self.IndividualAnomaly = float(ik['anomaly'])
        return s

    def set_rtj(self, data):
        requests.post(self.ContainerURL + '/set_rtj', json=data)

    def get_rtj(self):
        out = requests.get(self.ContainerURL + '/get_rtj')
        return out.text

    def get_azimuth(self):
        out = requests.get(self.ContainerURL + '/get_azimuth')
        self.Azimuth_local = float(out.text)
        return out.text

    # Set OrbitID
    def set_orbit(self, orbit):
        self.OrbitID = orbit

    # Define or set orbit parameters
    def set_orbit_stats(self, stats):
        orbit_id = stats['OrbitID']
        self.set_orbit(orbit_id)
        self.OrbitStats = stats

    # Send the satellite entity (container) orbit details
    def update_orbit_info(self):
        requests.post(self.ContainerURL+'/update_env', files=self.OrbitStats)

    def inference(self, data):
        results = requests.post(self.ContainerURL+'/predict', json=data)
        return results

    # Add a default(untrained) stage to the model in the satellite.
    def add_stage(self, stage_config):
        results = requests.post(self.ContainerURL + '/add_stage', json=stage_config)
        if results["result"]:
            self.SharedStages[self.NoSharedStages+1] = results["StageID"]
            self.NoSharedStages += 1

    # Add a previously trained stage to the model.
    def insert_stage(self, stage_details):
        results = requests.post(self.ContainerURL + '/insert_stage', json=stage_details)
        self.SharedStages[self.NoSharedStages + 1] = results["StageID"]
        self.NoSharedStages += 1

    def transfer_stages(self, satellite, project_id):
        # if self.NoSharedStages < 1:
        #     self.request_inter_stage(satellite, project_id, 'ss', 's_stage')
        # else:
        #     self.request_inter_stage(satellite, project_id, 'ss', 's_stage')
        #     for i in self.SharedStages:
        #         self.request_inter_stage(satellite, project_id, i, 'c_stage')
        self.request_inter_stage(satellite, project_id, 'ss', 's_stage')

    def reenter_stages(self, project_id, stage_id, stage_level):
        target_url = self.ContainerURL + '/get_stage'
        data = {"ProjectID": project_id, "StageID": stage_id, "StageLevel": stage_level, "Version": '1.0'}
        downloaded_stage = requests.post(target_url, json=data)
        data_planet = {"ProjectID": project_id, "StageID": stage_id, "StageLevel": stage_level, "Version": '1.0',
                       "SatelliteID": self.SatelliteID}
        # print('hipip')
        # print(data_planet)
        target_url_planet = self.OrbitalPlane.Planet.ContainerURL + '/update_stage'
        status_planet = requests.post(target_url_planet, files={'file': downloaded_stage.content}, params=data_planet)

    # Inter satellite transfer of model stages through planet.
    def request_inter_stage(self, satellite, project_id, stage_id, stage_level):
        target_url = self.OrbitalPlane.Planet.ContainerURL + '/get_stage'
        data = {"ProjectID": project_id, "StageID": stage_id, "StageLevel": stage_level, "Version": '1.0',
                'SatelliteID': satellite.SatelliteID}
        # print('ilidini')
        # print('target_sat:' + str(satellite.SatelliteID))
        # print('self_sat:' + str(self.SatelliteID))
        downloaded_stage = requests.post(target_url, json=data)
        # print('alidini')
        uploading_stage = requests.post(self.ContainerURL+'/set_stage', files={'file': downloaded_stage.content},
                                        params=data)
        # TODO: Corrections might be required here.
        if uploading_stage.text == 'success':
            return 'Transfer complete'
        elif downloaded_stage.status_code != 200:
            return 'Stage retrieval failed'
        else:
            return 'Uploading stage failed'


    def prep_for_aux(self, project_id):
        data = {"ProjectID": project_id}
        rt = requests.post(self.ContainerURL + '/prepare_for_aux', json=data)

    # Request aux stages for ensemble learning other than the current one (currently only second stage)
    def request_aux_stages(self, satellite, project_id, stage_id, stage_level):
            target_url = self.OrbitalPlane.Planet.ContainerURL + '/get_stage'
            data = {"ProjectID": project_id, "StageID": stage_id, "StageLevel": stage_level, "Version": '1.0',
                    'SatelliteID': satellite.SatelliteID}
            # print('ilidini')
            # print('target_sat:' + str(satellite.SatelliteID))
            # print('self_sat:' + str(self.SatelliteID))
            downloaded_stage = requests.post(target_url, json=data)
            # print(downloaded_stage.status_code)
            # print('alidini')
            uploading_stage = requests.post(self.ContainerURL + '/set_aux_stage', files={'file': downloaded_stage.content},
                                            params=data)
            # print(uploading_stage.status_code)
            # TODO: Corrections might be required here.
            if uploading_stage.text == 'success':
                return 'Transfer complete'
            elif downloaded_stage.status_code != 200:
                return 'Stage retrieval failed'
            else:
                return 'Uploading stage failed'

    # Model stages from planet during launch.
    def request_launch(self, project_id, stage_id=None):
        target_url = self.OrbitalPlane.Planet.ContainerURL + '/launch_model'
        if stage_id is None:
            stage_id = 'ss'
        data_p_p = {"ProjectID": project_id, "Version": '1.0', "StageLevel": 'p_stage'}
        downloaded_stage_p = requests.post(target_url, json=data_p_p)
        stage_p = downloaded_stage_p.content
        print(downloaded_stage_p.status_code)

        data_p_d = {"ProjectID": project_id, "Version": '1.0', "StageLevel": 'd_stage'}
        downloaded_stage_d = requests.post(target_url, json=data_p_d)
        stage_d = downloaded_stage_d.content

        data_p_ss = {"ProjectID": project_id, "Version": '1.0', "StageLevel": 's_stage'}
        downloaded_stage_ss = requests.post(target_url, json=data_p_ss)
        stage_ss = downloaded_stage_ss.content

        data_p_agent = {"ProjectID": project_id, "Version": '1.0', "StageLevel": 'agent'}
        downloaded_stage_agent = requests.post(target_url, json=data_p_agent)
        stage_agent = downloaded_stage_agent.content

        data_s_agent = {"ProjectID": project_id, "StageID": stage_id, "StageLevel": 'agent', "Version": '1.0'}
        uploading_stage_agent = requests.post(self.ContainerURL + '/set_stage', files={'file': stage_agent},
                                          params=data_s_agent)

        data_s_p = {"ProjectID": project_id, "StageID": stage_id, "StageLevel": 'p_stage', "Version": '1.0'}
        print('sending p_stage ' + str(time.time()))
        uploading_stage_p = requests.post(self.ContainerURL+'/set_stage', files={'file': stage_p},
                                        params=data_s_p)
        print(uploading_stage_p.status_code)

        data_s_d = {"ProjectID": project_id, "StageID": stage_id, "StageLevel": 'd_stage', "Version": '1.0'}
        uploading_stage_d = requests.post(self.ContainerURL+'/set_stage', files={'file': stage_d},
                                        params=data_s_d)

        data_s_ss = {"ProjectID": project_id, "StageID": stage_id, "StageLevel": 's_stage', "Version": '1.0'}
        uploading_stage_ss = requests.post(self.ContainerURL+'/set_stage', files={'file': stage_ss},
                                        params=data_s_ss)

        # TODO: Corrections might be required here.
        if uploading_stage_p.text == 'success' and uploading_stage_d.text == 'success' and \
                uploading_stage_ss.text == 'success' and uploading_stage_agent.text == 'success':
            return 'Transfer complete'
        elif downloaded_stage_p.status_code != 200:
            return 'Stage retrieval failed'
        else:
            return 'Uploading stage failed'

    def request_extras(self, project_id):
        target_url = self.OrbitalPlane.Planet.ContainerURL + '/launch_extras'
        data_sd = {"ProjectID": str(project_id), "Field": 'std'}
        downloaded_sd = requests.post(target_url, json=data_sd)
        uploading_sd = requests.post(self.ContainerURL+'/set_extras', files={'file': downloaded_sd.content},
                                        params=data_sd)
        print('testxc')
        print(downloaded_sd.status_code)
        print(uploading_sd.status_code)
        data_mea = {"ProjectID": project_id, "Field": 'mea'}
        downloaded_mea = requests.post(target_url, json=data_mea)
        uploading_mea = requests.post(self.ContainerURL+'/set_extras', files={'file': downloaded_mea.content},
                                        params=data_mea)
        print('testxv')
        print(downloaded_mea.status_code)
        print(uploading_mea.status_code)

    # Send model stage from current satellite to a planet.
    def controlled_reentry(self, planet, project_id):
        target_url = planet.ContainerURL
        data = {"ProjectID": project_id}
        downloaded_stage = requests.get(self.ContainerURL+'/update_stage', json=data)
        uploading_stage = requests.post(target_url, files=downloaded_stage)
        if uploading_stage.text == 'success':
            status = self.OrbitalPlane.satellite_reentry(self.SatelliteID)
            if status:
                print('re-entry complete!')  # Put this into logger.
                del self
        elif downloaded_stage.status_code != 200:
            return 'Stage retrieval failed'
        else:
            return 'Re-entry stage failed'



# y = Satellite(0.0, 8000)
# dataw = { "variance_of_wavelet": 0, "skewness_of_wavelet": 0.3,
#     "curtosis_of_wavelet": 0.1,
#     "entropy_of_wavelet": 0,
#     "retrain_model_": 0}
#
# time.sleep(100)
# url = 'http://0.0.0.0:8000/predict'
# p = requests.post(url, json=dataw)
# print(p.text)
