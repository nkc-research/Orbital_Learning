# OrbitalPlane defines the presence of satellites (clients) on a single plane with similar entropy and regulates
# acceptance of a new satellite or removal of a satellite from the orbit.
# Note: Terminology- Orbit and Orbital plane are the same. Orbital plane has been used to avoid any confusions.
# Assessor Data similarity to benchmark dataset in central node is referred to as Anomaly value
# Assessor Weight similarity to reference NN model in Central node is referred to as Entropy value
# Melioration rate is referred to as Azimuth value

import numpy as np
from time import time, sleep
import threading
import asyncio
import pandas as pd
import uuid
import requests
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import multiprocessing
from scipy import stats

def circle(list_name):
    while True:
        for connection in list_name:
            yield connection


def gmm_bic_score(estimator, X):
    X = np.array(X)
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

def sillhouette_sc(estimator, X):
    Y = estimator.fit_predict(X)
    return silhouette_score(X, Y)

class OrbitalPlane:

    def __init__(self, planet, entropy, anomaly, azimuth, orbital_system, local_stages, id=None):
        self.Azimuth = azimuth  # frequency at which the update takes place.
        self.AzimuthThreshold = 150
        if id is not None:
            self.OrbitalID = id
        else:
            self.OrbitalID = uuid.uuid4()
        self.LocalSatellites = {}
        self.LocalEntropy = entropy
        self.LocalAnomaly = anomaly
        self.AllEntropies = []
        self.AllAnomalies = []
        self.IncomingSatellites = []
        self.Planet = planet
        self.FreshOrbit = True
        self.LocalStages = local_stages
        self.StopOrbit = False
        if self.LocalStages:
            self.ActiveStage = self.LocalStages[-1]
        else:
            self.ActiveStage = 0
        self.TimeInterval = 3 # 86400  # seconds
        self.EntropyThreshold = 0.01
        self.AnomalyThreshold = 5.0
        self.Restrictions = True  # Placeholder for satellites to provide framework of accepting a new satellite
        self.OrbitalSystem = orbital_system
        self.ProcessThread = []
        self.SatDelEntropies = {}
        # loop = asyncio.get_event_loop()
        # task = loop.create_task(self.maintain_orbit())

        # self.maintain_orbit()
    def start_orbit(self):
        self.ProcessThread = threading.Thread(name='PMO_'+str(self.OrbitalID), target=self.maintain_orbit)
        self.ProcessThread.start()

    def stop_orbit(self):
        self.StopOrbit = True

    def add_satellite(self, satellite):
        self.LocalSatellites[satellite.SatelliteID] = satellite
        self.FreshOrbit = False
        if self.OrbitalID == 'def' and self.StopOrbit:
            self.StopOrbit = False
            self.start_orbit()

    def remove_satellite(self, satellite_id):
        del self.LocalSatellites[satellite_id]

    def close_orbit(self):
        del self

    def satellite_reentry(self, satellite_id):
        status = self.OrbitalSystem.signal_removal(satellite_id)
        if status:
            del self.LocalSatellites[satellite_id]
        else:
            print('Re-entry failed at orbital control system.')
    
    def cal_delta(self, s):
        dr = []
        ref = self.SatDelEntropies[s]
        for w, t in enumerate(self.LocalSatellites.keys()):
            if s == t:
                continue
            dr.append(abs(ref - self.SatDelEntropies[t]))
        if len(list(self.SatDelEntropies.keys()))>1:
            return float(np.max(dr))
        else:
            return 0.0
                
                
    def evaluate_entropy(self):
        ent_dict = {}
        ent_array = np.zeros(len(self.LocalSatellites.keys()))
        ana_array = np.zeros(len(self.LocalSatellites.keys()))
        ipl = list(self.LocalSatellites.keys())
        for i, s in enumerate(ipl):
            gof = self.LocalSatellites[s].get_entropy('1.0')
            go = self.LocalSatellites[s].IndividualEntropy
            ao = self.LocalSatellites[s].IndividualAnomaly
            self.SatDelEntropies[s] = self.LocalSatellites[s].IndividualEntropy
            if gof.status_code == 200:
                ent_dict[s] = str(go)
                ent_array[i] = str(go)
                ana_array[i] = str(ao)
            else:
                ent_dict[s] = 5000
                ent_array[i] = 5000     # TODO: test required.
                ana_array[i] = 5000
        self.LocalEntropy = np.mean(ent_array)
        self.LocalAnomaly = np.mean(ana_array)
        self.AllEntropies = ent_array
        self.AllAnomalies = ana_array
        return np.mean(ent_array), self.max_deviation(ent_array), ent_dict

    def enter_orbit(self, satellite):

        if self.Restrictions:
            if self.LocalSatellites:
            # lock = threading.Lock()
            # lock.acquire()
                self.IncomingSatellites.append(satellite)
            else:
                self.LocalSatellites[int(satellite.SatelliteID)] = satellite
            # lock.release()
            return True

    def launch_rocket(self):
        while True:
            try:
                readiness = requests.get(self.Planet.ContainerURL+'/ready_state')
                print(readiness.content.decode('ascii'))
                break
            except:
                continue
        if readiness.content.decode('ascii') == "\"True\"":
            for kl, ks in enumerate(list(self.LocalSatellites.keys())):
                self.LocalSatellites[ks].request_launch(1.0)
                self.LocalSatellites[ks].request_extras(1.0)
        else:
            print("error in model launch")

    # This will actively maintain orbit at a particular mean entropy and remove satellite with higher or lower
    # entropies. Before removal of any satellite, a request shall be sent to the orbital system manager, which
    # will allocate a new orbital plane or assign an existing orbital plane to the removed satellite.
    # No active satellite is to be left out of orbital system.
    def maintain_orbit(self):
        counter_loop = 1
        check_rtj = np.zeros(len(list(self.LocalSatellites.keys())))
        while True:
            # print(self.OrbitalSystem.StartFlag)
            if self.OrbitalSystem.StartFlag and not self.StopOrbit:
                # print('test 1')
                if len(list(self.LocalSatellites.keys())) < 1 and not self.FreshOrbit and self.OrbitalID != 'def':
                    lock = threading.Lock()
                    lock.acquire()
                    print('-----------------------')
                    print(str(self.OrbitalID))
                    print('removing empty orbit')
                    print(len(list(self.LocalSatellites.keys())))
                    print('-----------------------')
                    lock.release()
                    asyncio.run(self.OrbitalSystem.signal_empty_orbit(self.OrbitalID))
                    break
                elif len(list(self.LocalSatellites.keys())) < 1:
                    continue
                elif self.OrbitalID == 'def' and len(list(self.LocalSatellites.keys())) < 1:
                    break
                # sleep(self.TimeInterval - time() % self.TimeInterval)
                check_rtj = np.zeros(len(list(self.LocalSatellites.keys())))
                for kl, ks in enumerate(list(self.LocalSatellites.keys())):
                    if str(self.LocalSatellites[ks].get_rtj())[1:-1] == str("BBlink"):
                        # if check_rtj[kl] != 1:
                        #     print(ks)
                        #     print(len(list(self.LocalSatellites.keys())))
                        #     print('getting_rtj')
                        #     self.LocalSatellites[ks].time_the_update()
                        check_rtj[kl] = 1
                    # print(str(self.LocalSatellites[ks].get_rtj()[1:-1]))
                    # print(str(kl)+":"+str(check_rtj[kl]))


                if not np.all(check_rtj):
                    continue

                if counter_loop < 5 and self.OrbitalID == 'def':
                    for sat in list(self.LocalSatellites.keys()):
                        rtj_flag = {'RTJ': 'False'}

                        self.LocalSatellites[sat].set_rtj(rtj_flag)
                    counter_loop += 1
                    continue
######
                lock = threading.Lock()
                lock.acquire()
                print('--cheking enter orbit--')
                print(self.IncomingSatellites)
                print('-----------------------')
                lock.release()
                if self.IncomingSatellites:
                    for satellite in self.IncomingSatellites:
                        self.LocalSatellites[int(satellite.SatelliteID)] = satellite
                    self.IncomingSatellites = []
                ent_measure, ent_dev, ent_dict = self.evaluate_entropy()
                if counter_loop == 5 and self.OrbitalID == 'def':
                    #ent_measure, ent_dev, ent_dict = self.evaluate_entropy()
                    joint_array = list(zip(100*self.AllEntropies, self.AllAnomalies))
                    # clustering = GaussianMixture(n_components=2, random_state=0).fit_predict(joint_array)


                    # param_grid = {
                    #     "n_clusters": range(1, 4)
                    # }
                    # grid_search = GridSearchCV(
                    #     KMeans(), param_grid=param_grid, scoring=sillhouette_sc
                    # )
                    # grid_search.fit(joint_array)
                    # df = pd.DataFrame(grid_search.cv_results_)[
                    #     ["param_n_clusters", "mean_test_score"]
                    # ]
                    # df["mean_test_score"] = -df["mean_test_score"]
                    scores = []
                    for ty in range(2, 5):
                        ret = KMeans(n_clusters=ty).fit_predict(joint_array)
                        scores.append(silhouette_score(joint_array, ret))
                    dr_components = np.argmax(np.array(scores))
                    print(dr_components)
                    print(scores)
                    #clustering = DBSCAN(eps=0.5, min_samples=1).fit_predict(joint_array)
                    # clustering = GaussianMixture(n_components=int(dr_components)+2, random_state=0).fit_predict(joint_array)
                    clustering = KMeans(n_clusters=int(dr_components)+2).fit_predict(joint_array)
                    lock = threading.Lock()
                    lock.acquire()
                    print('--Clustering labels--')
                    print(joint_array)
                    print(clustering)
                    print('-----------------------')
                    lock.release()
                    unique_labels = list(set(clustering))
                    cluster_anomalies = np.zeros(len(unique_labels))
                    self.AllAnomalies = np.array(self.AllAnomalies)
                    for z, u in enumerate(unique_labels):
                        cluster_anomalies[z] = np.mean(self.AllAnomalies[clustering == u])
                    h_index = np.argmin(cluster_anomalies)
                    h_g = unique_labels[h_index]
                    sg = np.array(list(self.LocalSatellites.keys()))
                    del_ref_list = sg[clustering != h_g]
                    for nq in list(self.LocalSatellites.keys()):
                        if nq in del_ref_list:
                            self.LocalSatellites[nq].StatelessReason = 'entropy'
                            # asyncio.run(self.OrbitalSystem.signal_displacement(nq))
                            # print(self.OrbitalSystem.StatelessSatellites)
                            del self.LocalSatellites[nq]
                            del self.SatDelEntropies[nq]
                            check_rtj = np.zeros(len(list(self.LocalSatellites.keys())))
                            # new_orbit = self.OrbitalPlane(self.OrbitalSystem.Planets['def'], float(ent),
                            #                                       self.Satellites[sats].IndividualAnomaly,
                            #                                       self.Satellites[sats].Azimuth_local,
                            #                                       self, stage_nos)
                    exit_list = sg[clustering != h_g]
                    cluster_list = clustering[clustering != h_g]
                    lock = threading.Lock()
                    lock.acquire()
                    print('--googoo--')
                    print(exit_list)
                    print(cluster_list)
                    lock.release()
                    asyncio.run(self.OrbitalSystem.set_list(exit_list, cluster_list))
                    ent_measure, ent_dev, ent_dict = self.evaluate_entropy()
                ent_outliers = []
                thu = list(self.LocalSatellites.keys())
                if len(thu) > 2:
                    z = np.abs(stats.zscore(100*self.AllEntropies))
                    print('ssxx')
                    print(z)
                    ent_outliers_idx = np.where(z > 2)[0]
                    print(ent_outliers_idx)
                    thu = np.array(thu)
                    ent_outliers = thu[ent_outliers_idx]
                    print(ent_outliers)
                for nq in list(self.LocalSatellites.keys()):
                    ou = self.LocalSatellites[nq].get_azimuth() # This is NOT redundant. Required to update
                    # Azimuth_local within satellite

                    # lock = threading.Lock()
                    # lock.acquire()
                    # print('***********************')
                    # print(str(self.OrbitalID))
                    # print('Azimuth times')
                    # print(self.LocalSatellites[nq].Azimuth_local)
                    # print('***********************')
                    # lock.release()

                    # if self.cal_delta(nq) > self.EntropyThreshold:
                    if nq in ent_outliers:
                        self.LocalSatellites[nq].StatelessReason = 'entropy'
                        asyncio.run(self.OrbitalSystem.signal_displacement(nq))
                        print('entropy push')
                        del self.LocalSatellites[nq]
                        del self.SatDelEntropies[nq]
                        check_rtj = np.zeros(len(list(self.LocalSatellites.keys())))

                    elif self.cal_delta(nq) > self.EntropyThreshold and len(thu)<=2:
                        self.LocalSatellites[nq].StatelessReason = 'entropy'
                        asyncio.run(self.OrbitalSystem.signal_displacement(nq))
                        print('entropy push')
                        del self.LocalSatellites[nq]
                        del self.SatDelEntropies[nq]
                        check_rtj = np.zeros(len(list(self.LocalSatellites.keys())))

                    elif abs(self.LocalSatellites[nq].IndividualAnomaly - self.LocalAnomaly) > self.AnomalyThreshold:
                        self.LocalSatellites[nq].StatelessReason = 'azimuth'
                        asyncio.run(self.OrbitalSystem.signal_displacement(nq))
                        print('anomaly push')
                        del self.LocalSatellites[nq]
                        del self.SatDelEntropies[nq]
                        check_rtj = np.zeros(len(list(self.LocalSatellites.keys())))
                    # elif abs(self.LocalSatellites[nq].Azimuth_local - self.Azimuth) > self.AzimuthThreshold:
                    #     self.LocalSatellites[nq].StatelessReason = 'azimuth'
                    #     asyncio.run(self.OrbitalSystem.signal_displacement(nq))
                    #     print(self.OrbitalSystem.StatelessSatellites)
                    #     del self.LocalSatellites[nq]
                    #     del self.SatDelEntropies[nq]
                    #     check_rtj = np.zeros(len(list(self.LocalSatellites.keys())))
                if len(list(self.LocalSatellites.keys())) < 1 and self.OrbitalID != 'def':
                    lock = threading.Lock()
                    lock.acquire()
                    print('-----------------------')
                    print(str(self.OrbitalID))
                    print('removing empty orbit')
                    print(len(list(self.LocalSatellites.keys())))
                    print('-----------------------')
                    lock.release()
                    asyncio.run(self.OrbitalSystem.signal_empty_orbit(self.OrbitalID))
                    break
                elif len(list(self.LocalSatellites.keys())) < 1 and self.OrbitalID == 'def':
                    break
                azi = 0.0
                for nl in list(self.LocalSatellites.keys()):
                    azi += float(self.LocalSatellites[nl].Azimuth_local)
                # lock = threading.Lock()
                # lock.acquire()
                # print('-----------------------')
                # print(str(self.OrbitalID))
                # print('testing beta')
                # print(len(list(self.LocalSatellites.keys())))
                # print('-----------------------')
                # lock.release()
                if len(list(self.LocalSatellites.keys())) > 0:
                    self.Azimuth = azi/len(list(self.LocalSatellites.keys()))
########

                check_rtj = np.zeros(len(list(self.LocalSatellites.keys())))
                lock = threading.Lock()
                lock.acquire()
                print(str(self.OrbitalID)+' Loop_count:' + str(counter_loop))

                for thread in threading.enumerate():
                    print(thread.name)
                    print(thread.is_alive())
                    print(list(self.LocalSatellites.keys()))
                    print(circle(list(self.LocalSatellites.keys())))
                lock.release()
                l_list = list(self.LocalSatellites.keys())
                lr_list = circle(l_list)
                for ns in list(self.LocalSatellites.keys()):
                    self.LocalSatellites[ns].reenter_stages(1.0, 'ss', 's_stage')
                now_sat = 0
                for ns in range(len(list(self.LocalSatellites.keys()))):
                    if ns == 0:
                        now_sat = next(lr_list)
                    next_sat = next(lr_list)
                    self.LocalSatellites[now_sat].transfer_stages(self.LocalSatellites[next_sat], 1.0)
                    # This for loop is set for aux stage upload (ensemble learning)
                    self.LocalSatellites[now_sat].prep_for_aux(1.0)
                    for r, der in enumerate(list(self.LocalSatellites.keys())):
                        if der == now_sat:
                            # print('der darrred')
                            continue
                        self.LocalSatellites[now_sat].request_aux_stages(self.LocalSatellites[der], 1.0, stage_id=str(r)
                                                                         , stage_level='s_stage')
                    # TODO: change to planet vol
                    rtj_flag = {'RTJ': 'False'}

                    self.LocalSatellites[now_sat].set_rtj(rtj_flag)
                    now_sat = next_sat
                # else:
                    # self.OrbitalSystem.signal_empty_orbit(self)
                    # self.ProcessThread.join()
                # if (ent_dev < abs(self.Threshold - ent_measure)) or (self.Threshold + ent_measure) > ent_dev:
                #     print('inner sanctum')
                #     temp_list = list(ent_dict.values())
                #     temp_list_2 = temp_list - ent_measure
                #     temp_max = max(temp_list_2)
                #     temp_idx = temp_list_2.index(temp_max)
                #     rem_id = list(ent_dict.keys())[list(ent_dict.values()).index(temp_list[temp_idx])]
                #     status = self.OrbitalSystem.signal_displacement(rem_id, (temp_max + ent_measure))
                #     if status:
                #         del self.LocalSatellites[rem_id]
                # self.LocalEntropy = ent_measure

                self.OrbitalSystem.update_orbital_status(self.OrbitalID, self.LocalEntropy)
                counter_loop += 1
            else:
                break

    @staticmethod
    def max_deviation(a):
        avg = sum(a, 0.0) / len(a)
        max_dev = max(abs(el - avg) for el in a)
        return max_dev

