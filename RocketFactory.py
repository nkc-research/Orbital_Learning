# All helper modules to be used in development of orbital learning can be found here.
from pydantic import BaseModel
import OrbitalPlane
import Planet

class BuildDefaults:

    def __init__(self):
        pass

    @staticmethod
    def generate_planet(port=5005, vol=10):
        planet = Planet.Planet(port, vol)
        return {'def': planet}  # todo: write a code to generate docker container as a server

    @staticmethod
    def initialise_orbital_planes(system_name, planet):
        plane1 = OrbitalPlane.OrbitalPlane(planet, 1.0, 1.0, 200, system_name, [], id='def')  # 500 is default azimuth, change it
        # with need.
        return {'def': plane1}  # todo: write a code to build orbital planes



