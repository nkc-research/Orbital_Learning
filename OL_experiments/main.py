import docker

DC = docker.from_env()
planet_id = 10
no_clients = [0, 1, 2, 3, 4, 5, planet_id]
vols = {}
for i in no_clients:
    DC.volumes.create(name='localvol'+str(i), driver='local')
    inter = {'bind': '/mnt/data'+str(i)+'/', 'mode': 'rw'}
    vols['localvol'+str(i)] = inter

# path to dataset needs to be set here.
vols[' your path to dataset'] = {'bind': '/Dataset', 'mode': 'rw'} 
cont_id = DC.containers.run("orbitallearning/ol_experiments:latest", detach=True, volumes=vols)
