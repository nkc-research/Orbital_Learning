import numpy as np
import time
import os

client_data_names = {1: '0',
                     2: '1',
                     3: '2',
                     4: '3',
                     5: '4',
                     6: '5',
                    }
planet_id = 10
path = '/mnt/data' + str(planet_id) + '/'

x_data = np.load('/Dataset/Base_X' + '.npy', allow_pickle=True)
y_data = np.load('/Dataset/Base_Y' + '.npy', allow_pickle=True)
with open(path + 'Base_X_data.npy', 'wb') as f1:
    np.save(f1, x_data)
with open(path + 'Base_Y_data.npy', 'wb') as f2:
    np.save(f2, y_data)
time.sleep(500)
frequency_nos = 200
time_count = {1: time.time(), 2: time.time(), 3: time.time(), 4: time.time(), 5: time.time(), 6: time.time()}
frequency_times = {1: 200, 2: 200, 3: 200, 4: 200, 5: 200, 6: 200}
previous_vals = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
count_runs = 0
key = True
while True:
    for pl, i in enumerate(list(client_data_names.keys())):
        l_time = time.time()
        if abs(l_time - time_count[i]) > frequency_times[i] or key:

            path = '/mnt/data'+str(client_data_names[i])+'/'
            # while True:
            #     if os.path.exists(path + 'open_flag.npy'):
            #         flag = np.load(path + 'open_flag.npy')
            #         if flag:
            #             time.sleep(10)
            #     else:
            #         break
            # f_flag = open(path + 'open_flag.npy', 'wb')
            # flag = True
            # np.save(f_flag, flag)
            x_data = np.load('/Dataset/Age_Split_X_' + str(client_data_names[i]) + '.npy', allow_pickle=True)
            y_data = np.load('/Dataset/Age_Split_Y_' + str(client_data_names[i]) + '.npy', allow_pickle=True)
            x_test = np.load('/Dataset/Test_Age_Split_X_' + str(client_data_names[i]) + '.npy', allow_pickle=True)
            y_test = np.load('/Dataset/Test_Age_Split_Y_' + str(client_data_names[i]) + '.npy', allow_pickle=True) #Age_
            x = x_data[int(previous_vals[i]):int(previous_vals[i]) + frequency_nos]
            y = y_data[int(previous_vals[i]):int(previous_vals[i]) + frequency_nos]
            previous_vals[i] = int(previous_vals[i]) + frequency_nos
            print('doing now!')
            with open(path + 'Pending_X_data.npy', 'wb') as f1:
                np.save(f1, x)
            with open(path + 'Pending_Y_data.npy', 'wb') as f2:
                np.save(f2, y)
            with open(path + '1.0_X_data.npy', 'wb') as f3:
                np.save(f3, x_test)
            with open(path + '1.0_Y_data.npy', 'wb') as f4:
                np.save(f4, y_test)
            # flag = False
            # np.save(f_flag, flag)
            # f_flag.close()
            time_count[i] = time.time()
            count_runs += 1
            # if count_runs >= 30:
            #     frequency_times[2] = 200
    key = False

    if count_runs > 500:
        break
