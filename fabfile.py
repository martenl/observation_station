from fabric import Connection

with Connection('***', user='***', connect_kwargs={'password': '***'}) as c:
    c.put('detect_video.py', 'coral/pycoral/examples/detect_video.py')
    c.put('detect_video2.py', 'coral/pycoral/examples/detect_video2.py')
    #c.put('C:\\Users\\User\\Downloads\\bike_ride.mp4', 'coral/pycoral/test_data/bike_ride.mp4')
