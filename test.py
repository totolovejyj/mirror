import time
counter = 0 
while True :
	counter = counter + 1
	
	
	f = open('demofile.txt', 'w')
	f.write(str(counter))
	f.close()
	time.sleep(3)