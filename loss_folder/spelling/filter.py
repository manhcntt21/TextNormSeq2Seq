data = []
with open("test.txt", "r") as outfile:
	lines = outfile.readline().rstrip()
	i = 1
	while lines:
		tmp = lines	
		if i <= 9:  
			data.append(str(tmp[0]) + '\t' + str(tmp[1:]))
			lines = outfile.readline().rstrip()
			i = i + 1
		elif i  <= 99:
			data.append( str(tmp[0])+ str(tmp[1])+ '\t' + str(tmp[2:]))
			lines = outfile.readline().rstrip()
			i = i + 1
		else: 
			data.append( str(tmp[0]) + str(tmp[1]) + str(tmp[2]) + '\t' + str(tmp[3:]))
			lines = outfile.readline().rstrip()
			i = i + 1
#		print(lines[0])
#		lines = outfile.readline()
with open("test1.txt","w") as f:
	for i in data:
		f.write(i)
		f.write('\n')
