import numpy as np
import os

data = open("1-Harmonic_Oscillator.html")
lines = data.readlines()
n = len(lines)
for i in range(n):
	line = lines[i]
	if(line.startswith("<head")):	
		print("<head> [line: %d]"% n)
		print(line)
	if("/head>" in line): print("\t</head> [line: %d]"% n)
	if(line.startswith("<body")):	
		print("<body> [line: %d]"% n)
		print(line)
	if("/body>" in line): print("\t</body> [line: %d]"% n)
