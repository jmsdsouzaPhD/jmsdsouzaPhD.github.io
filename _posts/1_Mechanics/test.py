data = open("1-Harmonic_Oscillator.html")
lines = data.readlines()
txt = ""
for line in lines: txt += line

arq = open("write_test.txt","w")
arq.write(txt)
arq.close()
