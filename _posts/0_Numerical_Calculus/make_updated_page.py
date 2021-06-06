

data = open('1_python_intro.html')
lines = data.readlines()

new_file = open("Tutorial_1.html",'w')

header = open("../../header.html").read()
#header = header.replace("menu_style","../../menu_style")
header = header.replace("figures/","../../figures/")
#header = header.replace("./index.html","../../index.html")
header = header.replace("href=\"","href=\"../../")
header+= "\n <section style=\" float:left; padding:20px; width: 60%;\" >\n"

footer = open("../../footer.html").read()
footer = footer.replace("menu_style","../../menu_style")
footer = "\n </section>\n"+footer

new_file.write(header)
for line in lines: new_file.write(line)

new_file.write(footer)

'''
c1, c2 = True, True
for line in lines:
	cond1 = line.find("<body")
	cond2 = line.find("body>")
	if(cond1>=0 and c1):
		header = open("../../header.html").read()
		header = header.replace("menu_style","../../menu_style")
		line = header + "\n<section class=\"section_style\">\n" + line
		print(line) ; c1 = False
	if(cond2>=0 and c2):
		footer = open("../../footer.html").read()
		footer = footer.replace("menu_style","../../menu_style")
		line = "</section>\n\n" + line + footer 
		print(line) ; c2 = False
	new_file.write(line)
new_file.close()
'''
