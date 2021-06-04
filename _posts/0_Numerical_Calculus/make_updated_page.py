

data0 = open('1_python_intro.html')
lines = data0.readlines()

new_file = open("new_page.html",'w')
for line in lines:
	cond1 = line.find("<body")
	cond2 = line.find("body>")
	if(cond1>=0):
		header = open("../../header.html").read()
		header = header.replace("menu_style","../../menu_style")
		line = header + "\n<section class=\"section_style\">\n" + line
		print(line)
	if(cond2>=0):
		footer = open("../../footer.html").read()
		footer = footer.replace("menu_style","../../menu_style")
		line = "</section>\n\n" + line + footer 
		print(line)
	new_file.write(line)
new_file.close()
