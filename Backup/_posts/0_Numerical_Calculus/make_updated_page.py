
num = 0
while(num<1 or num>7):
	num = int(input("Insira o número do tutorial [1-7]: "))
	if(num<1 or num>7): print("\nError: Número Invalido!\n")

name1 = ("ipython/%d_python_intro.html"%num)
name2 = ("Tutorial_%d.html"%num)

data = open(name1)
new_file = open(name2,'w')
lines = data.readlines()

menu = open("Menu.txt").read()

header = open("../../header.html").read()
header = header.replace("menu_style","../../menu_style")
header = header.replace("figures/","../../figures/")
header = header.replace("href=\"","href=\"../../")
header = header.replace("<title></title>","<title>Introdução ao Python: %s</title>"%name2)
header+= "\n <body style=\"background-image: url('../../figures/programming.jpg');background-attachment: fixed;background-size: 100% 100%;\">"
header+= "\n<section style=\"padding:20px; width: 80%; margin-left:10%; background-color:rgba(255,255,255,0.98);\" >\n"
header+=menu
title = ("<title>Python: Tutorial %d</title>\n"%num)
header += title

footer = open("../../footer.html").read()
footer = footer.replace("menu_style","../../menu_style")

buttons = ("\n<br/><br/><center>\n<a href=\"ipython/%d_python_intro.ipynb\" download><button class=\"button1\"><i class=\"fa fa-download\" aria-hidden=\"true\"></i> Download .ipynb</button></a>&emsp;"%num)
buttons += ("\n<a href=\"python/code%d.py\" download><button class=\"button1\"><i class=\"fa fa-download\" aria-hidden=\"true\"></i>  Download .py</button></a>\n</center><br/><br/>"%num)
footer = buttons+"\n</section>\n</body>\n"+footer


buttons_style = (
"\n\t"+
"<style>/* CSS Script */" + "\n\t\t" +
	".button1{" + "\n\t\t\t" +
		"background-color:blue;" + "\n\t\t\t" +
		"color:white;" + "\n\t\t\t" +
		"text-align:center;" + "\n\t\t\t" +
		"margin: 3px 5px;" + "\n\t\t\t" +
		"padding: 15px 32px;" + "\n\t\t\t" +
		"font-weight:bold;" + "\n\t\t\t" +
		"font-size:14px;" + "\n\t\t\t" +
		"text-decoration: none;" + "\n\t\t\t" +
		"border-radius:20px;" + "\n\t\t\t" +
		"cursor:pointer;" + "\n\t\t\t" +
		"box-shadow: 0 3px 3px black;" + "\n\t\t" +
	"}" + "\n\t" +
"</style>" + "\n</head>"
)

header = header.replace("</head>","%s"%buttons_style)

new_file.write(header)
for line in lines: new_file.write(line)
new_file.write(footer)

