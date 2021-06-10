
num = 0
while(num<1 or num>7):
	num = int(input("Insira o número do tutorial [1-7]: "))
	if(num<1 or num>7): print("\nError: Número Invalido!\n")

name1 = ("%d_python_intro.html"%num)
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

footer = open("../../footer.html").read()
footer = footer.replace("menu_style","../../menu_style")
footer = "\n</section>\n</body>\n"+footer

new_file.write(header)
for line in lines: new_file.write(line)
new_file.write(footer)

