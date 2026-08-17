
name1 = "1-Harmonic_Oscillator.html"
name2 = "2-Damped_Harmonic_Oscillator.html"
name3 = "3-Coupled_Harmonic_Oscillators.html"
names = [name1,name2,name3]

titles = ["Oscilador Harmônico","Oscilador Harmônico Amortecido","Osciladores Harmônicos Acoplados"]

for i in range(3):
	name = names[i]
	title = titles[i]
	#=================================================================
	header = open("../../header.html").read()	# Abrindo o HEADE.HTML
	header = header.replace("menu_style","../../menu_style")
	header = header.replace("figures/","../../figures/")
	header = header.replace("href=\"","href=\"../../")
	header = header.replace("<title></title>","<title>Mecânica Clássica: %s</title>"%name[2:-5])
	#header+= "\n <body style=\"background-image: url('../../figures/programming.jpg');background-attachment: fixed;background-size: 100% 100%;\">"
	#header+= "\n<section class=\"section_style\" >\n"
	lines_header = header.split('\n') ; len_hdr = len(lines_header)
	for i in range(len_hdr):
		if("utf-8" in lines_header[i]): 		idx_head_1 = i+1
		if("</head>" in lines_header[i]):   idx_head_2 = i
		if("menu_style.css" in lines_header[i]):   idx_header_1 = i
	idx_header_2 = len_hdr-1

	# Preciso pegar o conteudo do <head>...</head> e do <header>...</header>
	head_conteudo = "" ;	header_conteudo = ""
	for i in range(idx_head_1,idx_head_2): 	 head_conteudo += lines_header[i]+"\n"
	for i in range(idx_header_1,idx_header_2): header_conteudo += lines_header[i]+"\n"
	#=================================================================
	footer = open("../../footer.html").read()	# Abrindo o FOOTER.HTML
	footer = footer.replace("menu_style","../../menu_style")
	#=================================================================
	# Criando um Aside com o vídeo da simulação
	aside = "\n<aside class=\"aside_style\">\n"
	aside+= "<h3 style=\"text-align: left; padding: 1%; color: white; background-color: rgb(3, 0, 161);\">Animação com o Blender 2.8</h3><br/>\n"
	aside+= "<video width=\"100%\" controls loop>\n"
	aside+= "  <source src=\"%s.mp4\" type=\"video/mp4\">\n"%name[0:-5]
	aside+= "  Your browser does not support HTML video.\n"
	aside+= "</video>\n"
	aside+= "</aside>\n"
	#=================================================================
	# Montando a versão final da pagina html
	
	data = open("simple_html/"+name)
	new_file = open(name,'w')
	lines = data.readlines() ; n = len(lines)
	for i in range(n):
		if("<head>" in lines[i]):  idx_head_1 = i
		if("</head>" in lines[i]): idx_head_2 = i
		if("<body" in lines[i]): 	
			idx_bdy1 = i
			lines[i] = lines[i].replace("\"JupyterLab Light\"", "\"JupyterLab Light\" style=\"background-image: url('../../figures/programming.jpg');background-attachment: fixed;background-size: 100% 100%;\"")
		if("/body>" in lines[i]): 	idx_bdy2 = i
		if("padding: var(--jp-notebook-padding);" in lines[i]):
			lines[i] = lines[i].replace("padding: var(--jp-notebook-padding);","/*padding: var(--jp-notebook-padding);*/")
		
	for i in range(idx_head_1+1):
		new_file.write(lines[i])
	new_file.write("\n"+head_conteudo+"\n")
	for i in range(idx_head_1+1,idx_bdy1+1):
		new_file.write(lines[i])
	
	new_file.write("\n\n"+header_conteudo+"\n\n<section class=\"section_style\" >\n\n")
	new_file.write("\n\n<h3 style=\"font-size:20pt;color: white; background-color: rgb(3,0,161); padding:%s; text-align: center;\">%s</h3>\n\n"%("1%",title))
	for i in range(idx_bdy1+1,idx_bdy2):
		new_file.write(lines[i])
	new_file.write("\n\n</section>\n\n")
	new_file.write(aside)
	new_file.write(footer+"\n\n </body>\n\n</html>")

