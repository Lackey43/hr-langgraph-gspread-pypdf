from merm import render_diagram


def outputImage(bytes):
	data = render_diagram(bytes)
	with open("chart.png" "wb") as img:
		img.write(data)
