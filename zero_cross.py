import re

regex = '<p data-gazettes="PersonDetails" resource="this:person-1"><span content="(.+?)." datatype="xsd:string" property="person:hasPersonDetails">'
pattern = re.compile(regex)
f = open("temp.html", "r")
html = f.read()
value = re.findall(pattern, html)
print value
