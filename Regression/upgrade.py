import pip
from subprocess import call

for dist in pip.get_installed_distributions():
	call("C:\Python27\Scripts\pip.exe install --upgrade " + dist.project_name, shell=True)