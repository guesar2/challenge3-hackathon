import qnexus as qnx
qnx.login()
qnx.devices.get_all().df()