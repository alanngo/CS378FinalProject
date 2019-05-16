w= open("condorize.sh","w+")
f= open("/u/matthewp/research/scripts/condorize.sh", "r")


if f.mode == 'r':
	contents =f.read()

f1 = f.readlines();

for x in f1:
	w.write(x)

w.close()
f.close()