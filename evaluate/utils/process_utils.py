from multiprocessing import Process

def launch_process(function, args):
	p = Process(target=function, args=args)
	p.daemon = True
	p.start()
	p.join()