import os
import subprocess
import time

root = '/home/yuemmawang/fathom/'
workloads = [
#('fathom/alexnet', 'alexnet')
#('fathom/vgg', 'vgg')
#('fathom/residual', 'residual')
('fathom/autoenc', 'autoenc')
]


capture_cmd = "capture_tpu_profile --service_addr=$TPU_IP:8466 --logdir=$HOME/xprof/perf_$WL_$BS --duration_ms=30000"

for path,wl in workloads:
  os.chdir(root)
  os.chdir(path)
  proc = subprocess.Popen(['bash', 'RUN'])
  
  while proc.poll() == None:
  
    #capture = subprocess.Popen(['capture_tpu_profile', '--service_addr=$TPU_IP:8466', '--logdir=$HOME/perf_' + wl.split('.')[0], '--duration_ms=1000'])
    capture_cmd = capture_cmd.replace('$WL', wl)
    capture_cmd = capture_cmd.replace('$BS', '128')
    f = open("/home/yuemmawang/fathom/scripts/CAPTURE",'w')
    f.write(capture_cmd)
    f.close()
    capture = subprocess.Popen(['bash', '/home/yuemmawang/fathom/scripts/CAPTURE'])
    print("Running capture %s", str(capture.pid))
    capture.wait()
    print("Finish capture %s", str(capture.pid))
    #time.sleep(10)
  
  if proc.poll() == None or capture.poll == None:
    print("Error!")
    time.sleep(1)

  
  
