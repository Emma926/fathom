import os
import subprocess
import time

root = '/home/yuemmawang/fathom/'
output = "/home/yuemmawang/fathom/outputs_test/"

cmd = " python $WL.py --master grpc://$TPU_IP:8470  \
  --data_dir=gs://cloudtpu-imagenet-data/train \
  --learning_rate $LR \
  --batch_size $BS \
  --train_steps $STEP" 

workloads = [
('fathom/alexnet', 'alexnet'),
('fathom/vgg', 'vgg'),
('fathom/residual', 'residual'),
#('fathom/autoenc', 'autoenc')
]

if not os.path.exists(output):
  print("Creating directory " + output)
  os.makedirs(output)

step = 100

for path,wl in workloads:
  for i in range(1):
    os.chdir(root)
    os.chdir(path)

    cmd = cmd.replace('$WL', wl)
    cmd = cmd.replace('$LR', '0.001')
    cmd = cmd.replace('$BS', '128')
    cmd = cmd.replace('$STEP', str(step))
    f = open('RUN_gen','w')
    f.write(cmd)
    f.close()

    os.system('cp RUN_gen ' + os.path.join(output, 'RUN.' + str(i)))
    with open(os.path.join(output, 'runtime-' + wl + '-' + str(i)), 'w') as outfile:
      proc = subprocess.Popen(['bash', 'RUN_gen'], stdout=outfile)
      proc.wait()
