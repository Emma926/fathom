import os
import subprocess
import time

root = '/home/yuemmawang/fathom/'
output = "/home/yuemmawang/fathom/outputs_sweep/"

cmd = " python $WL.py --master grpc://$TPU_IP:8470  \
  --data_dir=gs://cloudtpu-imagenet-data/train \
  --learning_rate $LR \
  --batch_size $BS \
  --train_steps $STEP" 

workloads = [
#('fathom/alexnet', 'alexnet'),
('fathom/vgg', 'vgg'),
('fathom/residual', 'residual'),
#('fathom/autoenc', 'autoenc')
]

if not os.path.exists(output):
  print("Creating directory " + output)
  os.makedirs(output)



for path,wl in workloads:
  for step in [1000]:
    os.chdir(root)
    os.chdir(path)
    if wl == 'alexnet':
      bs = 128
    if wl == 'resnet':
      bs = 32
    if wl == 'vgg':
      bs = 32
    cmd_curr = cmd[:]
    cmd_curr = cmd_curr.replace('$WL', wl)
    cmd_curr = cmd_curr.replace('$LR', '0.001')
    cmd_curr = cmd_curr.replace('$BS', str(bs))
    cmd_curr = cmd_curr.replace('$STEP', str(step))
    f = open('RUN_gen','w')
    f.write(cmd_curr)
    f.close()

    os.system('cp RUN_gen ' + os.path.join(output, 'RUN.' + str(bs) + '.' + str(step)))
    with open(os.path.join(output, 'runtime-' + wl + '-' + str(bs) + '-' + str(step)), 'w') as outfile:
      proc = subprocess.Popen(['bash', 'RUN_gen'], stdout=outfile)
      proc.wait()
