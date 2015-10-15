from regret_tuning import RegretTuning

r = RegretTuning(datafile='./satunsat_runtime.csv', request_time_slot=600, conf_per_slot=100)
regret_period = 600*76
n = 300
k = 10
gplot_file_means = './satunsat_means_xp4.dat'
gplot_file_max = './satunsat_max_xp4.dat'


r.shuffle_data()
 
u =  r.run_method('randw', regret_period, 10)
v =  r.run_method('randw', regret_period, 20)
w =  r.run_method('randw', regret_period, 30)
x =  r.run_method('randw', regret_period, 40)


size = regret_period
for i in range(1, n):
  r.shuffle_data()

  t_u = r.run_method('randw', regret_period, 10)
  u = [u[j]+t_u[j] for j in range(0,size)]
  
  t_v =  r.run_method('randw', regret_period, 20)
  v = [v[j]+t_v[j] for j in range(0,size)]

  t_w =  r.run_method('randw', regret_period, 30)
  w = [w[j]+t_w[j] for j in range(0,size)]

  t_x = r.run_method('randw', regret_period, 40)
  x = [x[j]+t_x[j] for j in range(0,size)]


u = [u[j]/n for j in range(0,size)]
v = [v[j]/n for j in range(0,size)]
w = [w[j]/n for j in range(0,size)]
x = [x[j]/n for j in range(0,size)]


desc = open(gplot_file_means, 'w')  
desc.write('#t lex rand(10) rand(20) rand(30) rand(40)\n')
for i in range(0, size):
  desc.write(str(i)+' '+str(u[i])+' '+str(v[i])+' '+str(w[i])+' '+str(x[i])+'\n')

desc1 = open(gplot_file_max, 'w')  
desc1.write('#lex rand(10) rand(20) rand(30) rand(40)\n')
desc1.write(' '+str(max(u))+' '+str(max(v))+' '+str(max(w))+' '+str(max(x))+'\n')

