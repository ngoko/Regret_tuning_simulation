from regret_tuning import RegretTuning

r = RegretTuning(datafile='./satunsat_runtime.csv', request_time_slot=600, conf_per_slot=160)
regret_period = 600*76
n = 300
k = 10
gplot_file_means = './satunsat_means_xp7.dat'
gplot_file_max = './satunsat_max_xp7.dat'


r.shuffle_data()
 
u =  r.run_method('lex', regret_period, k)
v =  r.run_method('rand', regret_period, k)
w =  r.run_method('randsort', regret_period, k)
x =  r.run_method('randsortdesc', regret_period, k)

y =  r.run_method('randw', regret_period, k)
z =  r.run_method('randwnoh', regret_period, k)

size = regret_period
for i in range(1, n):
  r.shuffle_data()

  t_u = r.run_method('lex', regret_period, k)
  u = [u[j]+t_u[j] for j in range(0,size)]
  
  t_v =  r.run_method('rand', regret_period, k)
  v = [v[j]+t_v[j] for j in range(0,size)]

  t_w =  r.run_method('randsort', regret_period, k)
  w = [w[j]+t_w[j] for j in range(0,size)]

  t_x = r.run_method('randsortdesc', regret_period, k)
  x = [x[j]+t_x[j] for j in range(0,size)]

  t_y =  r.run_method('randw', regret_period, k)
  y = [y[j]+t_y[j] for j in range(0,size)]

  t_z =  r.run_method('randwnoh', regret_period, k)
  z = [z[j]+t_z[j] for j in range(0,size)]  


u = [u[j]/n for j in range(0,size)]
v = [v[j]/n for j in range(0,size)]
w = [w[j]/n for j in range(0,size)]
x = [x[j]/n for j in range(0,size)]
y = [y[j]/n for j in range(0,size)]
z = [z[j]/n for j in range(0,size)]  


desc = open(gplot_file_means, 'w')  
desc.write('#t lex rand randsort randsortdesc randw randwnoh \n')
for i in range(0, size):
  desc.write(str(i)+' '+str(u[i])+' '+str(v[i])+' '+str(w[i])+' '+str(x[i])+' '+str(y[i])+' '+str(z[i])+'\n')

desc1 = open(gplot_file_max, 'w')  
desc1.write('# lex rand randsort randsortdesc randw randwnoh \n')
desc1.write(' '+str(max(u))+' '+str(max(v))+' '+str(max(w))+' '+str(max(x))+' '+str(max(y))+' '+str(max(z))+'\n')

