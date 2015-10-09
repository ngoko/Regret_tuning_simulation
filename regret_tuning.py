import random

class RegretTuning:

  def __init__(self, **kwargs):
'''
   In the construction, we set:
     1. The method we want to execute (Lex, Rand, etc.)
     2. The SAT runtime file (classified by instances and algorithms)
     3. The request_time _slot is the duration between two calls 
     for retrieving a new optimal configuration from the active learning framework 
     4. The conf_per_slot is the maximal number of evaluations that were done
     by the active learning framework between two requests
'''
    if 'method' in kwargs:
        self.method = kwargs['method']
    if 'datafile' in kwargs:
        self.datafile = kwargs['datafile']
    else:
         self.datacolumn = 2
    if 'request_time_slot' in kwargs:
        self.request_time_slot = kwargs['request_time_slot']
    if 'conf_per_slot' in kwargs:
    	self.conf_per_slot = kwargs['conf_per_slot']

    self.runtime = []

#   Load the datafile    
    desc = open(path.join(self.datafile), 'r')
    data = desc.readlines()
    items = data[1:len(data)]
    for line in items:	
    	line_run = line.split(',')
        for j in range(2, len(line_run)):
          self.runtime[i].append(float(line_run(j)))


  def get_runtime_Lex(self, runtime):
'''
   Return the local runtime of configurations found by the 
   lexicographic oracle. 
   With n configurations, the runtime is a vector of n/conf_per_slot elements
'''
    m = len(runtime[0,:])
    n = len(runtime[:,0])
    results = []

    opt_cost = 0
    opt_conf = 0
    for i in range(0, n)
	opt_cost += runtime[i,0]      

    results.append(opt_cost)
    conf_eval = 0

    for j in range(1, m):
	cur_cost = 0       
	i = 0        
	while cur_cost <= opt_cost:
	   cur_cost += runtime[i,j]	        
	   i += 1	
           conf_eval += 1
           if (conf_eval ==  self.conf_per_slot):        
             results.append(opt_cost)
             conf_eval = 0
        if(cur_cost < opt_cost):
             opt_cost = cur_cost
    if(conf_eval > 0):
	results.append(opt_cost)
    return results


  def get_runtime_Rand_1D(self, runtime):
'''
   Return the local runtime of configurations found by the 
   random oracle (configurations and benchmark are shuffled randomly)
   With n configurations, the runtime is a vector of n/conf_per_slot elements
'''
    m = len(runtime[0,:])
    n = len(runtime[:,0])
    results = []

#   shuffle the algorithms keys
    akeys = []
    for i in range(0, m):
      akeys.append(i)
    akeys = random.shuffle(akeys)
#   shuffle  the instances  keys
    ikeys = []
    for i in range(0, n):
      ikeys.append(i)
    ikeys = random.shuffle(ikeys)
    	
    opt_cost = 0
    opt_conf = 0
    for i in range(0, n)
	opt_cost += runtime[i,akeys[0]]      

    results.append(opt_cost)
    conf_eval = 0

    for j in range(1, m):
	cur_cost = 0       
	i = 0        
	while cur_cost <= opt_cost:
	   cur_cost += runtime[ikeys[i],akeys[j]]	        
	   i += 1	
           conf_eval += 1
           if (conf_eval ==  self.conf_per_slot):        
             results.append(opt_cost)
             conf_eval = 0
        if(cur_cost < opt_cost):
             opt_cost = cur_cost
    if(conf_eval > 0):
	results.append(opt_cost)
    return results

           

  def integrate_regret(self, local_results_dist, regret_period):
'''
   Return the regret we have from 0 to regret_period,
   assuming that the results are the oracle answers follow the  
   distribution described in  local_results_dist
'''
    opt = min(local_result_dist) 
    results = []
    results.append(local_results_dist[0]-opt)
    d = 1
    l_opt = 0
    for i in range(1, regret_period):
        if(d == self.request_time_slot):
           l_opt++
           d = 1
        else:
           d++
        results.append(local_results_dist[l_opt]-opt)
