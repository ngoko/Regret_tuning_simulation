import numpy
import random
from sklearn.cluster import KMeans
from sets import Set

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
	while (cur_cost <= opt_cost & i < n):
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


  def get_runtime_Rand_Rand(self, runtime):
'''
   Return the local runtime of configurations found by the 
   random oracle (configurations and benchmark are shuffled randomly)
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
	while (cur_cost <= opt_cost & i < n):
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



  def get_runtime_Rand_HardSoft(self, runtime):
'''
   Return the local runtime of configurations found by the 
   random oracle (configurations are shuffled randomly, benchmark are chosen randomly but 
   with hard instances in priority)
'''
    m = len(runtime[0,:])
    n = len(runtime[:,0])
    results = []
    delta = []
    delta_w = []

    w_index = 0

#   shuffle the algorithms keys
    akeys = []
    for i in range(0, m):
      akeys.append(i)
    akeys = random.shuffle(akeys)
    	
    opt_cost = 0
    opt_conf = 0
    for i in range(0, n):
	opt_cost += runtime[i,akeys[0]]      
        delta[i,0] = runtime[i,akeys[0]]      

    results.append(opt_cost)
    for u in range(0,n):
      delta_w[u, 0] = delta[u,0]
    w_index++

    conf_eval = 0

    for j in range(1, m):
#       separate instances in hard and softs 
        ikeys = self.separate_hard_soft(delta_w, n)

	cur_cost = 0       
	i = 0
	while (cur_cost <= opt_cost & i < n):
	   cur_cost += runtime[ikeys[i],akeys[j]]	        
           delta[ikeys[i],j] = runtime[ikeys[i],akeys[j]]
	   i++
           conf_eval++
           if (conf_eval ==  self.conf_per_slot):        
             results.append(opt_cost)
             conf_eval = 0

        if(cur_cost < opt_cost):
             opt_cost = cur_cost
   
        if (i == n-1):
            for u in range(0,n):
              delta_w[u, w_index] = delta[u,j]
            w_index++
                   
    if(conf_eval > 0):
	results.append(opt_cost)
    return results


  def get_runtime_Rand_W(self, runtime, k):
'''
   Return the local runtime of configurations found by the 
   random oracle (configurations are shuffled randomly, benchmark are chosen randomly but 
   with hard instances in priority)
   k defines the number of partitions for the k interesection problems.
'''
    m = len(runtime[0,:])
    n = len(runtime[:,0])
    results = []
    delta = []
    delta_w = []
    delta_l = []

    w_index = 0
    l_index = 0

#   shuffle the algorithms keys
    akeys = []
    for i in range(0, m):
      akeys.append(i)
    akeys = random.shuffle(akeys)
    	
    opt_cost = 0
    opt_conf = 0
    for i in range(0, n):
	opt_cost += runtime[i,akeys[0]]      
        delta[i,0] = runtime[i,akeys[0]]      

    results.append(opt_cost)
    for u in range(0,n):
      delta_w[u, 0] = delta[u,0]
    w_index++
	
    conf_eval = 0

    for j in range(1, m):
#       separate instances in hard and softs 
        ikeys = self.separate_wisdom_hard_soft(k, delta_w, delta_l, n)

	cur_cost = 0       
	i = 0
        for u in range(0,n):
          delta[u,j] = -1         
	while (cur_cost <= opt_cost & i < n):
	   cur_cost += runtime[ikeys[i],akeys[j]]	        
           delta[ikeys[i],j] = runtime[ikeys[i],akeys[j]]
	   i += 1	
           conf_eval += 1
           if (conf_eval ==  self.conf_per_slot):        
             results.append(opt_cost)
             conf_eval = 0

        if(cur_cost < opt_cost):
             opt_cost = cur_cost
   
        if (i == n-1):
          for u in range(0,n):
            delta_w[u, w_index] = delta[u,j]
          w_index++
        else:
          for u in range(0,n):
            delta_l[u, l_index] = delta[u,j]
          l_index++


                   
    if(conf_eval > 0):
	results.append(opt_cost)
    return results

           
  def separate_hard_soft(delta_w, n):
'''
   Based on the runtime in delta[akeys[0]..akeys[j]]
   separate the instances in hard and softs
'''
    k_means = KMeans(n_clusters=2)
    k_means.fit(delta_w)
    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    norm1  = numpy.linalg.norm(centroids[0,:])
    norm2  = numpy.linalg.norm(centroids[1,:])

    hard_index = 1
    if norm1 > norm2:
      hard_index = 0
   
    hard = []
    soft = []
  
    for i in range(0,n):
      if labels[i] == hard_index:
        hard.append(i)
      else:
        soft.append(i)
   
    keys = random.shuffle(hard)
    soft = random.shuffle(soft);
    for e in soft:
      keys.append(e)
   

    return keys
   

  def separate_wisdom_hard_soft(k, delta_w, delta_l, n):
'''
   Based on the runtime in delta[akeys[0]..akeys[j]]
   separate the instances in hard and softs but include the wisdom of the loser
'''
    hard_soft = self.separate_hard_soft(delta_w, n)
    set_delta_l = []
    for i in range(0,n):
      set_delta_l[i] = Set([])
      for j in range(0, len(delta_l[i])):
        if(delta_l[i,j] != -1):
          set_delta_l[i].add(delta_l[i,j])
    wisdom_loser = random.shuffle(self.k_intersection(k, set_delta_l, n))
    keys = [e for e in wisdom_loser]
    for (e in hard_soft): 
      if not(e in wisdom_loser): 
        keys.append(e)
    return keys
	

  def k_intersection(k, set_delta_l, n)
'''
    result of the k_intersection algorithm on delta_l
'''
    indices = []
    card  = []
    for i in range(0, n):
      card[i] = len(set_delta_l[i])
    opt = 0
    for i in range(1,n):
      if (card[i] > card[opt]):
        opt = i
    indices.append(opt)
    inter = set_delta_l[opt]
#  select the remaining k sets
    for j in range(1,k):
      opt = 0
      while (len(set_delta_l[opt]) == 0):
        opt++
      tmp = inter.copy()
      card[opt] = len(tmp.intersection_update(set_delta_l[opt]))
      for i in range(1,n):
        if (len(set_delta_l[i]) > 0):
          tmp = inter.copy()
          card[i] = len(tmp.intersection_update(set_delta_l[i]))       
        else:
          card[i] = 0
        if (card[i] > card[opt]):
          opt = i
      if (card[opt] > 0):
        indices.append(opt)
      inter = inter.intersection_update(set_delta_l[opt])
      for i in range(0,n):
        if(card[i] > 0):
          set_delta_l[i].difference_update(set_delta_ l[opt])

     return indices



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
