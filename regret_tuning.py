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
    else:
       self.method = 'lex'
    if 'datafile' in kwargs:
        self.datafile = kwargs['datafile']
    else:
         self.datacolumn = 2
    if 'request_time_slot' in kwargs:
        self.request_time_slot = kwargs['request_time_slot']
    if 'conf_per_slot' in kwargs:
    	self.conf_per_slot = kwargs['conf_per_slot']

    self.runtime = []
    self.akeys = []
    self.ikeys = []

#   Load the datafile    
    desc = open(self.datafile, 'r')
    data = desc.readlines()
    items = data[1:len(data)]
    i = 0
    for line in items:	
    	line_run = line.split(',')
        tmp = []
        for j in range(2, len(line_run)):
          tmp.append(float(line_run[j]))
        self.runtime.append(tmp)
        i += 1

  def shuffle_data(self):
    n = len(self.runtime)
    m = len(self.runtime[0])
#   shuffle the algorithms keys
    self.akeys = [i for i in range(0, m)]
    random.shuffle(self.akeys)
#   shuffle  the instances  keys
    self.ikeys = [i for i in range(0, n)]
    random.shuffle(self.ikeys)

  def get_runtime_Lex_pure(self, runtime):
    '''
     Return the local runtime of configurations found by the 
     lexicographic oracle. 
     With n configurations, the runtime is a vector of n/conf_per_slot elements
    '''
    n = len(runtime)
    m = len(runtime[0])
    results = []

    opt_cost = 0
    opt_conf = 0
    for i in range(0, n):
	opt_cost += runtime[i][0]      

    results.append(opt_cost)
    conf_eval = 0

    for j in range(1, m):
	cur_cost = 0       
	i = 0        
	while (cur_cost <= opt_cost and i < n):
	   cur_cost += runtime[i][j]	        
	   i += 1	
           conf_eval += 1
           if (conf_eval ==  self.conf_per_slot):        
             results.append(opt_cost)
             conf_eval = 0
#        print i, j, cur_cost, opt_cost
        if(cur_cost < opt_cost):
             opt_cost = cur_cost
    if(conf_eval > 0):
	results.append(opt_cost)
    return results


  def get_runtime_Lex(self, runtime):
    '''
     Return the local runtime of configurations found by the 
     random oracle (configurations and benchmark are shuffled randomly)
    '''
    n = len(runtime)
    m = len(runtime[0])
    results = []

    	
    opt_cost = 0
    opt_conf = 0
    for i in range(0, n):
	opt_cost += runtime[i][self.akeys[0]]      

    results.append(opt_cost)
    conf_eval = 0

    for j in range(1, m):
	cur_cost = 0       
	i = 0        
	while (cur_cost <= opt_cost and i < n):
	   cur_cost += runtime[self.ikeys[i]][self.akeys[j]]	        
	   i += 1	
           conf_eval += 1
           if (conf_eval ==  self.conf_per_slot):        
             results.append(opt_cost)
             conf_eval = 0
#        print i, j, cur_cost, opt_cost
        if(cur_cost < opt_cost):
             opt_cost = cur_cost
    if(conf_eval > 0):
	results.append(opt_cost)
    return results



  def get_runtime_Rand_HardSoft(self, runtime):
    '''
     Return the local runtime of configurations found by the 
     random oracle (configurations are shuffled randomly, benchmark are chosen randomly  
     with hard instances in priority)
    '''
    n = len(runtime)
    m = len(runtime[0])
    results = []
    delta = []
    delta_w = []

    w_index = 0

#   shuffle the algorithms keys
    l_akeys = [self.akeys[i] for i in range(0, m)]
    random.shuffle(l_akeys)
    l_ikeys = [self.ikeys[i] for i in range(0, n)]
    random.shuffle(l_ikeys)
    	
    opt_cost = 0
    opt_conf = 0
    delta = [[0 for i in range(0,m)] for j in range(0,n)] 
    for i in range(0, n):
	opt_cost += runtime[self.ikeys[i]][l_akeys[0]]      
        delta[self.ikeys[i]][0] = runtime[self.ikeys[i]][l_akeys[0]]      

    results.append(opt_cost)
    delta_w = [[0 for i in range(0,m)] for j in range(0,n)]     
    for u in range(0,n):
      delta_w[u][0] = delta[u][0]
    w_index = w_index+1

    conf_eval = 0

    for j in range(1, m):
#       separate instances in hard and softs 
        l_ikeys = self.separate_hard_soft(delta_w, n)

	cur_cost = 0       
	i = 0
	while (cur_cost <= opt_cost and i < n):
	   cur_cost += runtime[l_ikeys[i]][l_akeys[j]]	        
           delta[l_ikeys[i]][j] = runtime[l_ikeys[i]][l_akeys[j]]
	   i = i+1
           conf_eval += 1
           if (conf_eval ==  self.conf_per_slot):        
             results.append(opt_cost)
             conf_eval = 0
#        print i, j, cur_cost, opt_cost
        if(cur_cost < opt_cost):
             opt_cost = cur_cost
   
        if (i == n-1):
            for u in range(0,n):
              delta_w[u][w_index] = delta[u][j]
            w_index+=1
                   
    if(conf_eval > 0):
	results.append(opt_cost)
    return results


  def get_runtime_Rand_HardSoft_sort(self, runtime):
    '''
     Return the local runtime of configurations found by the 
     random oracle (configurations are shuffled randomly, benchmark are chosen randomly  
     with hard instances in priority)
    '''
    n = len(runtime)
    m = len(runtime[0])
    results = []
    delta = []
    delta_w = []

    w_index = 0

#   shuffle the algorithms keys
    l_akeys = [self.akeys[i] for i in range(0, m)]
    random.shuffle(l_akeys)
    l_ikeys = [self.ikeys[i] for i in range(0, n)]
    random.shuffle(l_ikeys)
    	
    opt_cost = 0
    opt_conf = 0
    delta = [[0 for i in range(0,m)] for j in range(0,n)] 
    for i in range(0, n):
	opt_cost += runtime[self.ikeys[i]][l_akeys[0]]      
        delta[self.ikeys[i]][0] = runtime[self.ikeys[i]][l_akeys[0]]      

    results.append(opt_cost)
    delta_w = [[0 for i in range(0,m)] for j in range(0,n)]     
    for u in range(0,n):
      delta_w[u][0] = delta[u][0]
    w_index = w_index+1

    conf_eval = 0

    for j in range(1, m):
#       separate instances in hard and softs 
        l_ikeys = self.separate_hard_soft_sort(delta_w, n)

	cur_cost = 0       
	i = 0
	while (cur_cost <= opt_cost and i < n):
	   cur_cost += runtime[l_ikeys[i]][l_akeys[j]]	        
           delta[l_ikeys[i]][j] = runtime[l_ikeys[i]][l_akeys[j]]
	   i = i+1
           conf_eval += 1
           if (conf_eval ==  self.conf_per_slot):        
             results.append(opt_cost)
             conf_eval = 0
#        print i, j, cur_cost, opt_cost
        if(cur_cost < opt_cost):
             opt_cost = cur_cost
   
        if (i == n-1):
            for u in range(0,n):
              delta_w[u][w_index] = delta[u][j]
            w_index+=1
                   
    if(conf_eval > 0):
	results.append(opt_cost)
    return results


  def get_runtime_Rand_W(self, runtime, k):
    '''
     Return the local runtime of configurations found by the 
     random oracle (configurations are shuffled randomly, benchmark are chosen randomly 
     with the windom of losers and hard instances in priority)
     k defines the number of partitions for the k interesection problems.
    '''
    n = len(runtime)
    m = len(runtime[0])
    results = []
    delta = []
    delta_w = []
    delta_l = []

    w_index = 0
    l_index = 0

#   shuffle the algorithms keys
    l_akeys = [self.akeys[i] for i in range(0, m)]
    random.shuffle(l_akeys)
    l_ikeys = [self.ikeys[i] for i in range(0, n)]
    random.shuffle(l_ikeys)
    	
    opt_cost = 0
    opt_conf = 0
    delta = [[0 for i in range(0,m)] for j in range(0,n)] 
    for i in range(0, n):
	opt_cost += runtime[self.ikeys[i]][l_akeys[0]]      
        delta[self.ikeys[i]][0] = runtime[self.ikeys[i]][l_akeys[0]]      

    results.append(opt_cost)
    delta_w = [[0 for i in range(0,m)] for j in range(0,n)]     
    for u in range(0,n):
      delta_w[u][0] = delta[u][0]
    w_index +=1
    conf_eval = 0

    for j in range(1, m):
#       separate instances in hard and softs 
        l_ikeys = self.separate_wisdom_hard_soft(k, delta_w, delta_l, n)
	cur_cost = 0       
	i = 0
        for u in range(0,n):
          delta[u][l_akeys[j]] = -1         
	while (cur_cost <= opt_cost and i < n):
	   cur_cost += runtime[l_ikeys[i]][l_akeys[j]]	        
           delta[l_ikeys[i]][j] = runtime[l_ikeys[i]][j]
	   i += 1	
           conf_eval += 1
           if (conf_eval ==  self.conf_per_slot):        
             results.append(opt_cost)
             conf_eval = 0

        if(cur_cost < opt_cost):
             opt_cost = cur_cost
   
        if (i == n-1):
          for u in range(0,n):
            delta_w[u][w_index] = delta[u][j]
          w_index +=1
        else:
          for u in range(0,n):
            delta_l[u][l_index] = delta[u][j]
          l_index +=1


                   
    if(conf_eval > 0):
	results.append(opt_cost)
    return results



  def get_runtime_Rand_W_noHard(self, runtime, k):
    '''
     Return the local runtime of configurations found by the 
     random oracle (configurations are shuffled randomly, benchmark are chosen with the wisdom
     of losers)
     k defines the number of partitions for the k interesection problems.
    '''   
    n = len(runtime)
    m = len(runtime[0])
    results = []
    delta = []
    delta_l = []

    l_index = 0

#   shuffle the algorithms keys
    l_akeys = [self.akeys[i] for i in range(0, m)]
    random.shuffle(l_akeys)
    l_ikeys = [self.ikeys[i] for i in range(0, n)]
    random.shuffle(l_ikeys)
    	
    opt_cost = 0
    opt_conf = 0
    for i in range(0, n):
	opt_cost += runtime[l_keys[i]][l_akeys[0]]      
        delta[self.ikeys[i]][l_akeys[0]] = runtime[l_ikeys[i]][l_akeys[0]]      
    results.append(opt_cost)
    conf_eval = 0
    for j in range(1, m):
#       separate instances in hard and softs 
        l_ikeys = self.separate_wisdom(k, delta_l, n)
	cur_cost = 0       
	i = 0
        for u in range(0,n):
          delta[u][l_akeys[j]] = -1         
	while (cur_cost <= opt_cost and i < n):
	   cur_cost += runtime[l_ikeys[i]][l_akeys[j]]	        
           delta[l_ikeys[i]][l_akeys[j]] = runtime[l_ikeys[i]][l_akeys[j]]
	   i += 1	
           conf_eval += 1
           if (conf_eval ==  self.conf_per_slot):        
             results.append(opt_cost)
             conf_eval = 0

        if(cur_cost < opt_cost):
             opt_cost = cur_cost
   
        if (i < n-1):
          for u in range(0,n):
            delta_l[u][l_index] = delta[u][l_akeys[j]]
          l_index +=1

                   
    if(conf_eval > 0):
	results.append(opt_cost)
    return results

           
  def separate_hard_soft(self, delta_w, n):
    '''
     Based on the runtime in delta[akeys[0]..akeys[j]]
     separate the instances in hard and softs
    '''
    k_means = KMeans(n_clusters=2)
    k_means.fit(delta_w)
    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    norm1  = numpy.linalg.norm(centroids[0])
    norm2  = numpy.linalg.norm(centroids[1])

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
   
    random.shuffle(hard)
    keys = [e for e in hard]
    random.shuffle(soft);
    for e in soft:
      keys.append(e)
   

    return keys
   


  def separate_hard_soft_sort(self, delta_w, n):
    '''
     Based on the runtime in delta[akeys[0]..akeys[j]]
     separate the instances in hard and softs
    '''
    k_means = KMeans(n_clusters=2)
    k_means.fit(delta_w)
    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    norm1  = numpy.linalg.norm(centroids[0])
    norm2  = numpy.linalg.norm(centroids[1])

    hard_index = 1
    if norm1 > norm2:
      hard_index = 0
   
    hard = []
    soft = []
  
    for i in range(0,n):
      w = sum(delta_w[i])
      if labels[i] == hard_index:
        hard.append([i,w])
      else:
        soft.append([i,w])
   
    hard = sorted(hard, key = lambda e: e[1])
    keys = [e[0] for e in hard]
    soft = sorted(soft, key = lambda e: e[1]);
    for e in soft:
      keys.append(e[0])
   

    return keys

  def separate_wisdom_hard_soft(self, k, delta_w, delta_l, n):
    '''
     Based on the runtime in delta[akeys[0]..akeys[j]]
     separate the instances in hard and softs but include the wisdom of the loser
    '''
    hard_soft = self.separate_hard_soft(delta_w, n)
    set_delta_l = [Set([]) for i in range(0,n)]
    for i in range(0,n):

      for j in range(0, len(delta_l[i])):
        if(delta_l[i][j] != -1):
          set_delta_l[i].add(delta_l[i,j])
    wisdom_loser = self.k_intersection(k, set_delta_l, n)
    random.shuffle(wisdom_loser)
    keys = [e for e in wisdom_loser]
    for e in hard_soft : 
      if not (e in wisdom_loser) :
        keys.append(e)
    return keys
	

  def separate_wisdom(k, delta_l, n):
    '''
     Based on the runtime in delta[akeys[0]..akeys[j]]
     separate the instances with the wisdom of the loser
    '''
    randkeys = []
    for i in range(0,n):
      randkeys[i] = i
    if(len(delta_l) == 0):
      random.shuffle(randkeys)
      return randkeys
    
    set_delta_l = []
    for i in range(0,n):
      set_delta_l[i] = Set([])
      for j in range(0, len(delta_l[i])):
        if(delta_l[i][j] != -1):
          set_delta_l[i].add(delta_l[i][j])
    random.shuffle(randkeys)
    wisdom_loser = self.k_intersection(k, set_delta_l, n)
    random.shuffle(wisdom_loser)
    keys = [e for e in wisdom_loser]
    for e in randkeys : 
      if not(e in wisdom_loser): 
        keys.append(e)
    return keys


  def k_intersection(k, set_delta_l, n):
    '''
      results of the k_intersection algorithm on delta_l
    '''
    indices = []
    card  = []
    for i in range(0, n):
      price[i] = len(set_delta_l[i])
    opt = 0
    for i in range(1,n):
      if (price[i] > price[opt]):
        opt = i
    indices.append(opt)
    inter = set_delta_l[opt].copy()
    set_delta_l[opt] = Set([])
#  select the remaining k sets
    for j in range(1,k):
      opt = 0
      while (len(set_delta_l[opt]) == 0):
        opt +=1
      tmp = inter.copy()
      price[opt] = len(tmp.intersection_update(set_delta_l[opt]))
      for i in range(1,n):
        if (len(set_delta_l[i]) > 0):
          tmp = inter.copy()
          price[i] = len(tmp.intersection_update(set_delta_l[i]))       
        else:
          price[i] = 0
        if (price[i] > price[opt]):
          opt = i
      if (price[opt] > 0):
        indices.append(opt)
      inter = inter.intersection_update(set_delta_l[opt])
      set_delta_l[opt] = Set([])

    return indices



  def integrate_regret(self, local_results_dist, regret_period):
    '''
     Return the regret we have from 0 to regret_period,
     assuming that the results are the oracle answers follow the  
     distribution described in  local_results_dist
    '''
    opt = min(local_results_dist) 
    results = []
    regret_t = local_results_dist[0]-opt
    dist_last_index = len(local_results_dist)-1
    results.append(regret_t)
    d = 1
    l_opt = 0
    for i in range(1, regret_period):
       if(d == self.request_time_slot):
          l_opt +=1
          d = 1
       else:
          d +=1
       regret_t += local_results_dist[min(l_opt,dist_last_index)]-opt
       results.append(regret_t)
    return results

  def run_method(self, method, regret_period, k=2):
   '''
     compute the regret vector depending on the method
   '''
   results = []
   local_dist = []
   if (method == 'lexnoshuf'):
     local_dist = self.get_runtime_Lex(self.runtime)
   elif (method == 'lex'):
     local_dist = self.get_runtime_Lex(self.runtime)
   elif (method == 'rand'):
     local_dist = self.get_runtime_Rand_HardSoft(self.runtime)
   elif (method == 'randw'):
     local_dist = self.get_runtime_Rand_W(self.runtime, k)
   elif (method == 'randwnoh'):
     local_dist = self.get_runtime_Rand_W_noHard(self.runtime)
   elif (method == 'randsort'):
     local_dist = self.get_runtime_Rand_HardSoft_sort(self.runtime)
   else:
     raise ValueError('the argument'+ method +' is unknown')
   results = self.integrate_regret(local_dist, regret_period)
   return results
