# -*- coding: utf-8 -*-
"""
Created on Mon Aug 7 15:39:19 2023

@author: Luis Sanchez Marquez, based on  Grey Wolf Optimizer (GWO) source codes version 1.0  for Matlab (Seyedali Mirjalili )

"""
import os
import numpy as np
import matplotlib.pyplot as plt

os.system('cls')

def Eggholder(x):
    return (-(x[1]+47)*(np.sin(np.sqrt(np.abs((x[0]/2)+(x[1]+47)))))) - (x[0]*np.sin(np.sqrt(np.abs(x[0]-(x[1]+47)))))

Max_corrida = 5

# Parametros iniciales
SearchAgents_no = 50 # Number of search agents
Max_iter = 200 # Maximum numbef of iterations
dim = 2

#LimInf = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
LimInf = np.array([-512, -450])#, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
LimSup = np.array([512, 450])#, 2.3, 3.1, 2.3, 3.1, 2.3, 3.1, 2.3, 3.1])

# LimInf = 0.2
# LimSup= 2.3

Table_Convergence_curve = np.zeros((Max_iter, Max_corrida))
Table_opt = np.zeros((Max_corrida, dim + 2))

corrida = 0

for corrida in range(0, Max_corrida,1): #1:Max_corrida
    #============================= Initalización =================================#
    # Empty particle
    Sparticula = {'Position': None, 'Cost': None}

    # Initialize Best Solution Ever Found
    Alpha = {'Position': None, 'Cost': np.inf} #np.inf para darle el valor más grande #change this to -inf for maximization problems
    Beta  = {'Position': None, 'Cost': np.inf} #np.inf para darle el valor más grande
    Delta = {'Position': None, 'Cost': np.inf} #np.inf para darle el valor más grande
         
    # Create Initial Population
    particula = []
    Boundary_no = np.size(LimSup) #numnber of boundaries
    if Boundary_no > 1: # If each variable has a different LimInf and LimSup
        particula_temp  = np.zeros((dim,SearchAgents_no))
    
    k = 0
    while (k < SearchAgents_no):
        
        #Initialize the positions of search agents
       
        # If the boundaries of all variables are equal and user enter a signle
        # number for both LimSup and LimInf
        if Boundary_no == 1:
            particula.append(Sparticula.copy()) #.append es para agregar un elemento más al final de la lista
            particula[k]['Position'] = np.random.uniform(LimInf, LimSup, dim) #random entre LimInf y LimSup con 10 características
            #particula[k]['Cost'] = Eggholder( particula[k]['Position'] ) # Objective Value
            
        # If each variable has a different LimInf and LimSup
        if Boundary_no > 1:
            particula.append(Sparticula.copy()) #.append es para agregar un elemento más al final de la lista
            #particula[k]['Position'] = np.random.uniform(0.2, 2.3, dim)
            for i in range(0, dim, 1): # 1:dim
                LimSup_i = LimSup[i]
                LimInf_i = [i]
                particula_temp[i,k] = np.random.uniform(LimInf_i, LimSup_i) #rand(SearchAgents_no,1).*(LimSup_i-LimInf_i)+LimInf_i;
                
            particula[k]['Position'] = particula_temp[:,k]
            
        particula[k]['Cost'] = Eggholder( particula[k]['Position'] ) # Objective Value
        
        # Update Alpha, Beta, and Delta    
        if (particula[k]['Cost'] < Alpha['Cost']):
            Alpha ['Position'] =  particula[k]['Position'] # Update  Alpha
            Alpha ['Cost'] =  particula[k]['Cost']
            
        if  particula[k]['Cost'] > Alpha ['Cost'] and particula[k]['Cost'] < Beta['Cost']:
            Beta ['Position'] =  particula[k]['Position'] # Update  Beta
            Beta ['Cost'] =  particula[k]['Cost']
            
        if particula[k]['Cost'] > Alpha ['Cost'] and particula[k]['Cost'] > Beta ['Cost'] and particula[k]['Cost'] < Delta['Cost']:
            Delta ['Position'] =  particula[k]['Position'] # Update Delta
            Delta ['Cost'] =  particula[k]['Cost']
            
        k = k + 1
        
    l = 0 # Loop counter
    #temporal = particula[0]['Position'] < LimSup
        
    # Main loop
    while l<Max_iter:
        
        ki = 0
        while (ki < SearchAgents_no):#in range (0, SearchAgents_no, 1):# 1:size(Positions,1)
            
            # Return back the search agents that go beyond the boundaries of the search space
            # Aplicar Límites
            
            Bandera4LimSup = particula[ki]['Position'] > LimSup
            Bandera4LimInf = particula[ki]['Position'] < LimInf
            particula[ki]['Position'] = (particula[ki]['Position']*(np.logical_not(Bandera4LimSup + Bandera4LimInf))) + LimSup*Bandera4LimSup + LimInf*Bandera4LimInf
            
            # Calculate objective function for each search agent
            particula[ki]['Cost'] = Eggholder( particula[ki]['Position'] ) # Objective Value
           # fitness=SOTDOA(Positions(i,:));
            
            # Update Alpha, Beta, and Delta    
            if (particula[ki]['Cost'] < Alpha['Cost']):
                Alpha ['Position'] =  particula[ki]['Position'] # Update  Alpha
                Alpha ['Cost'] =  particula[ki]['Cost']
                
            if  particula[i]['Cost'] > Alpha ['Cost'] and particula[ki]['Cost'] < Beta['Cost']:
                Beta ['Position'] =  particula[ki]['Position'] # Update  Beta
                Beta ['Cost'] =  particula[ki]['Cost']
                
            if particula[i]['Cost'] > Alpha ['Cost'] and particula[ki]['Cost'] > Beta ['Cost'] and particula[ki]['Cost'] < Delta['Cost']:
                Delta ['Position'] =  particula[ki]['Position'] # Update Delta
                Delta ['Cost'] =  particula[ki]['Cost']
            ki = ki + 1
        a = 2 - l*(2/Max_iter) # a decreases linearly fron 2 to 0
        # Update the Position of search agents including omegas
        for i in range(0, SearchAgents_no, 1): # 1:size(Positions,1)
            r1 = np.random.rand() # r1 is a random number in [0,1]
            r2 = np.random.rand() # r2 is a random number in [0,1]
                
            A1 = 2*a*r1 - a #Equation (3.3)
            C1 = 2*r2 # Equation (3.4)                
                
            D_alpha = np.abs(C1*Alpha['Position'] - particula[i]['Position'])  # Equation (3.5)-part 1 
            X1 = Alpha['Position'] - A1*D_alpha # Equation (3.6)-part 1
                
            r1 = np.random.rand()
            r2 = np.random.rand()
            
            A2 = 2*a*r1-a # Equation (3.3)
            C2 = 2*r2 # Equation (3.4)
                
            D_beta = np.abs(C2*Beta['Position'] - particula[i]['Position']) # Equation (3.5)-part 2
            X2 = Beta['Position'] - A2*D_beta # Equation (3.6)-part 2
                
            r1 = np.random.rand()
            r2 = np.random.rand()
                
            A3 = 2*a*r1 - a # Equation (3.3)
            C3 = 2*r2 # Equation (3.4)
                
            D_delta = np.abs(C3*Delta['Position'] - particula[i]['Position']) # Equation (3.5)-part 3
            X3 = Delta['Position'] - A3*D_delta # Equation (3.5)-part 3
                
            particula[i]['Position'] = (X1+X2+X3)/3 # Equation (3.7)
        
        Table_Convergence_curve[l,corrida] = Alpha['Cost'] 
        
        alpha_temp = np.zeros((1,np.size(Alpha['Position'])))

        for temp in range(0,np.size(Alpha['Position']),1):
            alpha_temp[0,temp] = Alpha['Position'][temp]
        Fila = np.concatenate((corrida+1,alpha_temp,Alpha['Cost']), axis=None)
        
        Table_opt[corrida] = (Fila)  
        
        l = l+1
    
    corrida = corrida +1
iteracion = np.arange(0, Max_iter, 1)
fig, f1 = plt.subplots()
f1.plot(iteracion, Table_Convergence_curve)
plt.title('Convergence Curve')
plt.xlabel('Iteration')
plt.ylabel('Cost')

        