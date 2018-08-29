 # -*- coding:utf-8 -*-   
# Author: zzy  
  
import numpy as np  
import random  
import timeit  
import scipy.io as sio  
from scipy.optimize import leastsq
import matplotlib.pyplot as plt 

min_t = 0
max_t = 1
popsize = 100
max_iter=5

#读入数据
def getmatdata(filepath):
    matfile = sio.loadmat(filepath)
    #print(matfile.keys())
    data = matfile['z']
    #print(type(data))
    #print(data.shape)
    return data
  
# 随机生成初始编码种群  
def getIntialPopulation(solutionlength, populationSize):  
    # 随机化初始种群为0  
    chromosomes = np.zeros((populationSize, int(solutionlength)), dtype=np.uint8)  
    for i in range(populationSize): 
        #此处使用的是均匀分布，彼此间有干扰，可考虑对每个数单独赋值，互不干扰，保证随机性 
        chromosomes[i, :] = np.random.randint(min_t, max_t, int(solutionlength))  
    # print('chromosomes shape:', chromosomes.shape)  
    return chromosomes  
  
#随机初始化权重系数
def getInitalWeightParas(popsize,itemsize):
    weightparas_pop = np.random.randint(1,5,(popsize,itemsize))

    return weightparas_pop

def leastsqt(population,weightparas_pop,data):
	#x0,x1,.....x7,y
	xishu_pop = np.zeros(weightparas_pop.shape)
	data_x = data[:,0:8]

	y = data[:,8]

	#print('this is data:',type(data_x),data_x.shape)
	#print(type(y),y.shape)
	for i in range(len(population)):
		solutioncode = population[i]
		weightparas_sol = weightparas_pop[i]
		xishu = leastsq(errorfunc, weightparas_sol, args=(solutioncode,data_x,y))
		#print('this is xishu of solution : ',i,'  :')
		nd_xishu = np.reshape(xishu[0],(1,len(xishu[0])))		
		xishu_pop[i,:]=nd_xishu[:]
	return xishu_pop
	
def solutionfunc(weightparas,solutioncode,data_x):
	calcu_y = np.zeros((len(data_x),))
	#print('solutionfunc:',data_x.shape)
	for k in range(len(data_x)):
		sample_k = data_x[k]
		#print('this is sample_k:',sample_k.shape,sample_k)
		cal_y_k = 0
		for i in range(30):
			funcitem = weightparas[i]
			xindex = 0
			for j in range((i+1)*8-8,(i+1)*8):
				funcitem = funcitem *pow(sample_k[xindex],solutioncode[j])
				xindex +=1
			cal_y_k = cal_y_k + funcitem
		cal_y_k = cal_y_k + weightparas[30]
		#print('target :',cal_y_k)
		calcu_y[k] = cal_y_k
	#print('this is calcu_y:',calcu_y.shape)
	return calcu_y


def solutionfunc_old(weightparas,solutioncode,x):
	targetfunc = 0
	#print('solutionfunc:',x.shape)
	for i in range(len(weightparas)):
		funcitem = weightparas[i]
		xindex = 0
		for j in range((i+1)*8-8,(i+1)*8):
			#print('this is x index:',x[xindex].shape)
			funcitem = funcitem *pow(x[xindex],solutioncode[j])
			xindex +=1
		targetfunc = targetfunc + funcitem
	#print('target shape:',targetfunc.shape,targetfunc)
	return targetfunc

def get_xishu_pop(population,data):
	#x0,x1,.....x7,y
	xishu_pop = np.zeros((len(population),31))
	data_x = data[:,0:8]

	y = data[:,8]

	#print('this is data:',type(data_x),data_x.shape)
	#print(type(y),y.shape)
	for i in range(len(population)):
		solutioncode = population[i]
		xishu = leastsqt_new(solutioncode,data_x,y)
		#print('this is xishu of solution : ',i,'  :')
		#print(type(xishu),xishu.shape,xishu)
		nd_xishu = np.reshape(xishu,(1,len(xishu)))
		xishu_pop[i,:]=nd_xishu[:]
	return xishu_pop

def leastsqt_new(solutioncode,data_x,y):
	data_x_31items = np.zeros((len(data_x),31))
	# print('solutionfunc:',data_x.shape)
	for k in range(len(data_x)):
		sample_k = data_x[k]
		# print('this is sample_k:',sample_k.shape,sample_k)
		cal_y_k = 0
		for i in range(30):
			x2item_i = 1
			xindex = 0
			for j in range((i + 1) * 8 - 8, (i + 1) * 8):
				x2item_i = x2item_i * pow(sample_k[xindex], solutioncode[j])
				xindex+=1
			data_x_31items[k,i] = x2item_i
	data_x_31items[:,30]=1.0
	# print('this is calcu_y:',calcu_y.shape)
	xishu = (np.linalg.pinv(data_x_31items)).dot(y)
	return xishu
	pass

def errorfunc(weightparas,solutioncode,data_x,y):
	#print('errorfunc:')
	return solutionfunc(weightparas,solutioncode,data_x) - y


# 定义适应度函数  
def fitnessFunction():  
    return lambda x: 21.5 + x[0] * np.sin(4 * np.pi * x[0]) + x[1] * np.sin(20 * np.pi * x[1])  
    pass  

#得到个体的适应度值及每个个体被选择的概率
def getfitvalue(population,xishu_pop,data):
	fitness_pop = np.zeros((len(population),2))
	data_x = data[:,0:8]
	y = data[:,8]
	for i in range(len(population)):
		solutioncode = population[i]
		xishu_sol = xishu_pop[i]
		error_value = np.absolute(errorfunc(xishu_sol,solutioncode,data_x,y))
		error_value[:] = pow(error_value[:],2)
		#print('this is type of errorvalue:',type(error_value))
		list(error_value)
		error_value.sort()
        #求平均
		fitness_pop[i,0]=sum(error_value)/len(error_value)
	fitness_pop = np.float64(fitness_pop)
	fitnesssum = sum(fitness_pop[:,0])
	fitness_pop[:,1] = fitness_pop[:,0]/fitnesssum
	#fitness_pop[:,1] = np.cumsum(fitness_pop[:,1])
	return fitness_pop 
  
  
# 新种群选择  
def selectNewPopulation(chromosomes, fitness_pop):  
    m, n = chromosomes.shape  
    newpopulation = np.zeros((m, n), dtype=np.uint8)  
    # 随机产生M个概率值  
    randoms = np.random.rand(m)
    probability = fitness_pop[:,1]
    for i, randoma in enumerate(randoms):  
        logical = probability >= randoma  
        index = np.where(logical == 1)  
        # index是tuple,tuple中元素是ndarray,从满足一定随机概率的多个个体中再随机选择一个
        if len(index[0])>0:       
            j = int(np.random.randint(0,len(index[0]),1))
            newpopulation[i, :] = chromosomes[index[0][j], :]  
        else:
            j = int(np.random.randint(0,m,1))
            newpopulation[i, :] = chromosomes[j, :] 
    return newpopulation  
    pass
  
# 新种群交叉  
def crossover(population, Pc=0.8):  
    """ 
    :param population: 新种群 
    :param Pc: 交叉概率默认是0.8 
    :return: 交叉后得到的新种群 
    """  
    # 根据交叉概率计算需要进行交叉的个体个数  
    m, n = population.shape  
    numbers = np.uint8(m * Pc)  
    # 确保进行交叉的染色体个数是偶数个  
    if numbers % 2 != 0:  
        numbers += 1  
    # 交叉后得到的新种群  
    updatepopulation = np.zeros((m, n), dtype=np.uint8)  
    # 产生随机索引,随机选择numbers个个体 
    index = random.sample(range(m), numbers)  
    # 不进行交叉的染色体进行复制  
    for i in range(m):  
        if not index.__contains__(i):  
            updatepopulation[i, :] = population[i, :]  
    # crossover  
    while len(index) > 0:  
        a = index.pop()  
        b = index.pop()  
        # 随机产生一个交叉点  
        crossoverPoint = random.sample(range(1, n), 1)  
        crossoverPoint = crossoverPoint[0]  
        # one-single-point crossover  
        updatepopulation[a, 0:crossoverPoint] = population[a, 0:crossoverPoint]  
        updatepopulation[a, crossoverPoint:] = population[b, crossoverPoint:]  
        updatepopulation[b, 0:crossoverPoint] = population[b, 0:crossoverPoint]  
        updatepopulation[b, crossoverPoint:] = population[a, crossoverPoint:]  
    return updatepopulation  
    pass  
  
# 染色体变异  
def mutation(population, Pm=0.01):  
    """ 
    :param population: 经交叉后得到的种群 
    :param Pm: 变异概率默认是0.01 
    :return: 经变异操作后的新种群 
    """  
    updatepopulation = np.copy(population)  
    m, n = population.shape  
    # 计算需要变异的基因个数  
    gene_num = np.uint8(m * n * Pm)  
    # 将所有的基因按照序号进行10进制编码，则共有m*n个基因  
    # 随机抽取gene_num个基因进行基本位变异  
    mutationGeneIndex = random.sample(range(0, m * n), gene_num)  
    # 确定每个将要变异的基因在整个染色体中的基因座(即基因的具体位置)  
    for gene in mutationGeneIndex:  
        # 确定变异基因位于第几个染色体  
        chromosomeIndex = gene // n  
        # 确定变异基因位于当前染色体的第几个基因位  
        geneIndex = gene % n  
        # mutation  
        change = int(np.random.randint(min_t,max_t,1))
        while(updatepopulation[chromosomeIndex, geneIndex] ==change  ):
            change = int(np.random.randint(min_t,max_t,1))
        updatepopulation[chromosomeIndex, geneIndex] = change
    return updatepopulation  
    pass  
  
def getbestfrompop(population,fitness_pop,xishu_pop):
	minerrorindex = 0
	
	for i in range(len(fitness_pop)):
		if fitness_pop[minerrorindex,0]>fitness_pop[i,0]:
			minerrorindex = i
		#if fitness_pop[minerrorindex,0]==fitness_pop[i,0]:
	best_solu = population[minerrorindex,:]	
	best_xishu = xishu_pop[minerrorindex,:]
	best_fitness = fitness_pop[minerrorindex,0]
	solution = [best_fitness,best_solu,best_xishu]
	return solution
	pass

def getoptres(solutionlist):
	bestindex = 0
	for i in range(len(solutionlist)):
		if solutionlist[bestindex][0] > solutionlist[i][0]:
			bestindex = i
	return solutionlist[bestindex]
	pass

def main():  
    # 每次迭代得到的最优解  
    optimallist = []  
    
    filepath = './data/ConcreteCompressiveStrength.mat'

    itemsize = 30
    solength = 240 
    
    # 得到初始种群编码  
    chromosomes = getIntialPopulation(solength, popsize)  
    print('种群',type(chromosomes),chromosomes.shape,len(chromosomes))
    weightparas_pop = getInitalWeightParas(popsize,itemsize)
    print('种群权重系数矩阵：',weightparas_pop.shape)
    data = getmatdata(filepath)
    print('测试数据：',data.shape)
    print('开始计算初始种群。。。。')
    # 利用最小二乘法计算种群系数矩阵  
    xishu_pop = get_xishu_pop(chromosomes,data)
    print('种群的系数矩阵：',len(xishu_pop)) 
    # 得到个体适应度值和个体的概率  
    fitness_pop = getfitvalue(chromosomes,xishu_pop,data)
    # 搜索每次迭代的最优解，以及最优解对应的目标函数的取值  
    optimalsolu = getbestfrompop(chromosomes,fitness_pop,xishu_pop)
    optimallist.append(optimalsolu)


    for iteration in range(max_iter):   
        print('开始循环迭代，第 ',iteration,' 次循环。。。。')
        # 选择新的种群  
        chromosomes = selectNewPopulation(chromosomes,fitness_pop)  
        # 进行交叉操作  
        chromosomes = crossover(chromosomes)  
        # mutation  
        chromosomes = mutation(chromosomes)  
        # 利用最小二乘法计算种群系数矩阵  
        xishu_pop = get_xishu_pop(chromosomes,data)
        
        # 得到个体适应度值和个体的概率  
        fitness_pop = getfitvalue(chromosomes,xishu_pop,data) 
        # 搜索每次迭代的最优解，以及最优解对应的目标函数的取值  
        optimalsolunew = getbestfrompop(chromosomes,fitness_pop,xishu_pop)
        if optimalsolunew[0] <= optimalsolu[0]:
            optimalsolu = optimalsolunew
            optimallist.append(optimalsolu)
        else:
            optimallist.append(optimalsolu)
     
    #对于optimallist中的点做可视化展示
    visual_data = np.zeros((len(optimallist),2))
    for i in range(len(optimallist)):
        visual_data[i,0] = i+1
        visual_data[i,1] = optimallist[i][0]
    plt.plot(visual_data[:,0],visual_data[:,1])
    plt.show()
    # 搜索最优解    
    optimalSolution = getoptres(optimallist) 
    return optimalSolution  


optimalfunc = main()  
with open('./data/result.txt','a') as file:
	file.write('\n\n\n\n最优目标函数的编码:')
	file.write('\n'+str(optimalfunc[1])  )
	file.write('\n最优目标函数的31项的系数,最后一项是常数作为偏移量:')
	file.write('\n'+str(optimalfunc[2]) )
	file.write('\n最优目标函数的适应度:')
	file.write('\n'+str(optimalfunc[0]) )

print('最优目标函数的编码:', optimalfunc[1])  
print('最优目标函数的30项的系数:', optimalfunc[2]) 
print('最优目标函数的适应度:', optimalfunc[0]) 






