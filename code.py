import random
import math
import numpy as np
import copy
import sys
import networkx as nx
import pandas as pd
from datetime import datetime
from itertools import compress

def main(mutatorsNo,meetings):      #this is the main function
    

    vowels={"a":((679,80),(1051,89)),"aa":((795,95),(1301,113))}   #original frequencies of these vowels, Formant1 and Fromant2 with sd
    networksize=20
    nattach=10
    mabove=90
    mbelow=10
    pabove=0
    
    datapath="/Users/joebarlow/Desktop/Project/"
    fn="Barabasi-N_"+str(networksize)+"-p_"+str(nattach)\
        +"-above_"+str(mabove)+"-below_"+str(mbelow)\
        +"-pabove_"+str(pabove)+".csv"

    for i in range(1):                #running the simulation 10 times
        run(vowels, mutatorsNo, meetings, datapath, fn, networksize, nattach, mabove, mbelow, pabove)


def run(vowels,mutatorsNo,meetings,datapath,fn,networksize,nattach,mabove,mbelow,pabove):
    
    vowel1=vowels['a']
    vowel2=vowels['aa']
    simulation=datetime.now().strftime("%d%m%Y%H%M%S")
    random.seed(a=int(simulation))
    
    community=[]
    population=nx.barabasi_albert_graph(networksize, nattach, seed=int(simulation))  #creating the population of 1000people
    
    
    for i in range(networksize): #going through the whole population
    
        size=population.degree(i)   #tell the degree of node i aka how many freinds the agent has
        neighbors=set()            
        
        for node,edge in population.edges(i):

            neighbors.add(edge)              #listing all the neighbours of an agent
            
        speaker=Speaker(vowels, size,neighbors)    #each agent has 3 things, the way they soeak, number of neighbours, list of neighbours
        community.append(speaker)     #adding the speaker to the community
    
    networkLength=[]
    medInterlocPopularity=[]
    interlocPopularity={}

    for speaker in community:       #going through each agent in community

        networkLength.append(speaker.size)  #adding the speaker's degree to a list, contains everyone's degree
        interlocPopularityTemp=[]

        for interloc in speaker.network:   #cycle through neigboirs of speaker in big for loop

            interlocPopularityTemp.append(community[interloc].size)       #list of speaker's neighbors' degree
            
        medInterlocSize=np.percentile(interlocPopularityTemp,50)     #returns number and half of the people are below this
        medInterlocPopularity.append(medInterlocSize)                   #make a list of these numbers for every speaker
        interlocPopularity[speaker]=medInterlocSize                      #same as above but vector***???

    percentile10=np.percentile(networkLength,10)      #value below which 10% of the degrees are
    median=np.percentile(networkLength,50)            #median of all degrees
    percentile90=np.percentile(networkLength,90)    #value above which only the biggest 10% is 
    
    mutators=mutate (community, vowel1, vowel2, mutatorsNo, medInterlocPopularity, mbelow, mabove, pabove)
    mutatorsIndice=set()    #list of all the indexes of each mutator, so we know who they are
   
    for speaker in mutators:

        mutatorsIndice.add(community.index(speaker))  #index of each speaker who are mutators
        
        
    speakersIndice=[x for x in list(range(0,networksize)) if x not in mutatorsIndice]
         
    interactionSpace(community, mutators, vowels, vowel1, vowel2, meetings,mutatorsIndice,speakersIndice,log,percentile10,median,percentile90,simulation, mutatorsNo, datapath) 
    


class Speaker(object):    #we are creating a class and put objects in it

    def __init__(self, vowels, size, neighbors):

        spVowels=generateSpeakerDutch(vowels)

        self.size=size
        self.vowels=spVowels 
        self.vowelsBeg=copy.deepcopy(self.vowels)    #copied vowels, we can make changes in this
        self.network=neighbors

        

    def prod(self):

        speech={}

        for vowel in self.vowels:

            f1=random.normalvariate(self.vowels[vowel][0][0], self.vowels[vowel][0][1]) 
            f2=random.normalvariate(self.vowels[vowel][1][0], self.vowels[vowel][1][1])

            speech[vowel]=(f1, f2)

        return speech

 

    def prodMutation(self, vowel1, vowel2):

        speech={}

        self.vowels['a']=self.vowels['aa']

        for vowel in self.vowels:

            f1=random.normalvariate(self.vowels[vowel][0][0], self.vowels[vowel][0][1])   #frequency + sd
            f2=random.normalvariate(self.vowels[vowel][1][0], self.vowels[vowel][1][1])   #frequency + sd
            speech[vowel]=(f1, f2)

        return speech

           

    def updateDistrib(self,speech):

        for vowel in self.vowels:

            updatedf1=0.1/(self.size)*speech[vowel][0]+(10.0*self.size-1)/(10.0*(self.size))*self.vowels[vowel][0][0]  #after each interaction the agent updates his frequency of F1 to be a bit similar to what he heard
            updatedf2=0.1/(self.size)*speech[vowel][1]+(10.0*self.size-1)/(10.0*self.size)*self.vowels[vowel][1][0]   #after each interaction the agent updates his frequency of F2 to be a bit similar to what he heard       
            updatedf1sd=math.sqrt((10.0*self.size-2)/(10.0*self.size-1)*self.vowels[vowel][0][1]**2+0.1/self.size*(speech[vowel][0]-self.vowels[vowel][0][0])**2)  #after each interaction the agent updates his standard deviation of frequency of F1 to be a bit similar to what he heard
            updatedf2sd=math.sqrt((10.0*self.size-2)/(10.0*self.size-1)*self.vowels[vowel][1][1]**2+0.1/self.size*(speech[vowel][1]-self.vowels[vowel][1][0])**2)   #after each interaction the agent updates his standard deviation of frequency of F1 to be a bit similar to what he heard
            
            self.vowels[vowel]=((updatedf1,updatedf1sd),(updatedf2, updatedf2sd))              



    def updateDistribMut(self,speech, vowel1, vowel2):   #this looks exactly the same as above but different????

        for vowel in self.vowels:

            if vowel==vowel1: continue

            else:

                updatedf1=0.1/self.size*speech[vowel][0]+(10.0*self.size-1)/(10.0*self.size)*self.vowels[vowel][0][0]
                updatedf2=0.1/self.size*speech[vowel][1]+(10.0*self.size-1)/(10.0*self.size)*self.vowels[vowel][1][0]               
                updatedf1sd=math.sqrt((10.0*self.size-2)/(10.0*self.size-1)*self.vowels[vowel][0][1]**2+0.1/self.size*(speech[vowel][0]-self.vowels[vowel][0][0])**2)
                updatedf2sd=math.sqrt((10.0*self.size-2)/(10.0*self.size-1)*self.vowels[vowel][1][1]**2+0.1/self.size*(speech[vowel][1]-self.vowels[vowel][1][0])**2)
                
                self.vowels[vowel]=((updatedf1,updatedf1sd),(updatedf2, updatedf2sd))

        self.vowels['a']=self.vowels['aa']          



def generateSpeakerDutch(vowels):

    speakerVowels={}

    for vowelName, vowelFormants in vowels.items():

        formantsTemp=[]

        for index, (formantMu, formantSigma) in enumerate(vowelFormants):

            mean=random.normalvariate(formantMu, formantSigma)      
            formantsTemp.append((mean,mean*0.02))

        speakerVowels[vowelName]=formantsTemp

    return speakerVowels



def distanceCalcMerge(speakerVowels, beginVowels):   #calculating how close they get to each other

    formants1=[speakerVowels['a'][0][0],speakerVowels['a'][1][0]]
    formants2=[speakerVowels['aa'][0][0],speakerVowels['aa'][1][0]]

    beginF1=[beginVowels['a'][0][0],beginVowels['a'][1][0]]
    beginF2=[beginVowels['aa'][0][0],beginVowels['aa'][1][0]]

    scoreNow=tokenDistance(formants1, formants2)
    scoreBeg=tokenDistance(beginF1, beginF2)

    scoreDiff=scoreNow-scoreBeg

    return scoreDiff

        

def tokenDistance (token1, token2):

    score=math.sqrt((token1[0]-token2[0])**2

                        +(token1[1]-token2[1])**2)

    return score



def mutate (community, vowel1, vowel2, mutatorsNo, medInterlocPopularity, mbelow, mabove, pabove):
    
    interlocabove=np.percentile(medInterlocPopularity,mabove)   # value above which the medium of friends' degree is higher than mabove, upper percentile
    interlocbelow=np.percentile(medInterlocPopularity,mbelow)   # value below which the medium of friends' degree is lower than mbelow, lower percintile
    
    nabove=int(mutatorsNo*pabove) #factors in pabove for upper percentile
    nbelow=mutatorsNo-nabove      # again factors in pabove for bottom percentile
    
    mutators_above= list(compress(community,medInterlocPopularity >= np.float32(interlocabove))) # for all speakers in the community compress to all those above 90th percentile
    mutators_below= list(compress(community,medInterlocPopularity <= np.float32(interlocbelow))) 
    
    if nabove>len(mutators_above) or nbelow > len(mutators_below):
        sys.exit('Not enough mutators in the required percentile')
    else:
        mutators_above_li=random.sample(mutators_above,nabove)
        mutators_below_li=random.sample(mutators_below,nbelow)
                                        
    mutators = mutators_above_li + mutators_below_li                                  

    for mutator in mutators: 
        mutator.vowels['a']=mutator.vowels['aa']   #we set their two vowels to be equal

    return mutators    
    

def interactionSpace (community, mutators, vowels, vowel1, vowel2, meetings,mutatorsIndice,speakersIndice,log,percentile10,median,percentile90,simulation, mutatorsNo, datapath):

    for time in range(meetings):

        bannedInterloc=set()

        for index, speaker in enumerate(community):

            if index in bannedInterloc: continue        #banned so move on

            if speaker.network.intersection(bannedInterloc)==speaker.network:        
                interloc=random.choice(list(speaker.network))                        #soemthing about the neighbors

            else: 
                updatedNetwork=speaker.network.difference(bannedInterloc)
                interloc=random.choice(list(updatedNetwork)) 

            bannedInterloc.add(interloc)
            bannedInterloc.add(index)

            if speaker not in mutators:
               tokenSp=speaker.prod()
            else:
                tokenSp=speaker.prodMutation(vowel1, vowel2)

            if community[interloc] not in mutators:
                tokenInterloc=community[interloc].prod()
            else:
                tokenInterloc=community[interloc].prodMutation(vowel1, vowel2)

            if speaker not in mutators:
                speaker.updateDistrib(tokenInterloc)
            else:
                speaker.updateDistribMut(tokenInterloc,vowel1, vowel2)

            if community[interloc] not in mutators:
                community[interloc].updateDistrib(tokenSp)
            else:
                community[interloc].updateDistribMut(tokenSp, vowel1, vowel2)
        
        if (time+1)%10==0:
        
            vowelChange(vowels,community,mutators,vowel1,mutatorsIndice,speakersIndice,log,time,percentile10,median,percentile90,simulation, mutatorsNo, datapath)
             
def vowelChange(vowels, community, mutators,vowel1,mutatorsIndice,speakersIndice,log,time,percentile10,median,percentile90,simulation, mutatorsNo, datapath):

    currentf1a=[]
    currentf2a=[]
    currentf1aa=[]
    currentf2aa=[]

    number=0
    
    for speaker in community:

        if speaker in mutators: continue
        
        if (abs(speaker.vowels['a'][0][0]-speaker.vowels['aa'][0][0])<4 and abs(speaker.vowels['a'][1][0]-speaker.vowels['aa'][1][0])<4): #no of speakers within limit
            number=number+1
        
        #change=distanceCalcMerge(speaker.vowels,speaker.vowelsBeg)
        currentf1a.append(speaker.vowels['a'][0][0])
        currentf2a.append(speaker.vowels['a'][1][0])
        currentf1aa.append(speaker.vowels['aa'][0][0])
        currentf2aa.append(speaker.vowels['aa'][1][0])
        
    #df=pd.DataFrame({'SimID': int(simulation),
                      #'speakersIndice': speakersIndice,
                      #'time': time,
                      #'f1a': currentf1a,
                      #'f2a': currentf2a,
                      #'f1aa': currentf1aa,
                      #'f2aa': currentf2aa,
                      #'mean': np.sqrt((np.mean(currentf1a)-np.mean(currentf1aa))**2+(np.mean(currentf2a)-np.mean(currentf2aa))**2)})
     
    #df.to_csv(datapath+'Barabasi4.csv',index=False, mode='a')
    
                            

                                                                                                                                                                              

def log(*info):

    sys.stdout.write(" ".join(str(x) for x in info)+'\n')
                                                                                                                                                                            
                                                                                                                                                                  
if __name__=='__main__':

    meetings=100

    mutatorsNo=2
    
    main(mutatorsNo,meetings)
