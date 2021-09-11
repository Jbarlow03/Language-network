# Language-network
This code simulates a scenario where language change occurs based on social network size and records the results. 

A population of an adjustable number (networksize) of agents (speakers) is created using an Albert Barabasi algorithm from the Python NetworkX package. 
This algorithm creates networks with a free-scale structure. In such networks the distribution of agents’ network size follows a power law. 

All agents in the community know two Dutch vowels, /ɑ/ and /a:/. Each agent is ascribed mean formant frequencies for these two vowels according to published data about the formant frequency distribution of these vowels in the population. 

In the population there is an adjustable number of mutators (mutatorsNo) who will be either connected to agents whose average network size is in the top 
10% of the population (Neighbors with Large social networks) or to agents whose average network size is in the bottom 10% of the population (Neighbors with Small Social Network). (pabove = 1 for large networks and pabove= 0 for small networks). The social network size is determined by the nodes degree.
For these mutators, the /a:/ category was defined to be a copy of their /ɑ/ category. 

Then there is an adjustable number of interactions (Meetings). In each round, each agent in the population will interact with at least one randomly 
sampled member of their network. Some agents interact with more than one other agent if more than one member of their network sampled them as an interlocutor. 
During each interaction, each agent produces one token of each vowel by sampling a token around their mean formant frequencies. Each interlocutor then updates 
their vowel categories to include the token they heard from their interlocutor. The new token is integrated with their prior vowel values by ascribing it 
a weight of 0.1/ (agent’s network size), and ascribing their previous knowledge the weight (1–0.1/(agent’s network size)) (updateDistrib). The weight ascribed to the new token implements the inverse relationship between network size and the weight ascribed to each speaker in the network.

What i want to do with the code input a way of assigning weights. I believe this can be done by assigning weights to the nodes (speakers) and edges (their connections) and then calling back to these in the UpdateDistrib function, i.e., weight/(agents network size) and (1-weight/(agent's network size)).
