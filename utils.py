import json
from collections import Counter
import numpy as np
import operator
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
import networkx as nx
import base64
from collections import defaultdict
import sys,os
import math
import random
import operator
import csv
import matplotlib.pylab as pyl
import itertools
import scipy as sp
from scipy import stats
from scipy import optimize
from scipy.integrate import quad


def load_jsons(data_folder='data/', tournaments=['Italy','England','Germany', 
                                                 'France','Spain', 
                                               'European_Championship','World_Cup']):
    """
    Load the json files with the matches, events, players and competitions
    
    Parameters
    ----------
    data_folder : str, optional
        the path to the folder where json files are stored. Default: 'data/'
        
    tournaments : list, optional
        the list of tournaments to load. 
        
    Returns
    -------
    tuple
        a tuple of four dictionaries, containing matches, events, players and competitions
        
    """
    # loading the matches and events data
    matches, events = {}, {}
    for tournament in tournaments:
        with open('./data/events/events_%s.json' %tournament) as json_data:
            events[tournament] = json.load(json_data)
        with open('./data/matches/matches_%s.json' %tournament) as json_data:
            matches[tournament] = json.load(json_data)

    # loading the players data
    players = {}
    with open('./data/players.json') as json_data:
        players = json.load(json_data)

    # loading the competitions data
    competitions={}
    with open('./data/competitions.json') as json_data:
        competitions = json.load(json_data)
        
    return matches, events, players, competitions


        
def get_datadriven_weight(position, normalize=True):
    """
    Get the probability of scoring a goal given the position of the field where 
    the event is generated.
    
    Parameters
    ----------
    position: tuple
        the x,y coordinates of the event
        
    normalize: boolean, optional
        if True normalize the weights
        
    
    """
    weights=np.array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   2.00000000e+00,   2.00000000e+00,
          0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   8.00000000e+00,   1.10000000e+01,
          1.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          4.00000000e+00,   4.00000000e+01,   1.28000000e+02,
          7.00000000e+01],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          9.00000000e+00,   1.01000000e+02,   4.95000000e+02,
          4.83000000e+02],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          6.00000000e+00,   9.80000000e+01,   5.60000000e+02,
          1.12000000e+03],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          8.00000000e+00,   9.30000000e+01,   5.51000000e+02,
          7.82000000e+02],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          3.00000000e+00,   6.70000000e+01,   3.00000000e+02,
          2.30000000e+02],
       [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.00000000e+00,   1.30000000e+01,   3.20000000e+01,
          1.10000000e+01],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.00000000e+00,   2.00000000e+00,   2.00000000e+00,
          2.00000000e+00],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
          0.00000000e+00]])
    
    x, y = position
    
    if x==100.0:
        x=99.9
    
    if y==100.0:
        y=99.9
    
    w = weights[int(y/10)][int(x/10)]
    
    if normalize: # normalize the weights
        w = w/np.sum(weights)
    return w
    
def get_weight(position):
    """
    Get the probability of scoring a goal given the position of the field where 
    the event is generated.
    
    Parameters
    ----------
    position: tuple
        the x,y coordinates of the event
    """
    x, y = position
    
    # 0.01
    if x >= 65 and x <= 75:
        return 0.01
    
    # 0.5
    if (x > 75 and x <= 85) and (y >= 15 and y <= 85):
        return 0.5
    if x > 85 and (y >= 15 and y <= 25) or (y >= 75 and y <= 85):
        return 0.5
    
    # 0.02
    if x > 75 and (y <= 15 or y >= 85):
        return 0.02
    
    # 1.0
    if x > 85 and (y >= 40 and y <= 60):
        return 1.0
    
    # 0.8
    if x > 85 and (y >= 25 and y <= 40 or y >= 60 and y <= 85):
        return 0.8
    
    return 0.0


def in_window(events_match, time_window):
    start, end = events_match[0], events[-1]
    return start['eventSec'] >= time_window[0] and end['eventSec'] <= time_window[1]



def segno(x):
    """
    Input:  x, a number
    Return:  1.0  if x>0,
            -1.0  if x<0,
             0.0  if x==0
    """
    if   x  > 0.0: return 1.0
    elif x  < 0.0: return -1.0
    elif x == 0.0: return 0.0

def standard_dev(list):
    ll = len(list)
    m = 1.0 * sum(list)/ll
    return ( sum([(elem-m)**2.0 for elem in list]) / ll )**0.5

def list_check(lista):
    """
    If a list has only one element, return that element. Otherwise return the whole list.
    """
    try:
        e2 = lista[1]
        return lista
    except IndexError:
        return lista[0]
    
def pdf(binsize, input, out='no', normalize=True, include_zeros=False, vmin='NA', vmax='NA',start_from='NA', closing_bin=False):
    """
    Return the probability density function of "input"
    using linear bins of size "binsize"

    Input format: one column of numbers

    Example:
    ---------
      a, m = 0.5, 1.
      data = np.random.pareto(a, 1000) + m
      xy = pdf(10.0, data)
    """
    # Detect input type:
    if input == sys.stdin:
    # the data come form standard input
        d = [ list_check(map(float,l.split())) for l in sys.stdin ]
    #         d = [ float(l) for l in sys.stdin if l.strip() ]
    elif isinstance(input, str):
        # the data come from a file
        d = [ list_check(map(float,l.split())) for l in open(input,'r') ]
    #         d = [ float(w) for w in open(input,'r') if w.split()]
    else:
        # the data are in a list
        try:
            iterator = iter(input)
            d = list(input)
        except TypeError:
            print ("The input is not iterable.")

    bin = 1.0*binsize
    d.sort()
    lend = len(d)
    hist = []
    if out != 'no' and out != 'stdout': f = open(out,'wb')

    j = 0
    if not isinstance(start_from, str):
        i = int(start_from / bin)+ 1.0 * segno(start_from)
    else:
        i = int(d[j] / bin) + 1.0 *segno(d[j])

    while True:
        cont = 0
        average = 0.0
        if i>=0: ii = i-1
        else:   ii = i
        # To include the lower end in the previous bin, substitute "<" with "<="
        while d[j] < bin*(ii+1):
            cont += 1.0
            average += 1.0
            j += 1
            if j == lend: break
        if cont > 0 or include_zeros==True:
            if normalize == True and i != 0:
                hist += [[ bin*(ii)+bin/2.0 , average/(lend*bin) ]]
            elif i != 0:
                hist += [[ bin*(ii)+bin/2.0 , average/bin ]]
        if j == lend: break
        i += 1
    if closing_bin:
        # Add the "closing" bin
        hist += [[ hist[-1][0]+bin , 0.0 ]]
    if out == 'stdout':
        for l in hist:
            print("%s %s" % (l[0],l[1]))
    elif out != 'no':
        for l in hist:
            f.write("%s %s\n" % (l[0],l[1]))
        f.close()
    if out == 'no': return hist

def lbpdf(binsize, input, out='no'):
    """
    Return the probability density function of "input"
    using logarithmic bins of size "binsize"

    Input format: one column of numbers

    Example:
    ---------
      a, m = 0.5, 1.
      data = np.random.pareto(a, 1000) + m
      xy = lbpdf(1.5, data)
    """
    # Detect input type:
    if input == sys.stdin:
    # the data come form standard input
        d = [ list_check(map(float,l.split())) for l in sys.stdin ]
    #         d = [ float(l) for l in sys.stdin if l.strip() ]
    elif isinstance(input, str):
        # the data come from a file
        d = [ list_check(map(float,l.split())) for l in open(input,'r') ]
    #         d = [ float(w) for w in open(input,'r') if w.split()]
    else:
        # the data are in a list
        try:
            iterator = iter(input)
            d = list(input)
        except TypeError:
            print ("The input is not iterable.")

    bin = 1.0*binsize
    d.sort()
    # The following removes elements too close to zero
    while d[0] < 1e-12:
        del(d[0])
    lend = len(d)
    tot = 0
    hist = []

    j = 0
    i = 1.0
    previous = min(d)

    while True:
        cont = 0
        average = 0.0
        next = previous * bin
        # To include the lower end in the previous bin, substitute "<" with "<="
        while d[j] < next:
            cont += 1.0
            average += 1.0
            j += 1
            if j == lend: break
        if cont > 0:
            hist += [[previous+(next-previous)/2.0 , average/(next-previous)]]
            tot += average
        if j == lend: break
        i += 1
        previous = next

    if out != 'no' and out != 'stdout': f = open(out,'wb')
    if out == 'stdout':
        for x,y in hist:
            print("%s %s" % (x,y/tot))
    elif out != 'no':
        f = open(out,'wb')
        for x,y in hist:
            f.write("%s %s\n" % (x,y/tot))
        f.close()
    if out == 'no': return [[x,y/tot] for x,y in hist]

class Parameter:
    def __init__(self, value):
        self.value = value

    def set(self, value):
        self.value = value

    def __call__(self):
        return self.value
    
def LSfit(function, parameters, y, x):
    """
    *** ATTENTION! ***
    *** _x_ and _y_ MUST be NUMPY arrays !!! ***
    *** and use NUMPY FUNCTIONS, e.g. np.exp() and not math.exp() ***

    _function_    ->   Used to calculate the sum of the squares:
                         min   sum( (y - function(x, parameters))**2 )
                       {params}

    _parameters_  ->   List of elements of the Class "Parameter"
    _y_           ->   List of observations:  [ y0, y1, ... ]
    _x_           ->   List of variables:     [ [x0,z0], [x1,z1], ... ]

    Then _function_ must be function of xi=x[0] and zi=x[1]:
        def f(x): return x[0] *  x[1] / mu()

        # Gaussian
            np.exp( -(x-mu())**2.0/sigma()**2.0/2.0)/(2.0*sigma()**2.0*np.pi)**0.5
        # Lognormal
            np.exp( -(np.log(x)-mu())**2.0/sigma()**2.0/2.0)/(2.0*sigma()**2.0*np.pi)**0.5/x

    Example:
        x=[np.random.normal() for i in range(1000)]
        variables,data = map(np.array,zip(*pdf(0.4,x)))

        # giving INITIAL _PARAMETERS_:
        mu     = Parameter(7)
        sigma  = Parameter(3)

        # define your _FUNCTION_:
        def function(x): return np.exp( -(x-mu())**2.0/sigma()**2.0/2.0)/(2.0*sigma()**2.0*np.pi)**0.5

        ######################################################################################
        USA QUESTE FORMULE
        #Gaussian formula
        #np.exp( -(x-mu())**2.0/sigma()**2.0/2.0)/(2.0*np.pi)**0.5/sigma()
        # Lognormal formula
        #np.exp( -(np.log(x)-mu())**2.0/sigma()**2.0/2.0)/(2.0*np.pi)**0.5/x/sigma()
        ######################################################################################


        np.exp( -(x-mu())**2.0/sigma()**2.0/2.0)/(2.0*np.pi)**0.5/sigma()

        # fit! (given that data is an array with the data to fit)
        popt,cov,infodict,mesg,ier,pcov,chi2 = LSfit(function, [mu, sigma], data, variables)
    """
    x = np.array(x)
    y = np.array(y)

    def f(params):
        i = 0
        for p in parameters:
            p.set(params[i])
            i += 1
        return y - function(x)

    p = [param() for param in parameters]
    popt,cov,infodict,mesg,ier = optimize.leastsq(f, p, maxfev=10000, full_output=1) #, warning=True)   #, args=(x, y))

    if (len(y) > len(p)) and cov is not None:
        #s_sq = (f(popt)**2).sum()/(len(y)-len(p))
        s_sq = (infodict['fvec']**2).sum()/(len(y)-len(p))
        pcov = cov * s_sq
    else:
        pcov = float('Inf')

    R2 = 1.0 - (infodict['fvec']**2.0).sum() / standard_dev(y)**2.0 / len(y)

    # Detailed Output: p,cov,infodict,mesg,ier,pcov,R2
    return popt,cov,infodict,mesg,ier,pcov,R2

def maximum_likelihood(function, parameters, data, full_output=True, verbose=True):
    """
    function    ->  callable: Distribution from which data are drawn. Args: (parameters, x)
    parameters  ->  np.array: initial parameters
    data        ->  np.array: Data

    Example:

        m=0.5
        v=0.5
        parameters = numpy.array([m,v])

        data = [random.normalvariate(m,v**0.5) for i in range(1000)]

        def function(p,x): return numpy.exp(-(x-p[0])**2.0/2.0/p[1])/(2.0*numpy.pi*p[1])**0.5

        maximum_likelihood(function, parameters, data)


        # # Check that is consistent with Least Squares when "function" is a gaussian:
        # mm=Parameter(0.1)
        # vv=Parameter(0.1)
        # def func(x): return numpy.exp(-(x-mm())**2.0/2.0/vv())/(2.0*numpy.pi*vv())**0.5
        # x,y = zip(*pdf(0.1,data,out='no'))
        # popt,cov,infodict,mesg,ier,pcov,chi2 = LSfit(func, [mm,vv], y, x)
        # popt
        #
        # # And with the exact M-L values:
        # mm = sum(data)/len(data)
        # vv = standard_dev(data)
        # mm, vv**2.0
    """

    def MLprod(p, data, function):
        return -np.sum(np.array([np.log(function(p,x)) for x in data]))

    return optimize.fmin(MLprod, parameters, args=(data,function), full_output=full_output, disp=verbose)

def get_event_name(event):
    event_name = ''
    try:
        if event['subEventName'] != '':
            event_name = event_names_df[(event_names_df.event == event['eventName']) & (event_names_df.subevent == event['subEventName'])].subevent_label.values[0]
        else:
            event_name = event_names_df[event_names_df.event == event['eventName']].event_label.values[0]
    except TypeError:
        #print event
        pass
    
    return event_name
    

def passing_network(match_Id = 2576105):

    for nation in nations:
        for match in matches[nation]:
            if match['wyId'] == match_Id:
                for competition in competitions:
                    if competition['wyId'] == match['competitionId']:
                        competition_name = competition['area']['name']
                team1_name = match['label'].split('-')[0].split(' ')[0]
                team2_name = match['label'].split('-')[1].split(' ')[1].split(',')[0]
    
    list_ev_match = []
    for ev_match in events[competition_name]:
        if ev_match['matchId'] == match_Id:
            if ev_match['eventName'] == 'Pass':
                list_ev_match.append(ev_match)

    df_ev_match = pd.DataFrame(list_ev_match)
    first_half_max_duration = np.max(df_ev_match[df_ev_match['matchPeriod'] == '1H']['eventSec'])

    #sum 1H time end to all the time in 2H
    for i in list_ev_match:
        if i['matchPeriod'] == '1H':
            i['eventSec'] = i['eventSec']
        else:
            i['eventSec'] += first_half_max_duration
            
    #selected the first team for each team
    team_1_time_player = []
    team_2_time_player = []
    for i in list_ev_match:
        if i['teamId'] == list_ev_match[0]['teamId']:
            team_1_time_player.append([i['playerId'],i['eventSec']])
        else: 
            team_2_time_player.append([i['playerId'],i['eventSec']])


    df_team_1_time_player = pd.DataFrame(team_1_time_player, columns=['sender','eventSec'])
    dict_first_team_1 = [(n, y-x) for n,x,y in zip(df_team_1_time_player.groupby('sender').agg('min').reset_index()['sender'],df_team_1_time_player.groupby('sender').agg('min').reset_index()['eventSec'],df_team_1_time_player.groupby('sender').agg('max').reset_index()['eventSec'])]
    dict_first_team1 = []
    for i in dict_first_team_1:
        if i[1] > 3000:
            dict_first_team1.append(i)
        else:
            pass
    dict_first_team_1 = dict(dict_first_team1)

    df_team_2_time_player = pd.DataFrame(team_2_time_player, columns=['sender','eventSec'])
    dict_first_team_2 = [(n, y-x) for n,x,y in zip(df_team_2_time_player.groupby('sender').agg('min').reset_index()['sender'],df_team_2_time_player.groupby('sender').agg('min').reset_index()['eventSec'],df_team_2_time_player.groupby('sender').agg('max').reset_index()['eventSec'])]
    dict_first_team2 = []
    for i in dict_first_team_2:
        if i[1] > 3000:
            dict_first_team2.append(i)
        else:
            pass
    dict_first_team_2 = dict(dict_first_team2)

    #split pass between team1 and team2
    team_1 = []
    team_2 = []
    for i in list_ev_match:
        if i['teamId'] == list_ev_match[0]['teamId']:
            if i['playerId'] in dict_first_team_1:
                team_1.append(i['playerId'])
            else:
                pass
        else:
            if i['playerId'] in dict_first_team_2:
                team_2.append(i['playerId'])

    df_team_1 = pd.DataFrame(team_1, columns=['sender'])
    df_team_2 = pd.DataFrame(team_2, columns=['sender'])

    #define sender and receiver
    df_team_1 = pd.DataFrame(team_1, columns=['sender'])
    df_team_1['receiver'] = df_team_1['sender'].shift(-1).dropna()
    df_team_2 = pd.DataFrame(team_2, columns=['sender'])
    df_team_2['receiver'] = df_team_2['sender'].shift(-1).dropna()


    #create list with sendere and receiver
    df_team1 = []
    for i,row in df_team_1.iterrows():
        if row['sender'] == row['receiver']:
            pass        
        else:
            df_team1.append([row['sender'],row['receiver']])
    df_team1 = pd.DataFrame(df_team1, columns=['sender','receiver'])  
    df_team1_count = df_team1.groupby(['sender','receiver']).size().reset_index(name="Time")

    df_team2 = []
    for i,row in df_team_2.iterrows():
        if row['sender'] == row['receiver']:
            pass        
        else:
            df_team2.append([row['sender'],row['receiver']])
    df_team2 = pd.DataFrame(df_team2, columns=['sender','receiver']) 
    df_team2_count = df_team2.groupby(['sender','receiver']).size().reset_index(name="Time")

    list_ev_match = []
    for ev_match in events['Italy']:
        if ev_match['matchId'] == match_Id: #Napoli - Juventus
            if ev_match['eventName'] == 'Pass':
                list_ev_match.append(ev_match)
            else:
                pass
        else:
            pass


    pos1 = {}
    x, y = [], []
    df_match = pd.DataFrame(list_ev_match)
    for player in list(df_team1['sender'].unique()):
        df_match_player = df_match[df_match.playerId==player]
        for i,row in df_match_player.iterrows():
            x.append(row['positions'][0]['x'])
            y.append(row['positions'][0]['y'])

        pos1[int(player)] = (int(np.mean(x)), int(np.mean(y)))

    pos2 = {}
    x, y = [], []
    df_match = pd.DataFrame(list_ev_match)
    for player in list(df_team2['sender'].unique()):
        if player != 8032:
            df_match_player = df_match[df_match.playerId==player]
            for i,row in df_match_player.iterrows():
                x.append(row['positions'][0]['x'])
                y.append(row['positions'][0]['y'])
        else:
            pass

        pos2[int(player)] = (int(np.mean(x)), int(np.mean(y)))

    def merge_two_dicts(x, y):
        z = x.copy()   # start with x's keys and values
        z.update(y)    # modifies z with y's keys and values & returns None
        return z

    pos_all = merge_two_dicts(pos1,pos2)
    
    # networkX team 1

    g = nx.Graph()
    for i,row in df_team1_count.iterrows():
        g.add_edge(row['sender'],row['receiver'],weight=row['Time'])
    nx.draw(g, pos1, with_labels=True, font_weight='bold')
    nx.write_gexf(g, "team_1.gexf")
    print (team1_name)
    print ('centrality =', nx.global_reaching_centrality(g))
    print ('algebric connectivity =', nx.algebraic_connectivity(g))
    print ('density =', nx.density(g))
    print ('')

    g = nx.Graph()
    for i,row in df_team2_count.iterrows():
        g.add_edge(row['sender'],row['receiver'],weight=row['Time'])
    nx.draw(g, pos2, node_color='b', with_labels=True, font_weight='bold')
    nx.write_gexf(g, "team_2.gexf")
    print (team2_name)
    print ('centrality =', nx.global_reaching_centrality(g))
    print ('algebric connectivity =', nx.algebraic_connectivity(g))
    print ('density =', nx.density(g))

    plt.show()

    print ('%s: mean ='%team1_name, df_team1_count.Time.mean()/df_team1_count.Time.sum()*100, '; std =', df_team1_count.Time.std()/df_team1_count.Time.sum()*100, '; count =', df_team1_count.Time.sum())
    print ('%s: mean ='%team2_name, df_team2_count.Time.mean()/df_team2_count.Time.sum()*100, '; std =', df_team2_count.Time.std()/df_team2_count.Time.sum()*100, '; count =', df_team2_count.Time.sum())
    


