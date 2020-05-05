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
import matplotlib.pyplot as plt 


EVENT_TYPES = ['Duel', 'Foul', 'Free Kick', 'Interruption', 
             'Offside', 'Others on the ball', 'Pass', 'Shot']


data_folder='data/'
def load_public_dataset(data_folder=data_folder, tournaments=['Italy','England','Germany', 
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
        
    # loading the competitions data
    teams={}
    with open('./data/teams.json') as json_data:
        teams = json.load(json_data)
        
    return matches, events, players, competitions, teams

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
    
def is_in_match(player_id, match):
    team_ids = list(match['teamsData'].keys())
    all_players = []
    for team in team_ids:
        in_bench_players = [m['playerId'] for m in match['teamsData'][team]['formation']['bench']]
        in_lineup_players = [m['playerId'] for m in match['teamsData'][team]['formation']['lineup']]
        substituting_players = [m['playerIn'] for m in match['teamsData'][team]['formation']['substitutions']]
        all_players += in_bench_players + in_lineup_players + substituting_players
    return player_id in all_players


    


