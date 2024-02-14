import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import sys
sys.path.insert(0, '../MIDI')
sys.path.insert(1, '../db')

from db.db import Database
#from dotenv import load_dotenv
#load_dotenv()

#import mido
#from mido import Message
import ai

#mido.set_backend(name='midojack', load=True)

#with mido.open_output(autoreset=True) as outport:
cont = True
database = Database()
while cont:
    song = ai.generate_song(database)
    print(song)
    print('Press [Enter] to generate a new song, "quit" to quit.')
    response = input()
    if response == 'quit':
        cont= False
        #outport.panic()
