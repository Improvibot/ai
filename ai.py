import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import sys
sys.path.insert(0, '../MIDI')
from MIDI.Container import *
from MIDI.Internal  import *
import MIDI.Internal as i
from MIDI.Message   import *

sys.path.insert(1, '../db')
#from db.db     import Database
from db.models import *

from datetime import date
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
#import mido
#from mido import MidiFile, MidiTrack, Message, merge_tracks
#from song import Song

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

NUM_NOTES = 8
NOTE_VALUES_MIN = 32
NOTE_VALUES_MAX = 96
VELOCITY_VALUES_MIN=40
VELOCITY_VALUES_MAX = 80
NOTE_LENGTHS = [0.25,   # 16th Note
                0.5,    # 8th  Note
                1.0,    # Quarter Note
                2.0     # Half Note
                ]
                #4.0]    # Whole Note
FEEDBACK_VALUES = [1, 2, 3, 4, 5]

epochs = 1  # Temporarily setting to 1 for testing purposes
#epochs = 1000


def generate_initial_sequence(num_notes):
    return {
        'note': np.random.randint(NOTE_VALUES_MIN, NOTE_VALUES_MAX, num_notes),
        'velocity': np.random.randint(VELOCITY_VALUES_MIN, VELOCITY_VALUES_MAX, num_notes),
        'length': np.random.choice(NOTE_LENGTHS, num_notes)
    }

model = Sequential([
    LSTM(128, input_shape=(NUM_NOTES, 3)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='mse')


def generate_midi(model_output):
    song = File()
    # Convert model output to MIDI messages
    track = song.add_track()
    curr_ts = TimeStamp(1, 1)
    for n in model_output:
        for note, velocity, length in n:
            logging.info(f'note: {note}\tvelocity: {velocity}\tlength: {length}')
            curr_ts = make_note(track, curr_ts, 0, int(note), int(velocity), length)
    return song

def make_note(track, ts, channel, note, velocity, length):
    new_ts = ts.add(length, i.TimeSignature(4, 4))
    track.append(NoteOn(ts, channel, note, velocity))
    track.append(NoteOff(new_ts, channel, note, velocity))
    return new_ts


def get_user_feedback():
        # Simulated function to get user feedback
            return np.random.choice(FEEDBACK_VALUES)

def train_model(model, input_sequence, feedback):
    target_output = np.array([input_sequence])  # Target output is the same as input for now

    # Adjust target output based on feedback
    feedback_factor = feedback / 5.0  # Normalize feedback to a factor between 0 and 1
    target_output[:, :, :2] *= feedback_factor  # Adjust note and velocity based on feedback
    target_output[:, :, 2] *= 1.0 / feedback_factor  # Adjust note length inversely based on feedback

    model.fit(input_sequence[np.newaxis, :, :], target_output, epochs=1, verbose=0)

def generate_song(database=None):
    for epoch in range(epochs):
        input_sequence = generate_initial_sequence(NUM_NOTES)
        input_sequence = np.array([
            input_sequence['note'],
            input_sequence['velocity'],
            input_sequence['length']
        ]).T  # Transpose to have shape (num_notes, 3)
        model_output = []
        model_output.append(model.predict(np.array([input_sequence])))
        model_output = [input_sequence]
        print(f'model_output: {model_output}')
        user_feedback = get_user_feedback()
        song = generate_midi(model_output)
        song.save('midi_ai')
        if database is not None:
            s = Song(title='generic title',
                     creation_date=date.today(),
                     data=song.encode(Encoding.MIDI))
            database.save(s)
        #song = generate_midi(outport, model_output)
        train_model(model, input_sequence, user_feedback)
        return song
