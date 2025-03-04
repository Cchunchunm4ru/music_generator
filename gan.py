import os
import numpy as np
import tensorflow as tf
import pretty_midi
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2DTranspose, Conv2D, LeakyReLU, BatchNormalization, Flatten, Dropout, GaussianNoise, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ---------------------------
# 1️⃣ MuseGAN Configuration
# ---------------------------
latent_dim = 1000  
seq_length = 32   
num_tracks = 4   
# ---------------------------
# 2️⃣ Load Reference MIDI File
# ---------------------------
def midi_to_latent_vector(midi_file):
    try:
        midi = pretty_midi.PrettyMIDI(midi_file)
        latent_vector = np.random.normal(0, 1, (1, latent_dim))
        notes = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append((note.pitch, note.start, note.end))
        if len(notes) > 0:
            for i, (pitch, start, end) in enumerate(notes[:latent_dim]):
                latent_vector[0, i] = (pitch - 21) / (108 - 21) + (end - start) * 0.1  
        return latent_vector
    except Exception as e:
        print(f"Error processing MIDI file: {e}")
        return np.random.normal(0, 1, (1, latent_dim))  

def midi_to_piano_roll(midi_file, resolution=4):
    try:
        midi = pretty_midi.PrettyMIDI(midi_file)
        piano_roll = midi.get_piano_roll(fs=resolution)
        piano_roll = piano_roll[:128, :seq_length * resolution]  
        return piano_roll
    except Exception as e:
        print(f"Error processing MIDI file: {e}")
        return np.zeros((128, seq_length * resolution))

# ---------------------------
# 3️⃣ Define MuseGAN Generator with Swish Activation
# ---------------------------
def swish(x):
    return x * tf.nn.sigmoid(x)

def build_generator():
    noise_input = Input(shape=(latent_dim,))
    
    x = Dense(1024 * 4 * 4)(noise_input)
    x = BatchNormalization()(x)
    x = Lambda(swish)(x)
    x = Reshape((4, 4, 1024))(x)
    
    x = Conv2DTranspose(512, kernel_size=3, strides=2, padding='same')(x)  
    x = BatchNormalization()(x)
    x = Lambda(swish)(x)
    
    x = Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(x)  
    x = BatchNormalization()(x)
    x = Lambda(swish)(x)
    
    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)  
    x = BatchNormalization()(x)
    x = Lambda(swish)(x)
    
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)  
    x = BatchNormalization()(x)
    x = Lambda(swish)(x)
    
    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')(x)   
    x = BatchNormalization()(x)
    x = Lambda(swish)(x)
    x = Dropout(0.3)(x)
    
    output = Conv2DTranspose(num_tracks, kernel_size=3, padding='same', activation='tanh')(x)
    
    return Model(noise_input, output, name="Generator")

# ---------------------------
# 4️⃣ Define MuseGAN Discriminator
# ---------------------------
def build_discriminator():
    input_shape = (128, 128, num_tracks)
    music_input = Input(shape=input_shape)
    
    x = GaussianNoise(0.1)(music_input)
    x = Conv2D(32, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(256, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)
    
    return Model(music_input, output, name="Discriminator")

# ---------------------------
# 5️⃣ Compile MuseGAN
# ---------------------------
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.00001, 0.5), metrics=['accuracy'])

gan_input = Input(shape=(latent_dim,))
generated_music = generator(gan_input)
discriminator.trainable = False
gan_output = discriminator(generated_music)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.00002, 0.5))

# ---------------------------
# 6️⃣ Train MuseGAN
# ---------------------------
epochs = 100 
batch_size = 64
reference_midi = r"C:\Users\zwano\Downloads\maestro-v3.0.0-midi\maestro-v3.0.0\2004\MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_07_Track07_wav.midi"


if not os.path.exists(reference_midi):
    print(f"Error: The file {reference_midi} does not exist.")
else:
    real_music = midi_to_piano_roll(reference_midi)
    real_music = np.expand_dims(real_music, axis=-1)  
    real_music = np.repeat(real_music, num_tracks, axis=-1) 
    real_music = np.expand_dims(real_music, axis=0)  
    real_music = np.repeat(real_music, batch_size, axis=0)  

    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        noise += np.random.normal(0, 0.1, noise.shape)  
        fake_music = generator.predict(noise)
        
        real_labels = np.ones((batch_size, 1)) * 0.8  
        fake_labels = np.zeros((batch_size, 1)) + 0.1
        
        d_loss_real = discriminator.train_on_batch(real_music, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_music, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        g_loss = gan.train_on_batch(noise, real_labels)
        
        if epoch % 10 == 0:  
            print(f"Epoch {epoch} | D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f}")

    # ---------------------------
    # 7️⃣ Generate MIDI Output from Reference
    # ---------------------------
    def generate_midi_from_reference(generator, reference_midi, output_file="generated_musegan.mid", max_duration=180, max_simultaneous_notes=10):
        latent_vector = midi_to_latent_vector(reference_midi)
        latent_vector += np.random.normal(0, 0.3, latent_vector.shape)  # Inject noise to promote variation
        generated_music = generator.predict(latent_vector).squeeze()

        # Debugging: Print the generated music values
        print("Generated Music (sample):")
        print(generated_music[:5, :5, :])  # Print a small sample of the generated music for inspection

        # Normalize the generated music values to the range [0, 1]
        generated_music = (generated_music - generated_music.min()) / (generated_music.max() - generated_music.min())

        # Check if the generated music is all zeros
        if np.all(generated_music == 0):
            print("Generated music is all zeros.")
            return

        # Convert the generated music directly to MIDI notes
        midi = pretty_midi.PrettyMIDI()
        instruments = [pretty_midi.Instrument(program=0) for _ in range(generated_music.shape[2])]  # Set program to 0 (Acoustic Grand Piano)

        for track_idx in range(generated_music.shape[2]):
            track = generated_music[:, :, track_idx]
            active_notes = []
            for time_step in range(track.shape[0]):
                current_notes = []
                for note_idx in range(track.shape[1]):
                    note_value = track[time_step, note_idx]
                    if note_value > 0.05:  # Lower threshold for note activation
                        pitch = note_idx + 21  # Ensure notes are within the audible range
                        if 21 <= pitch < 108:  # Ensure pitch is within valid range
                            start_time = time_step * 0.5  # Adjusting for time scaling to make the MIDI longer
                            end_time = start_time + 1.0  # Increase the duration of the notes
                            note_velocity = int(note_value * 127)  # Scale note value to MIDI velocity range
                            note = pretty_midi.Note(
                                velocity=note_velocity,
                                pitch=pitch,
                                start=start_time,
                                end=end_time
                            )
                            current_notes.append(note)
                # Limit the number of simultaneous notes
                if len(current_notes) > max_simultaneous_notes:
                    current_notes = sorted(current_notes, key=lambda x: x.velocity, reverse=True)[:max_simultaneous_notes]
                active_notes.extend(current_notes)
            instruments[track_idx].notes.extend(active_notes)

        for instrument in instruments:
            midi.instruments.append(instrument)

        # Trim the MIDI to the desired length (3 minutes)
        if midi.get_end_time() > max_duration:
            for instrument in midi.instruments:
                instrument.notes = [note for note in instrument.notes if note.start < max_duration]
                for note in instrument.notes:
                    if note.end > max_duration:
                        note.end = max_duration

        try:
            midi.write(output_file)
            print(f"Generated MIDI saved as {output_file}")
        except Exception as e:
            print(f"Error writing MIDI file: {e}")

    # Run Music Generation with Reference
    generate_midi_from_reference(generator, reference_midi)