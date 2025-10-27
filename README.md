![SoExcited~GIF](https://github.com/user-attachments/assets/59254f0f-0c1c-4ec1-9fbe-19ec7ab78b18)# Music Generator using GANs   

## 🎵 Project Overview  
**Music Generator** is a deep‐learning project that uses Generative Adversarial Networks (GANs) to create **MIDI files**, which are then converted into audio. The goal is to explore generative modelling in the music domain, synthesize new musical sequences, and listen to them as audio.  

## Features  
- Generates new musical sequences in MIDI format via a GAN architecture.  
- Converts those MIDI sequences into playable audio (WAV/MP3) for easy listening.  

## Architecture & Approach  
1. **Generator**: Takes random latent vectors (and optionally style / condition vector) and outputs a sequence in MIDI format (e.g., note events, velocity, timing).  
2. **Discriminator**: Learns to distinguish between real MIDI sequences (from dataset) and fake ones from the generator.  
3. **Training pipeline**:  
   - Dataset of MIDI files → preprocessing to convert into tensor representation.  
   - Generator & discriminator trained adversarially until generator produces plausible sequences.  
   - After generation, MIDI output is converted to audio (via midi2audio FluidSynth).  
4. **MIDI → Audio Conversion**: Uses pretty_midi to render MIDI into WAV/MP3 for listening.  

## Tech Stack  
- Language: Python 3.x  
- Deep Learning Framework: tensorflow
- MIDI Processing: pretty_midi
- Audio Rendering:midi2audio FluidSynth
- Dataset: MAESTRO


## ☆*: .｡. o(≧▽≦)o .｡.:*☆ OUTPUT
[midi.mp3](https://github.com/user-attachments/files/23170606/WhatsApp.Audio.2025-07-06.at.7.39.02.PM.online-audio-converter.com.mp3)

