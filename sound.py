from midi2audio import FluidSynth

def initialize_synthesizer(soundfont_path):
    try:
        fs = FluidSynth(soundfont_path)
        return fs
    except Exception as e:
        print(f"Error initializing FluidSynth: {e}")
        return None

def play_generated_midi(midi_file, soundfont_path):
    fs = initialize_synthesizer(soundfont_path)
    if fs is not None:
        fs.midi_to_audio(midi_file, "output.wav")
        print("✅ Successfully converted MIDI to WAV with piano sounds!")


soundfont = r"C:\Users\zwano\Downloads\maestro-v3.0.0-midi\maestro-v3.0.0\Piano Infinity.sf2"  
generated_midi = "generated_musegan.mid"  

play_generated_midi(generated_midi, soundfont)