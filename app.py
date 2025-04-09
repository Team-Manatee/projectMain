from flask import Flask, render_template, send_from_directory, request, send_file

from crypt import methods

# pip install soundfile # soundfile should fix any errors if they occur
# pip install ffmpeg

from distutils.command.upload import upload
from fileinput import filename
from os import scandir

import os
import torch
import torchaudio
from asteroid.models import ConvTasNet  # Ensure you have installed asteroid via pip
from torch.serialization import safe_globals

import shutil
import ffmpeg
import demucs.separate
import threading

# Initialize Flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac'}


#This is the start of Ethan's GUI routes that currently work with his templates


def check_file_eligibility(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/music')
def music_separation():
    return render_template('music.html')

@app.route('/conversation')
def conversation_separation():
    return render_template('conversation.html')

@app.route('/audio')
def audio_player():
    audio_files = []
    if os.path.exists(UPLOAD_FOLDER):
        audio_files = [f for f in os.listdir(UPLOAD_FOLDER) if check_file_eligibility(f)]
    return render_template('index.html', audio_files=audio_files)

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'fileUp' not in request.files:
        return render_template('results.html', msg="No file part")

    fileitem = request.files['fileUp']

    if fileitem.filename == '':
        return render_template('results.html', msg="No file selected")

    if fileitem and check_file_eligibility(fileitem.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], fileitem.filename)
        fileitem.save(filepath)
        flash(f"Uploaded {fileitem.filename} successfully!", "info")
        return redirect(url_for('audio_player'))
    else:
        return render_template('results.html', msg="File type not allowed. Please upload .mp3 or .wav files only.")

@app.route('/delete_audio/<filename>', methods=['POST'])
def delete_audio(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        flash(f"{filename} has been deleted.", "info")
    else:
        flash(f"{filename} not found.", "error")
    return redirect(url_for('audio_player'))

@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


#This is the beginning of Michael's music separator app's function


#this function takes the filename, checks for a '.' and splits the name at the final '.' to check if its extension matches an item from the ALLOWED_EXTENSIONS set
def check_file_eligibility(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#this allows the html file to extract the files from downloads
@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/separated/htdemucs/<folder_name>/<audio_name>')
def serve_sep_file(folder_name, audio_name):
    return send_from_directory(f'separated/htdemucs/{folder_name}', audio_name)

#directs anyone who arrives at the home URL to execute the index function
@app.route('/')
def index():
    #Get list of audio files in the uploads directory
    audio_files = []
    if os.path.exists(UPLOAD_FOLDER):
        audio_files = [f for f in os.listdir(UPLOAD_FOLDER) if check_file_eligibility(f)]
    #this sends the audio_files list to the html template
    return render_template('index.html', audio_files=audio_files)

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if request.method == 'POST':
            fileitem = request.files['fileUp']
            fileitem.save('uploads/' + fileitem.filename)
            # this sends the audio_files list to the html template
            audio_files = []
            if os.path.exists(UPLOAD_FOLDER):
                audio_files = [f for f in os.listdir(UPLOAD_FOLDER) if check_file_eligibility(f)]
            return render_template('index.html', audio_files=audio_files)



@app.route('/music')
def music_sep():
    # Stores absolute paths for audio files in the uploads directory
    audio_files = []

    # Stores file names
    file_names = []

    # Stores list of threads for each track's separation process
    thread_web = []

    # get the current working directory
    cur = f"{os.getcwd()}"

    # access uploads folder
    uploads = cur + "/uploads/"
    if os.path.exists(uploads):
        audio_files = [(uploads + f) for f in os.listdir(uploads) if check_file_eligibility(f)]
        file_names = [f for f in os.listdir(uploads) if check_file_eligibility(f)]

    for f in audio_files:
        print(f)
        thread_web.append(threading.Thread(target=perform_sep(f)))

    if(thread_web):
        for t in thread_web:
            t.start()
        for t in thread_web:
            t.join()

    # path to separated audio
    sep_audio_path = os.getcwd() + "/separated/htdemucs/"

    # stores track names (folder names) for header h2 in sepIndex.html
    song_name = []

    # stores separated audio files in each track folder
    sep_audio_files = []

    track_dict = {}

    # iterate through separated audio file directory
    sep_dir = os.scandir(sep_audio_path)
    for entry in sep_dir:
        # if entry is a folder...
        if entry.is_dir():

            # append the folder name
            song_name.append(entry.name)

            # iterate through the track folder
            t_fold = scandir(entry)
            for t in t_fold:
                # append the separated audio file name
                sep_audio_files.append(t.name)

            # define a dict entry for {"track_name" : [file1, file2, file3]}
            track_dict.update({f"{entry.name}" : sep_audio_files})

            # clear the separated audio files
            sep_audio_files.clear()



    return render_template(
        'sepIndex.html',
        file_names= song_name,
        audio_files=track_dict
    )

import shlex
def perform_sep(f):
    # --mp3 converts the output wav to mp3;
    # could also use --flac to output wav files as flac instead
    demucs.separate.main(shlex.split(f'--mp3 \"{f}\"'))



#This is the start of Jonny's vocal separator app's functions

def check_file_eligibility(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to load and resample audio to 16,000 Hz
def load_and_resample(file_path, target_sample_rate=16000):
    waveform, orig_sample_rate = torchaudio.load(file_path)
    if orig_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        print(f"Resampled from {orig_sample_rate} Hz to {target_sample_rate} Hz.")
    else:
        print(f"Audio is already at {target_sample_rate} Hz.")
    return waveform, target_sample_rate


# Function to split the audio waveform into segments of a given duration (in seconds)
def split_audio_into_segments(waveform, sample_rate, segment_duration=3):
    segment_length = int(segment_duration * sample_rate)  # e.g., 3 sec * 16000 = 48000 samples
    total_samples = waveform.shape[-1]
    segments = []
    for start in range(0, total_samples, segment_length):
        end = start + segment_length
        if end > total_samples:
            pad_length = end - total_samples
            segment = torch.nn.functional.pad(waveform[..., start:total_samples], (0, pad_length))
        else:
            segment = waveform[..., start:end]
        segments.append(segment)
    print(f"Split audio into {len(segments)} segments of {segment_duration} seconds each.")
    return segments


# Function to normalize audio so that its peak amplitude is at a target value (e.g., 0.99)
# Fixes the audio blowout
def normalize_audio(waveform, target_peak=0.99):
    max_amp = waveform.abs().max()
    if max_amp > 0:
        waveform = waveform / max_amp * target_peak
    return waveform


# Route to serve uploaded files
@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory('uploads', filename)


# Route to serve separated files
@app.route('/separated/<filename>')
def serve_separated_file(filename):
    return send_from_directory('separated', filename)


# Home route: list all uploaded audio files
@app.route('/')
def index():
    audio_files = []
    if os.path.exists(UPLOAD_FOLDER):
        audio_files = [f for f in os.listdir(UPLOAD_FOLDER) if check_file_eligibility(f)]
    return render_template('index.html', audio_files=audio_files)


# Upload page
@app.route('/upload')
def upload():
    return render_template('upload.html')


# Handle file upload
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if request.method == 'POST':
        fileitem = request.files['fileUp']
        if fileitem and check_file_eligibility(fileitem.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], fileitem.filename)
            fileitem.save(file_path)
    audio_files = []
    if os.path.exists(UPLOAD_FOLDER):
        audio_files = [f for f in os.listdir(UPLOAD_FOLDER) if check_file_eligibility(f)]
    return render_template('index.html', audio_files=audio_files)


# Set your Hugging Face token (Not sure if this is necessary or not)
#I removed the HF token because it messes up the git push

# Load the pre-trained ConvTasNet model from the 16k repository using safe_globals
with safe_globals(["numpy.core.multiarray.scalar"]):
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k", use_auth_token=HF_TOKEN)
model.eval()


def separate_audio(file_path):
    # Load and resample the audio to 16,000 Hz
    waveform, sample_rate = load_and_resample(file_path, target_sample_rate=16000)
    # Convert to mono if necessary
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Split the waveform into 3-second segments
    segments = split_audio_into_segments(waveform, sample_rate, segment_duration=3)

    separated_segments = []  # List to hold the separated sources for each segment
    for segment in segments:
        # Ensure the segment has shape (1, time)
        if segment.dim() == 1:
            segment = segment.unsqueeze(0)
        segment = segment.unsqueeze(0)  # Add batch dimension: (batch, time)
        with torch.no_grad():
            estimated_sources = model(segment)  # shape: (batch, n_sources, time)
        sources = estimated_sources.squeeze(0)  # shape: (n_sources, time)
        separated_segments.append(sources)

    return separated_segments, sample_rate


# Route to display the vocal separation page (with dropdown)
@app.route('/vocal')
def vocal():
    audio_files = []
    if os.path.exists(UPLOAD_FOLDER):
        audio_files = [f for f in os.listdir(UPLOAD_FOLDER) if check_file_eligibility(f)]
    if not audio_files:
        return render_template('results.html', msg="No audio files available for separation. Please upload some files.")
    return render_template('vocal.html', audio_files=audio_files)


# Route to perform voice separation, concatenate segments, normalize, and save one file per source
@app.route('/separate', methods=['POST'])
def separate():
    selected_file = request.form.get("audio_file")
    if not selected_file or not check_file_eligibility(selected_file):
        return render_template('results.html', msg="Invalid file selected.")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
    if not os.path.exists(file_path):
        return render_template('results.html', msg="File does not exist.")
    try:
        separated_segments, sample_rate = separate_audio(file_path)
    except Exception as e:
        return render_template('results.html', msg="Error during separation: " + str(e))

    # Concatenate segments for each source along the time dimension.
    n_sources = separated_segments[0].shape[0]
    concatenated_sources = []
    for src_idx in range(n_sources):
        segments_for_source = [seg[src_idx] for seg in separated_segments]
        concatenated_source = torch.cat(segments_for_source, dim=-1)
        concatenated_sources.append(concatenated_source)

    separated_folder = 'separated'
    os.makedirs(separated_folder, exist_ok=True)
    separated_files = []

    # Normalize and save one file per source
    for src_idx, source in enumerate(concatenated_sources):
        source = normalize_audio(source)  # Normalize to prevent clipping
        output_filename = f"{os.path.splitext(selected_file)[0]}_source{src_idx + 1}.wav"
        output_filepath = os.path.join(separated_folder, output_filename)
        # Ensure correct shape [channels, time] for saving
        if source.dim() == 1:
            source = source.unsqueeze(0)
        else:
            source = source.unsqueeze(0)
        torchaudio.save(output_filepath, source, sample_rate)
        separated_files.append(output_filename)

    msg = "Voice separation complete. Separated files: " + ", ".join(
        [f"<a href='/separated/{f}'>{f}</a>" for f in separated_files]
    )
    return render_template('results.html', msg=msg)


#runs the app
if __name__ == '__main__':
    app.run(debug=False) #this runs the app with debug mode active