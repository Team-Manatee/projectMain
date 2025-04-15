from flask import Flask, render_template, send_from_directory, request, send_file, flash, redirect, url_for
import os
import torch
import torchaudio
from asteroid.models import ConvTasNet
from torch.serialization import safe_globals
import demucs.separate
import shlex

# ============================================================================
# FLASK APPLICATION INITIALIZATION
# ============================================================================
# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flash messages

# Define constants for file storage locations
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create separated folder for both audio separators
SEPARATED_FOLDER = 'separated'
os.makedirs(SEPARATED_FOLDER, exist_ok=True)

# Define allowed audio file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac'}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
# Helper function to check if a file has an allowed extension
def check_file_eligibility(filename):
    """
    Checks if the file has an allowed audio extension.

    Args:
        filename (str): The name of the file to check

    Returns:
        bool: True if the file has an allowed extension, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to load and resample audio to 16,000 Hz
def load_and_resample(file_path, target_sample_rate=16000):
    """
    Loads an audio file and resamples it to the target sample rate.

    Args:
        file_path (str): Path to the audio file
        target_sample_rate (int): Desired sample rate in Hz (default: 16000)

    Returns:
        tuple: (waveform, sample_rate) - The loaded and resampled audio
    """
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
    """
    Splits an audio waveform into fixed-length segments.

    Args:
        waveform (torch.Tensor): The audio waveform
        sample_rate (int): Sample rate of the audio
        segment_duration (int): Duration of each segment in seconds (default: 3)

    Returns:
        list: List of waveform segments
    """
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
def normalize_audio(waveform, target_peak=0.99):
    """
    Normalizes audio waveform to have a specific peak amplitude.

    Args:
        waveform (torch.Tensor): The audio waveform to normalize
        target_peak (float): Desired peak amplitude between 0 and 1 (default: 0.99)

    Returns:
        torch.Tensor: Normalized audio waveform
    """
    max_amp = waveform.abs().max()
    if max_amp > 0:
        waveform = waveform / max_amp * target_peak
    return waveform


# ============================================================================
# AUDIO SEPARATION FUNCTIONS
# ============================================================================
# Perform music separation function using Demucs
def perform_sep(file_path):
    """
    Separates music tracks using Demucs.

    Args:
        file_path (str): Path to the audio file to separate

    Note:
        This function uses the demucs.separate module to separate music into
        individual instrument tracks like drums, bass, vocals, etc.
    """
    # --mp3 converts the output wav to mp3
    demucs.separate.main(shlex.split(f'--mp3 \"{file_path}\"'))


# Get Hugging Face token from environment variable
HF_TOKEN = os.environ.get('HUGGING_FACE_TOKEN', '')

# Load the pre-trained ConvTasNet model for vocal separation
try:
    # Initialize the vocal separation model
    with safe_globals(["numpy.core.multiarray.scalar"]):
        if HF_TOKEN:
            model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri3Mix_sepclean_16k", use_auth_token=HF_TOKEN)
        else:
            model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri3Mix_sepclean_16k")
    model.eval()
except Exception as e:
    print(f"Error loading ConvTasNet model: {e}")
    model = None


# Separate audio function for vocal separation
def separate_audio(file_path):
    """
    Separates vocal and non-vocal components from an audio file.

    Args:
        file_path (str): Path to the audio file to process

    Returns:
        tuple: (separated_segments, sample_rate) - The separated audio segments and sample rate

    Note:
        This function uses the ConvTasNet model to separate vocals from the background.
        It works by processing the audio in small segments and then separating each segment.
    """
    # Check if model is loaded
    if model is None:
        raise Exception("Vocal separation model is not loaded.")

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


# ============================================================================
# FLASK ROUTES - MAIN PAGES
# ============================================================================
# Routes
@app.route('/')
def home():
    """Renders the home page."""
    return render_template('home.html')


@app.route('/music')
def music_separation():
    """
    Renders the music separation page.

    Collects all eligible audio files and passes them to the template.
    """
    audio_files = []
    if os.path.exists(UPLOAD_FOLDER):
        audio_files = [f for f in os.listdir(UPLOAD_FOLDER) if check_file_eligibility(f)]
    return render_template('music.html', audio_files=audio_files)


@app.route('/music_sep', methods=['POST'])
def music_sep():
    """
    Processes the selected audio file using Demucs for music separation.

    This route handles the actual separation of music tracks into components
    like drums, bass, vocals, etc. and displays the results.
    """
    # Get the selected file from the form
    selected_file = request.form.get("audio_file")
    if not selected_file or not check_file_eligibility(selected_file):
        return render_template('seperatedVocals.html', msg="Invalid file selected.")

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
    if not os.path.exists(file_path):
        return render_template('seperatedVocals.html', msg="File does not exist.")

    # Process the selected file with separation
    try:
        print(f"Processing: {file_path}")
        perform_sep(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return render_template('seperatedVocals.html', msg=f"Error during music separation: {str(e)}")

    # Path to separated audio
    sep_audio_path = os.getcwd() + "/separated/htdemucs/"

    # If the path doesn't exist, create it or return an error message
    if not os.path.exists(sep_audio_path):
        os.makedirs(sep_audio_path, exist_ok=True)
        return render_template('seperatedVocals.html', msg="No separated audio files found. Please try again.")

    # Look for the folder matching the filename (without extension)
    filename_base = os.path.splitext(selected_file)[0]
    song_folder = None

    try:
        sep_dir = os.scandir(sep_audio_path)
        for entry in sep_dir:
            if entry.is_dir() and entry.name == filename_base:
                song_folder = entry.name
                break
    except Exception as e:
        print(f"Error scanning separated directory: {e}")
        return render_template('seperatedVocals.html', msg=f"Error accessing separated files: {str(e)}")

    if not song_folder:
        return render_template('seperatedVocals.html',
                               msg="Could not find separated audio files. The separation may have failed.")

    # Get the list of separated tracks
    track_dict = {}
    try:
        sep_audio_files = []
        t_fold = os.scandir(f"{sep_audio_path}/{song_folder}")
        for t in t_fold:
            # Append the separated audio file name
            sep_audio_files.append(t.name)

        # Define a dict entry for {"track_name" : [file1, file2, file3]}
        track_dict = {f"{song_folder}": sep_audio_files}
    except Exception as e:
        print(f"Error accessing separated files: {e}")
        return render_template('seperatedVocals.html', msg=f"Error accessing separated files: {str(e)}")

    return render_template(
        'seperatedMusic.html',
        file_names=[song_folder],
        audio_files=track_dict,
        original_file=selected_file
    )


@app.route('/conversation')
def conversation_separation():
    """Renders the conversation separation page (redirects to vocal separation)."""
    return render_template('vocal.html')


@app.route('/vocal')
def vocal():
    """
    Renders the vocal separation page.

    Collects all eligible audio files and passes them to the template.
    If no audio files are found, displays an error message.
    """
    audio_files = []
    if os.path.exists(UPLOAD_FOLDER):
        audio_files = [f for f in os.listdir(UPLOAD_FOLDER) if check_file_eligibility(f)]
    if not audio_files:
        return render_template('vocal.html', msg="No audio files available for separation. Please upload some files.")
    return render_template('vocal.html', audio_files=audio_files)


@app.route('/separate', methods=['POST'])
def separate():
    """
    Processes the selected audio file to separate vocals.

    This route handles the actual separation of vocals from the background
    using the ConvTasNet model and displays the results.
    """
    selected_file = request.form.get("audio_file")
    if not selected_file or not check_file_eligibility(selected_file):
        return render_template('seperatedVocals.html', msg="Invalid file selected.")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
    if not os.path.exists(file_path):
        return render_template('seperatedVocals.html', msg="File does not exist.")

    if model is None:
        return render_template('seperatedVocals.html',
                               msg="Vocal separation model failed to load. Please check the console for details.")

    try:
        separated_segments, sample_rate = separate_audio(file_path)
    except Exception as e:
        return render_template('seperatedVocals.html', msg="Error during separation: " + str(e))

    # Concatenate segments for each source along the time dimension.
    n_sources = separated_segments[0].shape[0]
    concatenated_sources = []
    for src_idx in range(n_sources):
        segments_for_source = [seg[src_idx] for seg in separated_segments]
        concatenated_source = torch.cat(segments_for_source, dim=-1)
        concatenated_sources.append(concatenated_source)

    separated_files = []

    # Normalize and save one file per source
    for src_idx, source in enumerate(concatenated_sources):
        source = normalize_audio(source)  # Normalize to prevent clipping
        output_filename = f"{os.path.splitext(selected_file)[0]}_source{src_idx + 1}.wav"
        output_filepath = os.path.join(SEPARATED_FOLDER, output_filename)
        # Ensure correct shape [channels, time] for saving
        if source.dim() == 1:
            source = source.unsqueeze(0)
        torchaudio.save(output_filepath, source, sample_rate)
        separated_files.append(output_filename)

    msg = "Voice separation complete. The following sources have been extracted:"

    # Pass both the message, the list of files, and the original file to the template
    return render_template('seperatedVocals.html',
                           msg=msg,
                           separated_files=separated_files,
                           original_file=selected_file)


@app.route('/audio_player')
def audio_player():
    """
    Renders the audio player page.

    Collects all eligible audio files and passes them to the template.
    """
    # Get list of audio files in the uploads directory
    audio_files = []
    if os.path.exists(UPLOAD_FOLDER):
        audio_files = [f for f in os.listdir(UPLOAD_FOLDER) if check_file_eligibility(f)]
    # This sends the audio_files list to the html template
    return render_template('audioPlayer.html', audio_files=audio_files)


# ============================================================================
# FLASK ROUTES - FILE OPERATIONS
# ============================================================================
@app.route('/upload')
def upload():
    """Renders the upload page."""
    return render_template('upload.html')


@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """
    Handles the file upload process.

    Validates the uploaded file, saves it to the uploads directory,
    and redirects to the audio player page.
    """
    if 'fileUp' not in request.files:
        return render_template('seperatedVocals.html', msg="No file part")

    fileitem = request.files['fileUp']

    if fileitem.filename == '':
        return render_template('seperatedVocals.html', msg="No file selected")

    if fileitem and check_file_eligibility(fileitem.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], fileitem.filename)
        fileitem.save(filepath)
        flash(f"Uploaded {fileitem.filename} successfully!", "info")
        return redirect(url_for('audio_player'))
    else:
        return render_template('seperatedVocals.html', msg="File type not allowed. Please upload .mp3 or .wav files only.")


@app.route('/delete_audio/<filename>', methods=['POST'])
def delete_audio(filename):
    """
    Handles the deletion of audio files.

    Removes the specified file from the uploads directory and
    redirects to the audio player page with a confirmation message.
    """
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        flash(f"{filename} has been deleted.", "info")
    else:
        flash(f"{filename} not found.", "error")
    return redirect(url_for('audio_player'))


# ============================================================================
# FLASK ROUTES - FILE SERVING
# ============================================================================
# Routes to serve files
@app.route('/uploads/<filename>')
def serve_file(filename):
    """
    Serves an uploaded audio file.

    Args:
        filename (str): Name of the file to serve
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/separated/<filename>')
def serve_separated_file(filename):
    """
    Serves a separated audio file (from vocal separation).

    Args:
        filename (str): Name of the file to serve
    """
    return send_from_directory(SEPARATED_FOLDER, filename)


@app.route('/separated/htdemucs/<folder_name>/<audio_name>')
def serve_sep_file(folder_name, audio_name):
    """
    Serves a separated audio file (from music separation).

    Args:
        folder_name (str): Name of the folder containing the file
        audio_name (str): Name of the file to serve
    """
    return send_from_directory(f'separated/htdemucs/{folder_name}', audio_name)


# ============================================================================
# RUNNING APPLICATION
# ============================================================================
if __name__ == '__main__':
    app.run(debug=False)
