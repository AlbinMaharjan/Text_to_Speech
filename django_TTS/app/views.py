from .vocoder import text_to_speech_synthesis
from django.shortcuts import render
from django.http import JsonResponse
import os

# Path to your static folder
STATIC_AUDIO_PATH = 'app/static'

def index(request):
    return render(request, 'index.html')

def login(request):
    return render(request, 'login.html', {'status': request.session.get('status', False)})

def convert_text_to_speech(request):
    if request.method == 'POST':
        input_text = request.POST.get('text', '')
        
        # Validate input text
        if not input_text.strip():
            return JsonResponse({'success': False, 'message': 'Text input cannot be empty!'})

        try:
            # Call your TTS synthesis function
            text_to_speech_synthesis(input_text=input_text)
            
            # Assuming the audio is saved as `output.wav` in the static folder
            audio_file_name = 'output.wav'
            audio_file_path = os.path.join(STATIC_AUDIO_PATH, audio_file_name)
            
            # Ensure the file exists
            if not os.path.exists(audio_file_path):
                return JsonResponse({'success': False, 'message': 'Audio generation failed! File not found.'})

            return JsonResponse({'success': True, 'audio_file_path': f'/static/{audio_file_name}'})
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    return JsonResponse({'success': False, 'message': 'Invalid request method.'})
